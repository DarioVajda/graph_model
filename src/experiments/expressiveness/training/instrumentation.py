"""
Instrumentation for the real training runs: speed, memory, and graph sparsity.

Two independent concerns, intentionally kept apart:

  * **Speed + memory** are properties of the *training step*, so they are
    captured live from the real ``Trainer`` loop via :class:`StepMemCallback`
    (per-optimizer-step wall time, peak CUDA memory, and — see below — host RAM
    across the whole process tree).

  * **Token / block sparsity** are properties of the *data* (graph structure +
    ``k_hop`` + how the collator packs/pads) and do **not** depend on training,
    so :func:`measure_density` computes them once on a random subset of the
    dataset — never inside the training loop.

Both reuse existing machinery: ``transformers.TrainerCallback`` and
``src.models.flex_attn.density.compute_density`` (no new math here).

**Host RAM is measured, not inferred.** Until this file grew :class:`HostMemProbe`
every host-memory number in the project was a post-mortem ``sacct`` MaxRSS: one
scalar, after the fact, with no shape and no attribution. Three separate host-RAM
OOM kills were then "explained" by mechanisms nobody had measured (molecules
PLAN.md §8.4.9, kgqa TODO_cwq.md §"full-split eval", kgqa README.md §WebQSP+RRWP)
and each was answered by raising ``--mem`` rather than by a diagnosis. The probe
samples the three numbers that separate the candidates:

  * the trainer process's own ``VmRSS``/``VmHWM``, which is a resident dataset
    plus whatever the step loop accumulates;
  * the summed RSS of every descendant, which is what the dataloader worker
    fan-out costs (refcounting breaks copy-on-write, so a resident dataset is
    paid once per worker);
  * the **cgroup**'s ``memory.current`` / ``memory.peak``, which is the number the
    OOM killer actually compares against ``--mem`` — it counts shared pages once
    and includes the page cache and the loader's shared-memory segments, none of
    which a sum over per-process RSS gets right.

Everything degrades to ``None`` where ``/proc`` or the cgroup files are not
readable, so this is safe on any host.
"""

import json
import os
import statistics
import time
import random

import torch
from transformers import TrainerCallback

from ....utils import GraphCollatorV2
from ....models.flex_attn.density import compute_density


def _cuda_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


# ── Host RAM: this process, its descendants, and the cgroup that kills the job ───

_GB = 1024.0 ** 3      # /proc and cgroupfs report binary units; so does Slurm's --mem


def _read_first(paths):
    """First readable file among ``paths``, stripped — or ``None``."""
    for path in paths:
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            continue
    return None


def _status_kb(pid):
    """``{VmRSS, VmHWM, RssShmem}`` in kB from ``/proc/<pid>/status`` (``None`` if gone)."""
    wanted = ("VmRSS:", "VmHWM:", "RssShmem:")
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(wanted):
                    key, value = line.split(":", 1)
                    out[key + ":"] = int(value.split()[0])
                    if len(out) == len(wanted):
                        break
    except (OSError, ValueError, IndexError):
        return None
    return out or None


def _descendant_pids(root):
    """Every descendant pid of ``root``, by one pass over ``/proc``.

    ``/proc/<pid>/task/<tid>/children`` would be cheaper but needs
    ``CONFIG_PROC_CHILDREN``, which FRIDA's kernel does not carry, so this builds
    the parent map instead. One pass is a few milliseconds at the process counts
    involved, and it is called on a step stride rather than per step.
    """
    children = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                data = f.read()
            # comm can contain spaces and parentheses; ppid is the field after the
            # last ')'. Splitting the whole line would mis-index for e.g. "(a b)".
            ppid = int(data[data.rindex(")") + 1:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(name))
    found, stack = [], list(children.get(root, ()))
    while stack:
        pid = stack.pop()
        found.append(pid)
        stack.extend(children.get(pid, ()))
    return found


def _cgroup_paths():
    """Directory of this process's cgroup-v2 memory files, plus the v1 fallback.

    Inside the container the cgroup namespace usually makes ``/proc/self/cgroup``
    read ``0::/`` with the job's own cgroup mounted straight at
    ``/sys/fs/cgroup``, but an unnamespaced mount needs the path appended. Both
    are tried, and cgroup v1's differently-named files after them.
    """
    rel = ""
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    rel = parts[2]
                    break
    except OSError:
        pass
    roots = [f"/sys/fs/cgroup{rel}", "/sys/fs/cgroup"]
    return {
        "current": [f"{r}/memory.current" for r in roots]
                   + [f"{r}/memory/memory.usage_in_bytes" for r in roots],
        "peak": [f"{r}/memory.peak" for r in roots]
                + [f"{r}/memory/memory.max_usage_in_bytes" for r in roots],
        "limit": [f"{r}/memory.max" for r in roots]
                 + [f"{r}/memory/memory.limit_in_bytes" for r in roots],
        "stat": [f"{r}/memory.stat" for r in roots]
                + [f"{r}/memory/memory.stat" for r in roots],
    }


#: The ``memory.stat`` lines worth carrying. ``anon`` is the part that cannot be
#: reclaimed and therefore the part that actually triggers the OOM killer; ``file``
#: is page cache, which a memory-mapped Arrow feature store fills as a run walks the
#: corpus and which the kernel gives back under pressure. A peak dominated by
#: ``file`` and one dominated by ``anon`` need completely different fixes, and
#: ``memory.current`` alone cannot tell them apart.
_CGROUP_STAT_KEYS = ("anon", "file", "file_mapped", "shmem", "kernel", "pagetables", "slab")


class HostMemProbe:
    """One host-RAM reading: parent RSS, descendant RSS, and cgroup usage.

    Deliberately allocation-free and stdlib-only — no ``psutil`` dependency for
    what is four small reads under ``/proc``. Every field is ``None`` rather than
    an exception when the kernel does not expose it.
    """

    def __init__(self):
        self.pid = os.getpid()
        self._cgroup = _cgroup_paths()
        self.available = _status_kb(self.pid) is not None

    def cgroup_gb(self, key):
        """``"current"`` / ``"peak"`` / ``"limit"`` from cgroupfs, in GB (or ``None``)."""
        raw = _read_first(self._cgroup.get(key, ()))
        if raw is None or raw == "max":
            return None
        try:
            return int(raw) / _GB
        except ValueError:
            return None

    def cgroup_stat_gb(self):
        """``{anon, file, ...}`` from ``memory.stat``, in GB. Empty where absent."""
        raw = _read_first(self._cgroup.get("stat", ()))
        if not raw:
            return {}
        out = {}
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in _CGROUP_STAT_KEYS:
                try:
                    out["cg_" + parts[0] + "_gb"] = int(parts[1]) / _GB
                except ValueError:
                    continue
        return out

    def sample(self):
        """A flat dict of GB readings (see the module docstring for what each is)."""
        me = _status_kb(self.pid)
        kids = [_status_kb(p) for p in _descendant_pids(self.pid)]
        kids = [k for k in kids if k]
        kid_rss = sum(k.get("VmRSS:", 0) for k in kids)
        shmem = (me or {}).get("RssShmem:", 0) + sum(k.get("RssShmem:", 0) for k in kids)
        return {
            "self_rss_gb": (me["VmRSS:"] / 1024 ** 2) if me else None,
            "self_hwm_gb": (me["VmHWM:"] / 1024 ** 2) if me else None,
            "children_rss_gb": kid_rss / 1024 ** 2 if me else None,
            "tree_rss_gb": ((me["VmRSS:"] + kid_rss) / 1024 ** 2) if me else None,
            # Shared pages are charged to every process that maps them, so
            # `tree_rss_gb` double-counts them and the cgroup does not. Recording
            # the total makes that difference attributable instead of a mystery.
            "tree_shmem_gb": shmem / 1024 ** 2 if me else None,
            "n_children": len(kids),
            "cgroup_gb": self.cgroup_gb("current"),
            "cgroup_peak_gb": self.cgroup_gb("peak"),
            "cgroup_limit_gb": self.cgroup_gb("limit"),
            **self.cgroup_stat_gb(),
        }


# ── Live speed + memory from the real training loop ──────────────────────────────

class StepMemCallback(TrainerCallback):
    """Record per-optimizer-step wall time, peak CUDA memory, and host RAM.

    ``on_step_begin`` / ``on_step_end`` fire once per *optimizer* step (i.e. after
    gradient accumulation), so each timing is a full effective training step. The
    first step(s) are dropped in :meth:`summary` because they include flex's
    one-time ``torch.compile`` (seconds) rather than steady-state cost.

    Host RAM is sampled on a stride — every ``host_every_steps`` optimizer steps
    and every ``host_every_predict`` eval batches — plus unconditionally at
    ``train_begin``, at the end of every evaluation, at ``train_end``, and
    wherever the caller drops a :meth:`mark`. The stride keeps the cost off the
    step loop (a few ms per sample, ~400 samples in a six-hour run); the
    unconditional points are what make a load-time plateau, an eval-time sawtooth
    and a monotonic creep tell themselves apart in the trace.

    ``trace_path`` (JSONL, one sample per line) is where the shape lives; the
    peaks come back from :meth:`summary` so they land in ``runs.jsonl`` next to
    ``peak_gb``.
    """

    def __init__(self, trace_path=None, host_every_steps=25, host_every_predict=200):
        self.device = _cuda_device()
        self.step_ms = []
        self._t0 = None
        self.trace_path = trace_path
        self.host_every_steps = max(1, int(host_every_steps))
        self.host_every_predict = max(1, int(host_every_predict))
        self.host = HostMemProbe()
        self.host_samples = []
        self._wall0 = time.time()
        self._predict_calls = 0
        self._trace_failed = False
        if self.trace_path:
            try:
                os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)
            except OSError as exc:                            # noqa: BLE001
                print(f"[hostmem] cannot create trace directory ({exc}); "
                      "the trace will be kept in memory only.")
                self.trace_path = None

    # ── host sampling ────────────────────────────────────────────────────────
    def mark(self, event, state=None, **extra):
        """Take a host-RAM sample labelled ``event``. Never raises."""
        if not self.host.available:
            return None
        try:
            sample = self.host.sample()
        except Exception as exc:                              # noqa: BLE001
            print(f"[hostmem] sampling failed ({type(exc).__name__}: {exc}); disabling.")
            self.host.available = False
            return None
        sample = {"event": event, "t_s": round(time.time() - self._wall0, 1),
                  "step": getattr(state, "global_step", None),
                  "epoch": round(state.epoch, 3) if getattr(state, "epoch", None) else None,
                  "cuda_gb": (torch.cuda.memory_allocated(self.device) / _GB
                              if self.device.type == "cuda" else None),
                  **sample, **extra}
        self.host_samples.append(sample)
        if self.trace_path and not self._trace_failed:
            try:
                with open(self.trace_path, "a") as f:
                    f.write(json.dumps(sample) + "\n")
            except OSError as exc:                            # noqa: BLE001
                # A trace is measurement *about* the run and must never lose one.
                print(f"[hostmem] cannot append to {self.trace_path} ({exc}); "
                      "keeping the trace in memory only.")
                self._trace_failed = True
        return sample

    def _peak(self, key):
        values = [s[key] for s in self.host_samples if s.get(key) is not None]
        return max(values) if values else None

    # ── Trainer hooks ────────────────────────────────────────────────────────
    def on_train_begin(self, args, state, control, **kwargs):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        # The first reading of the run: datasets are loaded and the model is on the
        # GPU, but no step has run. Everything above this line in the trace is
        # resident payload; everything above it that later grows is not.
        self.mark("train_begin", state)

    def on_step_begin(self, args, state, control, **kwargs):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._t0 is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.step_ms.append((time.perf_counter() - self._t0) * 1000.0)
        n = len(self.step_ms)
        if n == 1 or n % self.host_every_steps == 0:
            self.mark("step", state)

    def on_prediction_step(self, args, state, control, **kwargs):
        self._predict_calls += 1
        if self._predict_calls % self.host_every_predict == 0:
            self.mark("predict", state, predict_batch=self._predict_calls)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.mark("evaluate_end", state)

    def on_train_end(self, args, state, control, **kwargs):
        self.mark("train_end", state)

    def summary(self, warmup=1):
        """Aggregate steady-state ms/step (dropping ``warmup`` steps) + peak GB.

        The ``host_*`` fields are the point of the trace: ``host_cgroup_peak_gb``
        is kernel-tracked rather than sampled, so it is the true peak against
        ``--mem`` no matter what the sampling stride missed, and
        ``host_rss_train_begin_gb`` is the resident payload the run starts from.
        """
        steady = self.step_ms[warmup:] or self.step_ms
        peak_gb = (torch.cuda.max_memory_allocated(self.device) / 1e9
                   if self.device.type == "cuda" else float("nan"))
        begin = next((s for s in self.host_samples if s["event"] == "train_begin"), None)
        cgroup_peak = self._peak("cgroup_peak_gb")
        if self.host.available:
            # `memory.peak` is monotonic, so read it once more at summary time:
            # a spike between the last sample and here is still in it.
            try:
                live = [v for v in (cgroup_peak, self.host.cgroup_gb("peak")) if v is not None]
                cgroup_peak = max(live) if live else None
            except Exception:                                 # noqa: BLE001
                pass
        return {
            "n_steps": len(self.step_ms),
            "step_ms_mean": statistics.mean(steady) if steady else float("nan"),
            "step_ms_median": statistics.median(steady) if steady else float("nan"),
            "peak_gb": peak_gb,
            # These two are not on the same footing and must not be differenced.
            # `self_hwm_gb` is `VmHWM`, a kernel high-water mark over the whole
            # process lifetime, so it can already be above anything this trace
            # sampled; `tree_rss_gb` is the current sum at sample time. A tree
            # peak *below* the self peak is therefore normal, not a defect.
            "host_rss_self_peak_gb": self._peak("self_hwm_gb"),
            "host_rss_tree_peak_gb": self._peak("tree_rss_gb"),
            "host_rss_train_begin_gb": begin.get("tree_rss_gb") if begin else None,
            "host_cgroup_peak_gb": cgroup_peak,
            "host_cgroup_limit_gb": self._peak("cgroup_limit_gb"),
            # Sampled, so these are lower bounds unlike `memory.peak` above — but
            # they are the split that decides what a peak means: anon is what the
            # OOM killer cannot reclaim, file is page cache it can.
            "host_cgroup_anon_peak_gb": self._peak("cg_anon_gb"),
            "host_cgroup_file_peak_gb": self._peak("cg_file_gb"),
            "host_mem_samples": len(self.host_samples),
            "host_mem_trace": self.trace_path if not self._trace_failed else None,
        }


# ── Standalone sparsity on a random subset (data property, not training) ─────────

def _density_of_batch(out, k_hop, block_size, device):
    """Run ``compute_density`` on a collated batch (moving the needed tensors)."""
    node_ids = out["node_ids"].to(device)
    prompt_node = out["prompt_node"].to(device)
    pad_mask = out["attention_mask"].to(device)
    k_hop_mask = out["k_hop_mask"].to(device) if out.get("k_hop_mask") is not None else None
    return compute_density(node_ids, prompt_node, pad_mask, k_hop_mask, k_hop,
                           block_size=block_size)


def measure_density(dataset, tokenizer, pad_token_id, magnetic_m, k_hop, k_hop_directed,
                    batch_size, num_sample_graphs=16, num_sample_batches=8,
                    block_size=128, seed=0, device=None):
    """Token + block sparsity for ``k_hop`` on a random subset of ``dataset``.

    * **Token (element) sparsity** — measured per graph with an *unpadded* v2
      collator (``pad_to_block=False``, batch=1 ⇒ ``L`` = real length), so it is a
      backend-independent property of the graph + ``k_hop``.
    * **Block sparsity** — measured on *flex-packed* batches (``pad_to_block=True``)
      of the real ``batch_size``: the fraction of 128×128 blocks flex must still
      compute (its realized skip rate). Reported as the flex-packing value; eager
      is dense and does not skip blocks.
    """
    device = device or _cuda_device()

    element_collator = GraphCollatorV2(
        tokenizer=tokenizer, pad_token_id=pad_token_id, k_hop=k_hop,
        k_hop_directed=k_hop_directed, magnetic_m=magnetic_m, pad_to_block=False)
    block_collator = GraphCollatorV2(
        tokenizer=tokenizer, pad_token_id=pad_token_id, k_hop=k_hop,
        k_hop_directed=k_hop_directed, magnetic_m=magnetic_m, pad_to_block=True)

    rng = random.Random(seed)
    n = len(dataset)

    # Token sparsity: per-graph, unpadded (L = real packed length).
    token_sps, unpadded_Ls = [], []
    for i in rng.sample(range(n), min(num_sample_graphs, n)):
        out = element_collator([dataset[i]])
        d = _density_of_batch(out, k_hop, block_size, device)
        token_sps.append(1.0 - d.element_density)
        unpadded_Ls.append(out["node_ids"].shape[1])

    # Block sparsity: flex-packed batches of the real batch_size (realized skip rate).
    block_sps, padded_Ls = [], []
    bs = min(batch_size, n)
    for _ in range(num_sample_batches):
        idx = rng.sample(range(n), bs)
        out = block_collator([dataset[i] for i in idx])
        d = _density_of_batch(out, k_hop, block_size, device)
        block_sps.append(1.0 - d.block_density)
        padded_Ls.append(out["node_ids"].shape[1])

    def _ms(xs):
        return (statistics.mean(xs) if xs else float("nan"),
                statistics.stdev(xs) if len(xs) > 1 else 0.0)

    tok_mean, tok_std = _ms(token_sps)
    blk_mean, blk_std = _ms(block_sps)
    return {
        "k_hop": k_hop,
        "token_sparsity_mean": tok_mean, "token_sparsity_std": tok_std,
        "block_sparsity_mean": blk_mean, "block_sparsity_std": blk_std,
        "unpadded_L_min": min(unpadded_Ls) if unpadded_Ls else None,
        "unpadded_L_max": max(unpadded_Ls) if unpadded_Ls else None,
        "padded_L_max": max(padded_Ls) if padded_Ls else None,
        "n_graphs": len(token_sps), "n_batches": len(block_sps),
    }


# ── Reporting ────────────────────────────────────────────────────────────────────

def format_results_table(rows):
    """Render a combined metrics table.

    Each row is a dict of pre-formatted strings with keys:
    ``config, accuracy, ms_step, peak_gb, token_sp, block_sp``.
    """
    cols = [
        ("config", "config", 20),
        ("accuracy", "accuracy", 16),
        ("ms_step", "ms/step", 14),
        ("peak_gb", "peak GB", 10),
        ("token_sp", "token sp", 14),
        ("block_sp", "block sp", 14),
    ]
    header = " | ".join(f"{h:<{w}}" for _, h, w in cols)
    lines = ["=" * len(header), header, "-" * len(header)]
    for r in rows:
        lines.append(" | ".join(f"{str(r.get(k, '')):<{w}}" for k, _, w in cols))
    lines.append("=" * len(header))
    return "\n".join(lines)

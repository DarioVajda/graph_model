"""Per-example WebQSP scoring, stratified by graph size — which mechanism sank arm N?

`032` closed negatively: arm N reaches 22.7% of WebQSP headroom against arm 2's
87.8%. The README attributes that to a capacity bottleneck — a 64-wide pool cannot
injectively summarize a row of `N` partners once `N` exceeds it. That hypothesis was
written against "up to 512 nodes". The built test split says otherwise:

    min 3   p25 35   median 61.5   mean 118.5   p90 336   max 512
    fraction of graphs above the 64-wide pool: 47.7%

Less than half the split is past the bound. If capacity were the whole story, the
other half should behave like GraphQA (where every graph is <= 21 nodes, 3x inside
the pool) and the aggregate deficit would be partial — not the 65 pp of headroom
actually lost. So node count alone does not account for the magnitude, and a second
mechanism is implicated.

The competing explanation is ROLE DEGENERACY, and it is N-independent: a pooled row
is a permutation-invariant marginal over partners, so two nodes with the same
neighbourhood profile receive the same vector at ANY width. Knowledge graphs are
saturated with such nodes by construction (many entities of one type hanging off one
CVT, interchangeable under the retrieved subgraph's structure), and WebQSP's task is
to pick WHICH one. `kgqa/README.md`'s "duplicate keys" mechanism is the same surface.

The two make opposite predictions on a split that ALREADY EXISTS — graph size varies
3 -> 512 within WebQSP at fixed dataset, task and sequence length, which is the
`N`-sweep the README lists as open (item 2) without needing to run one. This module
reads it off the trained checkpoints.

## The pre-registered reading

Written before the jobs land. `d(n)` is arm N's per-example F1 deficit against arm 2
on the same question, `n` the graph's node count, and the cut is the pool width, 64.

| observation | reading |
|---|---|
| `d` concentrated above n=64, small and flat below | **capacity.** The pool is mis-sized, not mis-designed. `d_struct` is then the lever — but see the README's own note that at `d_struct=256` the appended head width is 4x `head_dim` and the cost argument against arm 2's `2M=128` disappears, so this wins the science and loses the engineering. |
| `d` roughly flat in `n`, large even on the smallest graphs | **degeneracy.** A marginal cannot separate role-equivalent nodes at any width, so no `d_struct` rescues it. Closes the line for a reason no size sweep would have found, and kills tandem too (phase already carries pairs; the pooled channel adds a descriptor that is degenerate exactly where WebQSP needs resolution). |
| `d` grows with `n` but is already large at n<=64 | **both**, degeneracy dominant. Same verdict as above; capacity is an aggravator. |

The ablation carries a second, independent prediction. I argued that softmax pooling
is a FURTHER compression on top of an already-saturated pool: above the bound it is
useful selection (GraphQA paths: +2.67 pp, 3/3 seeds), below it, it discards partners
a mean would have kept. So:

    attn - uniform, per example, should DECREASE with n, and cross zero near n ~ 64.

If instead it is flat in `n`, the "softmax over-concentrates at scale" story is wrong
and the aggregate ablation numbers (-0.64 pp at 5e-3, -2.10 pp at 2e-2) need another
explanation.

## Why this is trustworthy

Every metric is computed by importing `kgqa.evaluate`'s own scoring primitives — not
reimplemented — and the per-example loop mirrors `generative_eval` line for line,
including its skip rule (no "Answer:" anchor, or no gold => the example is dropped
from the mean). `run` therefore reproduces the sweep's recorded aggregate as a
by-product, and CHECKS that it does: a reload that silently lost the graph bias would
otherwise make every arm look equally bad and manufacture the flat-in-`n` result that
argues for degeneracy. That failure has precedent in this project, so it is a hard
error, not a warning.

    python3 -m src.experiments.bias_experiments.nonlinear_bias.stratified_eval submit [--dry-run]
    python3 -m src.experiments.bias_experiments.nonlinear_bias.stratified_eval report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import statistics as st
import subprocess
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SWEEP = "036_stratified_eval"
OUT_ROOT = os.path.join(REPO, "src/experiments/bias_experiments/nonlinear_bias/results", SWEEP)
CONTAINER = "/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh"

# The pool width, and therefore the pre-registered cut for the size split.
POOL_WIDTH = 64
BINS = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 513)]


# --------------------------------------------------------------------------- #
# Which runs
# --------------------------------------------------------------------------- #
# Three arms at bias_lr=5e-3, 3 seeds each. 5e-3 is the only LR at which all
# three were measured, and it is arm N's best WebQSP cell (22.7% headroom) and
# the LR arm 2's 87.8% was measured at — so the comparison is like-for-like and
# maximally favourable to the arm under test.
#
# 021 and 032 share every data key (question_node, max_nodes, versions, n_max,
# data_seed, dfv3, rel_mode, magnetic_m, max_spd), differing only in
# `magnetic_dim` which is model-side and NOT in the dataset cache key. So all
# three arms score the identical test split and per-example indices align,
# which is what makes the paired deltas below meaningful.
ARMS = [
    ("arm2-linear", "src/experiments/bias_experiments/mixed_bias/results/021_webqsp_baselines",
     lambda r: "magnetic_linearTrue" in r and "bias_lr0.005" in r),
    ("N-uniform", "src/experiments/bias_experiments/nonlinear_bias/results/032_webqsp_nonlinear",
     lambda r: "pooluniform" in r and "bias_lr0.005" in r),
    ("N-attn", "src/experiments/bias_experiments/nonlinear_bias/results/032_webqsp_nonlinear",
     lambda r: "poolattn" in r and "bias_lr0.005" in r),
]

# Runner bookkeeping — meaningless outside the original sweep, and --runs-jsonl
# would append a bogus record to a closed sweep's log if replayed.
_RUNNER_FLAGS = {"--runs-jsonl", "--run-name", "--sweep-id", "--resume-from"}


def discover():
    """[(arm, sweep_id, run_name, job_script, checkpoint_dir, runs_jsonl)] for every selected run."""
    out = []
    for arm, results_dir, match in ARMS:
        sweep_id = os.path.basename(results_dir)
        jobs = sorted(glob.glob(os.path.join(REPO, results_dir, "jobs", "*.sh")))
        if not jobs:
            raise SystemExit(f"no job scripts under {results_dir}/jobs")
        for job in jobs:
            run_name = os.path.basename(job)[len(sweep_id) + 1:-3]
            if not match(run_name):
                continue
            ck_run = os.path.join(REPO, "checkpoints/kgqa", f"{sweep_id}_{run_name}")
            if not os.path.isdir(ck_run):
                raise SystemExit(f"missing checkpoint dir {ck_run}")
            out.append((arm, sweep_id, run_name, job, ck_run,
                        os.path.join(REPO, results_dir, "runs.jsonl")))
    return out


def latest_checkpoint(run_dir):
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda q: int(q.rsplit("-", 1)[1]))
    if not cks:
        raise SystemExit(f"no checkpoint-* under {run_dir}")
    # save_total_limit=1 + load_best_model_at_end=True => the survivor is the
    # BEST checkpoint, which is the one the recorded metrics were scored on.
    # The aggregate check in `run` is what actually confirms that.
    return cks[-1]


# --------------------------------------------------------------------------- #
# Replaying a run's config
# --------------------------------------------------------------------------- #
def flags_from_job_script(path):
    """The kgqa CLI flags of a sweep-generated job script, minus runner bookkeeping.

    Replaying the ORIGINAL flags (rather than re-deriving a config) is what
    guarantees this scores the same dataset build the run trained on: every data
    cache key is a function of these flags.
    """
    line = None
    with open(path) as f:
        for raw in f:
            if raw.strip().startswith("python") and "src.experiments.kgqa" in raw:
                line = raw.strip()
                break
    if line is None:
        raise SystemExit(f"no `python -m src.experiments.kgqa` line in {path}")

    toks = shlex.split(line)
    toks = toks[toks.index("src.experiments.kgqa") + 1:]
    out, i = [], 0
    while i < len(toks):
        if toks[i] in _RUNNER_FLAGS:
            i += 2                       # flag + its value
            continue
        out.append(toks[i])
        i += 1
    return out


def config_from_job_script(path):
    from src.experiments.kgqa.__main__ import build_parser, config_from_args
    return config_from_args(build_parser().parse_args(flags_from_job_script(path)))


def recorded_test_f1(runs_jsonl, run_name):
    """The `test_webqsp_f1` this run logged, for the reload check."""
    if not os.path.exists(runs_jsonl):
        return None
    for raw in open(runs_jsonl):
        if not raw.strip():
            continue
        rec = json.loads(raw)
        if rec.get("sweep_run") == run_name:
            return rec.get("test_webqsp_f1")
    return None


# --------------------------------------------------------------------------- #
# The per-example loop
# --------------------------------------------------------------------------- #
@torch.no_grad()
def per_example_records(model, dataset, tokenizer, collator, question_end,
                        max_new_tokens, answer_sep, device, limit=None):
    """Mirror of `evaluate.generative_eval`, emitting one record per scored example.

    Deliberately NOT a refactor of that function: it stays untouched so this
    cannot perturb the training path, and the primitives are imported from it so
    the two cannot drift on scoring. The skip rule is identical, so the mean of
    these records is the aggregate that function would have returned.
    """
    from src.experiments.kgqa.evaluate import (
        eval_f1, eval_hit1, eval_hit, parse_answer_list, _find_prefix_len, eval_indices)

    was_training = model.training
    model.eval()
    # Generation batches are unbucketed, so the flex path cannot serve the
    # prefill; generative_eval makes the same swap for the same reason.
    impl = getattr(model.config, "graph_attn_impl", None)
    if impl == "flex":
        model.config.graph_attn_impl = "eager"

    records = []
    for i in eval_indices(len(dataset), None):
        item = dataset[i]
        pn = int(item["prompt_node"])
        ids = list(item["input_ids"][pn])
        cut = _find_prefix_len(ids, question_end)
        gold = [a for a in dataset.graphs[i].graph.get("gold_answers", []) if a]
        if cut is None or not gold:
            continue

        gen_item = dict(item)
        gen_item["input_ids"] = [list(x) for x in item["input_ids"]]
        gen_item["input_ids"][pn] = ids[:cut]
        gen_item.pop("labels", None)

        batch = collator([gen_item])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_answer_list(text, sep=answer_sep)

        if pred:
            f1 = float(eval_f1(pred, gold)[0])
            hits1 = float(eval_hit1(pred, gold))
            hitstar = float(eval_hit(pred, gold))
        else:
            f1 = hits1 = hitstar = 0.0

        records.append({
            "i": i,
            "num_nodes": int(item["num_nodes"]),
            "f1": f1, "hits1": hits1, "hit_star": hitstar,
            "n_gold": len(gold), "n_pred": len(pred),
        })
        if len(records) % 200 == 0:
            print(f"  [{len(records)}] running F1 "
                  f"{st.mean(r['f1'] for r in records):.4f}", flush=True)
        if limit and len(records) >= limit:
            break

    if impl == "flex":
        model.config.graph_attn_impl = impl
    if was_training:
        model.train()
    return records


# Parameters that are EXACTLY ZERO at initialisation and can only be non-zero if
# the checkpoint's graph bias was restored. This is the direct reload check, and
# it is the discriminating one: on a failed reload `from_pretrained` leaves the
# constructor's init, under which W_val/W_attn are randomly non-zero and would
# pass a naive "some bias parameter is non-zero" test. These would not.
_TRAINED_FROM_ZERO = {
    "gamma_in": "magnetic_nonlinear",       # zero-init gain; bias is identically 0 at 0
    "proj.0.weight": "magnetic_linear",     # zeros_ at init (LinearMagneticBias)
}


def assert_bias_loaded(model):
    """Fail unless a zero-at-init graph-bias parameter came back non-zero.

    The F1 check below is the end-to-end version of this, but it only fires after
    a full generation pass; this catches the same failure in one second, and it
    still fires under --limit where the aggregate cannot be compared.
    """
    found = {}
    for name, p in model.named_parameters():
        for suffix, arm in _TRAINED_FROM_ZERO.items():
            if name.endswith(suffix) and "graph_bias" in name:
                found.setdefault(arm, []).append(float(p.detach().float().abs().max()))
    if not found:
        raise SystemExit(
            "FATAL: no zero-at-init graph-bias parameter found in the loaded model "
            f"(looked for {sorted(_TRAINED_FROM_ZERO)}). Cannot verify the reload.")
    for arm, mags in found.items():
        hi = max(mags)
        print(f"[stratified] reload check: {arm} — {len(mags)} zero-at-init tensors, "
              f"max|.| = {hi:.6f}", flush=True)
        if hi == 0.0:
            raise SystemExit(
                f"FATAL: every {arm} zero-at-init tensor is still exactly 0.0 — the "
                f"graph bias did NOT reload. This run would score at the no-bias "
                f"floor and read as a flat-in-n deficit, which is the conclusion "
                f"this experiment exists to test. Refusing to continue.")


def cmd_run(a):
    """Score ONE checkpoint per-example and write its JSONL."""
    from transformers import AutoTokenizer
    from src.models import GTLMLlamaForCausalLM
    from src.utils import GraphCollatorV2
    from src.experiments.kgqa.load_data import load_data
    from src.train import get_device

    cfg = config_from_job_script(a.job_script)
    ck = latest_checkpoint(a.checkpoint_run)
    device = get_device()
    print(f"[stratified] arm={a.arm} ck={ck}\n[stratified] device={device}", flush=True)

    torch.set_num_threads(1)
    model = GTLMLlamaForCausalLM.from_pretrained(
        ck, graph_attn_impl="eager", torch_dtype=cfg.torch_dtype)
    model.to(device).eval()
    assert_bias_loaded(model)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    question_end = tokenizer(cfg.question_end_str, add_special_tokens=False)["input_ids"]

    _, _, test_sets = load_data(cfg)
    dataset = test_sets["webqsp"]
    print(f"[stratified] test split: {len(dataset)} graphs", flush=True)

    # train.py's gen_collator, verbatim: unbucketed, since the dense decode path
    # cannot use flex's block alignment.
    collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        magnetic_m=cfg.collate_magnetic_m, pad_to_block=False,
        node_position_mode=cfg.node_position_mode, max_spd=cfg.max_spd)

    records = per_example_records(
        model, dataset, tokenizer, collator, question_end,
        max_new_tokens=cfg.gen_max_new_tokens, answer_sep=cfg.answer_parse_sep,
        device=device, limit=a.limit)

    got = st.mean(r["f1"] for r in records)
    want = recorded_test_f1(a.runs_jsonl, a.run_name)
    print(f"[stratified] {len(records)} scored   mean F1 {got:.4f}   "
          f"recorded {want if want is None else f'{want:.4f}'}", flush=True)

    if a.limit:
        # A prefix of the split is not the split; the aggregate is not comparable.
        # The smoke job exists to prove the path runs at all — assert_bias_loaded
        # above is what it actually gates on.
        print(f"[stratified] --limit {a.limit}: SMOKE ONLY, aggregate check skipped.",
              flush=True)
        return 0

    # THE gate. A silently-failed bias reload lands every arm near the no-bias
    # floor, which would read as "the deficit is flat in n" — i.e. it would
    # manufacture the degeneracy conclusion out of a loading bug. Fail loudly.
    if want is not None and abs(got - want) > a.tol:
        raise SystemExit(
            f"FATAL: reproduced test F1 {got:.4f} but the run recorded {want:.4f} "
            f"(|delta| {abs(got - want):.4f} > tol {a.tol}). The checkpoint did not "
            f"reload as trained — most likely the graph bias. Every number from this "
            f"job would be an artifact; refusing to write it.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(json.dumps({
            "kind": "meta", "arm": a.arm, "run_name": a.run_name,
            "checkpoint": os.path.relpath(ck, REPO),
            "n_scored": len(records), "mean_f1": got, "recorded_f1": want,
        }) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[stratified] wrote {a.out}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Submission
# --------------------------------------------------------------------------- #
def cmd_submit(a):
    runs = discover()
    jobs_dir, logs_dir = os.path.join(OUT_ROOT, "jobs"), os.path.join(OUT_ROOT, "logs")
    per_ex = os.path.join(OUT_ROOT, "per_example")
    for d in (jobs_dir, logs_dir, per_ex):
        os.makedirs(d, exist_ok=True)

    def write_job(arm, sweep_id, run_name, job_script, ck_run, runs_jsonl,
                  suffix="", limit=None):
        label = f"{sweep_id}_{run_name}{suffix}"
        out_jsonl = os.path.join(per_ex, f"{arm}__{sweep_id}_{run_name}.jsonl")
        sh = os.path.join(jobs_dir, f"{label}.sh")
        with open(sh, "w") as f:
            f.write("#!/usr/bin/env bash\nset -uo pipefail\n"
                    "python -m src.experiments.bias_experiments.nonlinear_bias.stratified_eval run"
                    f" --arm {shlex.quote(arm)}"
                    f" --run-name {shlex.quote(run_name)}"
                    f" --job-script {shlex.quote(os.path.relpath(job_script, REPO))}"
                    f" --checkpoint-run {shlex.quote(os.path.relpath(ck_run, REPO))}"
                    f" --runs-jsonl {shlex.quote(os.path.relpath(runs_jsonl, REPO))}"
                    f" --out {shlex.quote(os.path.relpath(out_jsonl, REPO))}"
                    + (f" --limit {limit}" if limit else "") + "\n")
        os.chmod(sh, 0o755)
        return label, os.path.relpath(sh, REPO)

    labels, scripts = [], []
    for r in runs:
        lbl, sh = write_job(*r)
        labels.append(lbl)
        scripts.append(sh)

    # The smoke job: one checkpoint, a handful of examples. It proves the whole
    # path — checkpoint reload, bias restore, dataset resolve, generate, score,
    # write — before nine jobs queue behind a defect they would all share. This
    # is the README's own discipline ("an end-to-end 6-step WebQSP run on the
    # sweep's exact rendered flags ... before any array was submitted"), and the
    # array is submitted --dependency=afterok on it so a failure costs one job.
    #
    # It runs a `magnetic_nonlinear` checkpoint deliberately: that head is the one
    # with a from_pretrained defect in its history (the 030/031 void), so it is
    # the reload most worth proving before nine jobs depend on it.
    smoke_run = next((r for r in runs if r[0].startswith("N-")), runs[0])
    smoke_label, smoke_script = write_job(*smoke_run, suffix="__SMOKE", limit=a.smoke)

    # A dispatcher on disk rather than a --wrap one-liner: the sweep's generated
    # sbatch_commands.sh nests four levels of quoting to do this inline, which is
    # unreadable and easy to get subtly wrong when hand-built.
    def write_dispatch(name, labels_, scripts_, array):
        path = os.path.join(jobs_dir, name)
        with open(path, "w") as f:
            f.write("#!/usr/bin/env bash\nset -uo pipefail\n"
                    f"cd {shlex.quote(REPO)}\n"
                    f"LABELS=({' '.join(shlex.quote(x) for x in labels_)})\n"
                    f"SCRIPTS=({' '.join(shlex.quote(x) for x in scripts_)})\n"
                    + ('i="${SLURM_ARRAY_TASK_ID:?not an array job}"\n' if array else "i=0\n")
                    + f"exec srun --container-image={CONTAINER} --container-mounts=/shared:/shared \\\n"
                    f"  env HOME={shlex.quote(os.path.expanduser('~'))} PYTHONUNBUFFERED=1 \\\n"
                    f"      SWEEP_PROJECT_ROOT={shlex.quote(REPO)} \\\n"
                    f"      SWEEP_VENV_BIN={shlex.quote(os.path.join(REPO, '.venv/bin'))} \\\n"
                    f"      SWEEP_LOGIN={shlex.quote(os.path.join(REPO, 'login.sh'))} \\\n"
                    f"  bash {shlex.quote(os.path.join(REPO, 'sweep/slurm_launch.sh'))}"
                    ' "${LABELS[$i]}" "${SCRIPTS[$i]}"\n')
        os.chmod(path, 0o755)
        return path

    dispatch = write_dispatch("dispatch.sh", labels, scripts, array=True)
    smoke_dispatch = write_dispatch("dispatch_smoke.sh", [smoke_label], [smoke_script],
                                    array=False)

    constraint = "|".join(f"GPU_BRD:{g}" for g in a.gpus.split("|"))
    # A feature constraint, not a gres TYPE — sbatch_tests.sh documents why the
    # gres form silently excludes nodes whose gres registers as e.g. A100_80GB.
    base = ["sbatch", "--parsable", "-p", a.partition, "-A", "povejmo",
            "--gres", "gpu:1", "--constraint", constraint, "-c", "8", "--mem", "128G"]
    logs_rel = os.path.relpath(logs_dir, REPO)

    smoke_cmd = base + ["-t", "00:30:00", "-J", f"kgqa_{SWEEP}_smoke",
                        "-o", os.path.join(logs_rel, f"{SWEEP}_smoke_%j.slurm.out"),
                        smoke_dispatch]
    array_cmd = base + ["-t", a.time, "-J", f"kgqa_{SWEEP}",
                        "-o", os.path.join(logs_rel, f"{SWEEP}_%A_%a.slurm.out"),
                        "--array", f"0-{len(labels) - 1}%{len(labels)}"]

    print(f"smoke ({a.smoke} examples, gates the array):\n  "
          + " ".join(shlex.quote(c) for c in smoke_cmd))
    print("\narray (9 runs, submitted --dependency=afterok on the smoke):\n  "
          + " ".join(shlex.quote(c) for c in array_cmd + ["<dispatch>"]) + "\n")
    for lbl in labels:
        print(f"  {lbl}")
    if a.dry_run:
        print("\n--dry-run: wrote job scripts, did not submit.")
        return 0

    def submit(cmd):
        res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout + res.stderr, file=sys.stderr)
            raise SystemExit("sbatch failed")
        # --parsable prints "jobid[;cluster]", but this cluster's sbatch also
        # emits backfill chatter ("start 6 days from now, ...") on stdout, so
        # .strip() is not the job id. Take the first all-digit token, exactly as
        # sbatch_tests.sh does with `grep -oE '^[0-9]+$'`.
        for line in res.stdout.splitlines():
            head = line.strip().split(";")[0]
            if head.isdigit():
                return head
        print(res.stdout + res.stderr, file=sys.stderr)
        raise SystemExit("sbatch returned no parsable job id")

    smoke_job = a.depend or submit(smoke_cmd)
    array_job = submit(array_cmd + [f"--dependency=afterok:{smoke_job}", dispatch])

    print(f"\nsubmitted smoke job {smoke_job}  ({a.smoke} examples)")
    print(f"submitted array job {array_job}  ({len(labels)} runs, held until the smoke passes)")
    print(f"  smoke log: {logs_rel}/{SWEEP}_smoke_{smoke_job}.slurm.out")
    print(f"  array log: {logs_rel}/{SWEEP}_{array_job}_*.slurm.out")
    print(f"  results:   {os.path.relpath(per_ex, REPO)}/")
    print(f"  then:      python3 -m src.experiments.bias_experiments.nonlinear_bias.stratified_eval report")
    return 0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _load_per_example():
    """{arm: {run_name: {i: record}}} over whatever has landed."""
    out = {}
    for p in sorted(glob.glob(os.path.join(OUT_ROOT, "per_example", "*.jsonl"))):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        meta = rows[0]
        out.setdefault(meta["arm"], {})[meta["run_name"]] = {
            r["i"]: r for r in rows[1:]}
    return out


def _bin_of(n):
    for lo, hi in BINS:
        if lo <= n < hi:
            return f"{lo}-{hi - 1}"
    return f"{BINS[-1][0]}+"


def cmd_report(a):
    data = _load_per_example()
    if not data:
        print(f"nothing under {os.path.relpath(OUT_ROOT, REPO)}/per_example — "
              f"has the array finished?")
        return 1

    for arm in data:
        n = {r: len(v) for r, v in data[arm].items()}
        print(f"{arm}: {len(data[arm])} runs, {n}")
    print()

    # Seed-averaged per-example F1, per arm: mean over that arm's runs for the
    # examples every one of its runs scored.
    avg = {}
    for arm, runs in data.items():
        common = set.intersection(*(set(v) for v in runs.values()))
        avg[arm] = {i: st.mean(runs[r][i]["f1"] for r in runs) for i in common}
    nodes = {}
    for arm, runs in data.items():
        for r in runs.values():
            for i, rec in r.items():
                nodes[i] = rec["num_nodes"]

    arms = [x for x in ("arm2-linear", "N-uniform", "N-attn") if x in avg]
    common = sorted(set.intersection(*(set(avg[x]) for x in arms)))
    print(f"{len(common)} examples scored by every arm\n")

    def table(rows, title, cols):
        print(title)
        head = f"{'bin (nodes)':>14} {'n':>5} " + " ".join(f"{c[0]:>13}" for c in cols)
        print(head + "\n" + "-" * len(head))
        for label, idx in rows:
            if not idx:
                continue
            cells = []
            for c in cols:
                vals = [c[1](i) for i in idx]
                cells.append(f"{st.mean(vals):13.4f}")
            print(f"{label:>14} {len(idx):5d} " + " ".join(cells))
        print()

    groups = {}
    for i in common:
        groups.setdefault(_bin_of(nodes[i]), []).append(i)
    ordered = [(f"{lo}-{hi - 1}", groups.get(f"{lo}-{hi - 1}", [])) for lo, hi in BINS]

    table(ordered, "Per-example F1 by graph size (seed-averaged)",
          [(a_, (lambda i, a_=a_: avg[a_][i])) for a_ in arms])

    if "arm2-linear" in avg:
        for a_ in arms:
            if a_ == "arm2-linear":
                continue
            table(ordered, f"DEFICIT  {a_} - arm2-linear  (pp of F1)",
                  [(a_, (lambda i, a_=a_: 100 * (avg[a_][i] - avg["arm2-linear"][i])))])

    if "N-attn" in avg and "N-uniform" in avg:
        table(ordered, "ABLATION  N-attn - N-uniform  (pp of F1)",
              [("attn-uniform", lambda i: 100 * (avg["N-attn"][i] - avg["N-uniform"][i]))])

    # The pre-registered cut.
    small = [i for i in common if nodes[i] <= POOL_WIDTH]
    large = [i for i in common if nodes[i] > POOL_WIDTH]
    print(f"The pre-registered cut at the pool width (n <= {POOL_WIDTH} vs n > {POOL_WIDTH}):")
    print(f"  {'':>14} {'n<=64':>10} {'n>64':>10} {'delta':>10}")
    for a_ in arms:
        s, l = st.mean(avg[a_][i] for i in small), st.mean(avg[a_][i] for i in large)
        print(f"  {a_:>14} {s:10.4f} {l:10.4f} {l - s:10.4f}")
    for a_ in arms:
        if a_ == "arm2-linear":
            continue
        ds = 100 * st.mean(avg[a_][i] - avg["arm2-linear"][i] for i in small)
        dl = 100 * st.mean(avg[a_][i] - avg["arm2-linear"][i] for i in large)
        print(f"  deficit {a_:>6} vs arm2: {ds:+.2f} pp (small)  {dl:+.2f} pp (large)  "
              f"-> {'CAPACITY' if ds > -3 and dl < ds - 3 else 'FLAT/DEGENERACY'}")
    print(f"\n  ({len(small)} small / {len(large)} large; the split is near-even by "
          f"construction — median WebQSP graph is ~61 nodes.)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="stratified_eval", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="score ONE checkpoint per-example (runs on the GPU node)")
    r.add_argument("--arm", required=True)
    r.add_argument("--run-name", required=True)
    r.add_argument("--job-script", required=True)
    r.add_argument("--checkpoint-run", required=True)
    r.add_argument("--runs-jsonl", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tol", type=float, default=0.02,
                   help="max |reproduced - recorded| test F1 before the run is "
                        "rejected as a bad reload (the no-bias floor is 6+ pp below).")
    r.add_argument("--limit", type=int, default=None,
                   help="score only the first N examples (smoke only — the "
                        "aggregate check is skipped and nothing is written).")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("submit", help="submit the smoke gate, then the array behind it")
    s.add_argument("--partition", default="frida")
    s.add_argument("--gpus", default="B200|B300")
    s.add_argument("--time", default="02:00:00")
    s.add_argument("--smoke", type=int, default=8,
                   help="examples for the gating smoke job (0 = skip the gate).")
    s.add_argument("--depend", default=None, metavar="JOBID",
                   help="reuse an already-submitted smoke job as the array's "
                        "afterok dependency instead of submitting a new one.")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_submit)

    q = sub.add_parser("report", help="stratify the landed per-example files")
    q.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

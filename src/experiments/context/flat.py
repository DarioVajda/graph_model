"""
Flat-text arm: the same needle, serialized as one ordinary token sequence.

The README §3.1 grid came back at ceiling in all 25 cells, which by itself cannot separate
"GTLM is robust to dilution" from "this task is trivial". Worse, the graph arm
could not test length at all: **per-node RoPE reset caps position ids at T**
(README §1), so a 64,640-token cell is 126 segments each positioned 0..511.

Serializing the same graph flat is therefore not a baseline but the missing
measurement — flat positions run 0..64,639 for real. If flat degrades where the
graph arm holds, per-node position reset buys length robustness that flat
serialization lacks; if flat also holds, the task is trivial for any LM.

Design notes:

  * **No data rebuild.** ``.gtds`` already stores per-node ``text``, so this is a
    collate-time transformation of the very same 25 test splits — the flat and
    graph arms score literally the same items.
  * **Shuffled order.** Content nodes are emitted in an order drawn from a
    per-graph seeded RNG, so the gold node's position carries no information and
    the arms are not separated by a spurious ordering cue. Mirrors the kgqa flat
    arm's ``flat_shuffle_lines``.
  * **Re-tokenized, not concatenated.** Joining the stored per-node ``input_ids``
    would not equal tokenizing the joined string (BPE merges the preceding space
    into a following token), and a flat-text arm must be what the tokenizer
    actually produces for flat text.
  * ``grid_eval`` is reused unchanged, so EM and the distractor/malformed
    decomposition are computed by the same code that produced README §3.1.
"""

import numpy as np
import torch
from transformers import Trainer

from ...train import get_device

from .data import ANSWER_PREFIX
from .evaluate import grid_eval, wilson_interval, windowed_forward, windowed_loss
from .process_dataset import cell_split_name, load_split
from ._io import append_jsonl

# The node separator. A numbered header rather than the node's own id, because the
# id already appears inside the node's KV sentence and repeating it there would
# hand the model a second copy of the very string it has to match. Letters+digits
# only in a "Ll Dd" shape, which cannot collide with a CODE_TEMPLATE ("LLDLL")
# code — see data.build_code_pool.
ARTICLE_HEADER = "## Article {i}"


def serialize_graph(g, order):
    """Render one graph as flat markdown: shuffled content nodes, then the QUESTION.

    Returns the text UP TO AND INCLUDING ``ANSWER_PREFIX``; the gold code is
    appended by the caller so the supervised span stays exactly ``code_len + 1``
    tokens (code + EOS), identical to the graph arm.
    """
    qn, pn = g.graph["question_node"], g.graph["prompt_node"]
    parts = []
    for i, node in enumerate(order):
        parts.append(ARTICLE_HEADER.format(i=i + 1))
        parts.append(g.nodes[node]["text"])
        parts.append("")
    parts.append(g.nodes[qn]["text"])
    parts.append(ANSWER_PREFIX)
    return "\n".join(parts)


def content_order(g, seed):
    """Deterministic shuffled order of the content nodes of ``g``."""
    qn, pn = g.graph["question_node"], g.graph["prompt_node"]
    nodes = [k for k in g.nodes if k not in (qn, pn)]
    rng = np.random.default_rng(seed)
    rng.shuffle(nodes)
    return nodes


class FlatCollator:
    """Collate ``TextGraphDataset`` items into FLAT causal-LM batches.

    Emits only ``input_ids`` / ``attention_mask`` / ``labels`` — no graph tensors —
    so the batch feeds a stock ``AutoModelForCausalLM``. All but the final
    ``code_len + 1`` positions are masked to -100, matching the graph arm's
    supervised span exactly.

    Batches are built one graph at a time (``batch_size=1`` in ``grid_eval``);
    with more than one row the shorter rows are left-padded, which keeps the
    supervised span at the end of every row where the label mask expects it.
    """

    def __init__(self, tokenizer, code_len, data_seed=42):
        self.tokenizer = tokenizer
        self.code_len = code_len
        self.data_seed = data_seed
        self.n_supervised = code_len + 1        # code tokens + EOS

    def _row(self, graph, index):
        order = content_order(graph, self.data_seed + index)
        prefix = serialize_graph(graph, order)
        full = f"{prefix} {graph.graph['gold_code']}"
        ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        ids = ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(ids)
        labels[-self.n_supervised:] = ids[-self.n_supervised:]
        return ids, labels

    def __call__(self, items):
        """``items`` carry no text, so the graphs are attached by ``FlatCellView``."""
        rows = [self._row(it["_graph"], it["_index"]) for it in items]
        width = max(len(ids) for ids, _ in rows)
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        input_ids, attn, labels = [], [], []
        for ids, lab in rows:
            gap = width - len(ids)
            input_ids.append([pad] * gap + ids)
            attn.append([0] * gap + [1] * len(ids))
            labels.append([-100] * gap + lab)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class FlatCellView:
    """Adapts a built split to what ``grid_eval`` + ``FlatCollator`` expect.

    ``grid_eval`` indexes ``dataset[i]`` for the collator and reads
    ``dataset.graphs[i].graph`` for the gold/distractor codes. The flat arm needs
    the graph itself at collate time, so items carry it (and their index, which
    seeds the shuffle) rather than the graph tensors the graph arm uses.
    """

    def __init__(self, split):
        self._split = split
        self.graphs = split.graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return {"_graph": self.graphs[i], "_index": i}


def build_flat_model(cfg, device=None):
    """The PRETRAINED backbone, unmodified — no LoRA, no graph bias, no adapter.

    Stage 0 scores the base model as-is. Teacher-forced EM means it faces no
    format barrier: the answer tokens are in the input and it only has to place
    the argmax on them.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=cfg.torch_dtype, attn_implementation="sdpa")
    model.to(device)
    model.eval()
    return model, tokenizer


def run_flat_grid_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None):
    """Stage 0 — score the pretrained model, flat-serialized, over all 25 cells."""
    import os
    grid_jsonl = runs_jsonl or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "flat_grid.jsonl")
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name

    torch.set_num_threads(1)
    device = get_device()
    model, tokenizer = build_flat_model(cfg, device)
    collator = FlatCollator(tokenizer, cfg.code_len, data_seed=cfg.data_seed)

    train_cells = set(cfg.train_cells())
    # Largest first: if the top cell already holds, there is no contour (README §A.12).
    cells = sorted(cfg.cells(), key=lambda c: cfg.cell_length(*c), reverse=True)

    # Under a k mixture the grid is a 3-axis product: a cell contributes one record
    # per k, not one in total. `hops` goes into every record either way, so single-k
    # and mixture grids stay readable by the same analysis code.
    mixed = bool(cfg.hop_counts)
    records = []
    for (n, t) in cells:
      for hops in cfg.hops_list():
        split = load_split(cfg, cell_split_name(n, t, hops if mixed else None))
        view = FlatCellView(split)
        metrics = grid_eval(model, view, collator, tokenizer, device=device,
                            batch_size=1, verbose=True)
        lo, hi = wilson_interval(round(metrics["em"] * metrics["n"]), metrics["n"])
        record = {
            "mode": "flat_grid",
            "arm": "flat_text",
            **sweep_meta,
            "checkpoint_path": None,
            "pretrained": True,
            "n_nodes": n, "tokens_per_node": t, "hops": hops,
            "expected_len": cfg.cell_length(n, t),
            "in_train_distribution": (n, t) in train_cells,
            "max_train_len": cfg.max_train_len,
            "em_ci_low": lo, "em_ci_high": hi,
            "seed": cfg.seed, "data_seed": cfg.data_seed,
            **metrics,
        }
        append_jsonl(grid_jsonl, record)
        records.append(record)
        print(f"[flat] N={n:4d} T={t:4d}  L={metrics['packed_len']:6d}  "
              f"EM={metrics['em']:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"distractor={metrics['distractor_rate']:.3f}  "
              f"malformed={metrics['malformed_rate']:.3f}"
              f"{'' if (n, t) in train_cells else '   (extrapolation)'}")
        del split, view

    print(f"\n[flat] wrote {len(records)} cell records to {grid_jsonl}")
    return records


# ── Stage 1: the TRAINED flat arm ──────────────────────────────────────────────

class FlatTrainer(Trainer):
    """Stock ``Trainer`` with the windowed loss/eval, for a plain causal LM.

    ``ContextGraphTrainer`` cannot be reused: it extends ``GraphTrainerV2``, whose
    bias-parameter save/restore and graph-tensor handling have no meaning for an
    arm with no graph bias at all. The windowed forward is shared, so the two arms
    compute loss and EM by identical code over identical spans.
    """

    def __init__(self, *args, train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler

    def _get_train_sampler(self, *args, **kwargs):
        if self._train_sampler is not None:
            return self._train_sampler
        return super()._get_train_sampler(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        logits, window_labels = windowed_forward(model, inputs, labels)
        loss = windowed_loss(logits, window_labels, num_items_in_batch)
        return (loss, {"logits": logits}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = dict(self._prepare_inputs(inputs))
        labels = inputs.pop("labels")
        with torch.no_grad():
            logits, window_labels = windowed_forward(model, inputs, labels)
            loss = windowed_loss(logits, window_labels).detach()
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits.detach(), window_labels)


def run_flat_train_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None, resume=False):
    """Stage 1 — train the flat arm at matched budget, then score all 25 cells.

    Training and grid scoring share one job: the model is already resident, and a
    separate scoring job would reload it for no reason.

    Two deliberate departures from the graph arm's recipe, both from README §3.1:

      * **1 epoch.** Dev EM was 1.0000 by epoch 0.5 and never moved; 3 epochs was
        measured waste.
      * **``eval_loss`` selects the checkpoint**, not ``eval_em_accuracy``. In README §3.1
        EM tied at ceiling across all six evals so "best" broke toward the
        earliest, handing the grid an epoch-0.5 model. eval_loss was still
        improving monotonically and therefore actually discriminates.
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    from ...utils import set_wandb_project
    from ...utils.text_graph_trainer_v2 import make_compute_metrics, shift_logits_for_metrics
    from ...train import select_active_params, print_trainable_parameters
    from .config import EXPERIMENT_NAME
    from .train import CellGroupedSampler, _bind_local_device, _cells_of, _dist_context

    runs_jsonl = runs_jsonl or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "flat_train_runs.jsonl")
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name
    internal_run = (f"{sweep_id}_{run_name}" if (sweep_id and run_name)
                    else f"flat_{cfg.run_name()}")
    output_dir = f"./checkpoints/{EXPERIMENT_NAME}/{internal_run}"

    report_to = "wandb" if cfg.wandb_project else "none"
    if cfg.wandb_project:
        set_wandb_project(cfg.wandb_project)

    torch.set_num_threads(1)
    _bind_local_device()
    rank, world_size = _dist_context()
    if world_size > 1:
        print(f"[ddp] rank {rank}/{world_size}  "
              f"effective batch = {cfg.batch_size} x {cfg.accumulation_steps} x {world_size} "
              f"= {cfg.batch_size * cfg.accumulation_steps * world_size} graphs/step. "
              "Divide accumulation_steps by world_size to match a single-GPU recipe.")
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=cfg.torch_dtype, attn_implementation="sdpa")
    model.to(device)

    print("Loading data...")
    train_split = load_split(cfg, "train")
    dev_split = load_split(cfg, "dev")
    train_view, dev_view = FlatCellView(train_split), FlatCellView(dev_split)
    collator = FlatCollator(tokenizer, cfg.code_len, data_seed=cfg.data_seed)
    train_cells = _cells_of(train_split)
    print(f"Train: {len(train_view)}  Dev: {len(dev_view)}  "
          f"cells in training distribution: {len(cfg.train_cells())}/{len(cfg.cells())}")

    # LoRA only — there is no graph bias in this arm, so no active_params.
    model = select_active_params(model, active_params=[], lora=cfg.lora_config())
    print_trainable_parameters(model)

    # Optimizer steps: under DDP the ranks step together, so divide by world_size
    # too. Only `warmup_steps` reads this, but a wrong value changes the schedule.
    steps_per_epoch = max(1, len(train_view)
                          // (cfg.batch_size * cfg.accumulation_steps * world_size))
    total_steps = steps_per_epoch * cfg.num_epochs
    gc = cfg.gradient_checkpointing

    training_args = TrainingArguments(
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        output_dir=output_dir,
        logging_steps=10,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.accumulation_steps,
        gradient_checkpointing=gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gc else None,
        dataloader_num_workers=cfg.num_workers,
        dataloader_persistent_workers=(cfg.num_workers > 0),
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        metric_for_best_model="eval_loss",       # see docstring
        greater_is_better=False,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to,
        run_name=internal_run,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": cfg.lr / 10},
        warmup_steps=max(1, total_steps // 10),
        weight_decay=0.1,
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=False,
    )

    trainer = FlatTrainer(
        model=model, args=training_args,
        train_dataset=train_view, eval_dataset=dev_view,
        data_collator=collator,
        compute_metrics=make_compute_metrics(include_f1=False),
        preprocess_logits_for_metrics=shift_logits_for_metrics,
        train_sampler=CellGroupedSampler(train_cells, cfg.batch_size, seed=cfg.seed),
    )
    trainer.train(resume_from_checkpoint=resume)

    # Single-process from here: inference plus appended JSONL rows. On every rank
    # this would repeat the grid eval world_size times and duplicate every record.
    if not trainer.is_world_process_zero():
        return None

    dev_metrics = grid_eval(model, dev_view, collator, tokenizer, device=device)
    print("\n" + "=" * 50 + "\nFlat arm — dev mixture (teacher-forced EM):\n" + "=" * 50)
    for k, v in dev_metrics.items():
        print(f"  {k}: {v}")

    checkpoint_path = trainer.state.best_model_checkpoint or output_dir
    append_jsonl(runs_jsonl, {
        "mode": "flat_train", "arm": "flat_text", **sweep_meta,
        "run_name": internal_run, "data_config_key": cfg.data_config_key(),
        "model_name": cfg.model_name, "lora_r": cfg.lora_r,
        "num_epochs": cfg.num_epochs, "seed": cfg.seed, "data_seed": cfg.data_seed,
        "max_train_len": cfg.max_train_len,
        "checkpoint_path": checkpoint_path,
        **{f"dev_{k}": v for k, v in dev_metrics.items()},
    })

    # Score the full grid with the trained model, in this same job.
    grid_jsonl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "results", "flat_trained_grid.jsonl")
    train_cell_set = set(cfg.train_cells())
    mixed = bool(cfg.hop_counts)
    for (n, t) in sorted(cfg.cells(), key=lambda c: cfg.cell_length(*c), reverse=True):
      for hops in cfg.hops_list():
        split = load_split(cfg, cell_split_name(n, t, hops if mixed else None))
        view = FlatCellView(split)
        metrics = grid_eval(model, view, collator, tokenizer, device=device, batch_size=1)
        lo, hi = wilson_interval(round(metrics["em"] * metrics["n"]), metrics["n"])
        append_jsonl(grid_jsonl, {
            "mode": "flat_trained_grid", "arm": "flat_text", **sweep_meta,
            "checkpoint_path": checkpoint_path, "pretrained": False,
            "n_nodes": n, "tokens_per_node": t, "hops": hops,
            "expected_len": cfg.cell_length(n, t),
            "in_train_distribution": (n, t) in train_cell_set,
            "max_train_len": cfg.max_train_len,
            "em_ci_low": lo, "em_ci_high": hi,
            "seed": cfg.seed, "data_seed": cfg.data_seed,
            **metrics,
        })
        print(f"[flat-trained] N={n:4d} T={t:4d}  L={metrics['packed_len']:6d}  "
              f"EM={metrics['em']:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"distractor={metrics['distractor_rate']:.3f}"
              f"{'' if (n, t) in train_cell_set else '   (extrapolation)'}")
        del split, view

    if report_to == "wandb":
        import wandb
        if wandb.run is not None:
            wandb.finish()
    return dev_metrics

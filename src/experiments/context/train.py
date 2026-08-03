"""
Train ONE context configuration and log ONE record.

The model sees a mixture of (N, T) cells capped at ``max_train_len``; the grid is
scored afterwards by ``grid.py``. Everything length-specific about this file is in
two places:

  * :class:`CellGroupedSampler` — batches never mix cells, so the collator's
    padding is block alignment only and the compiled flex kernel sees one shape
    per cell (README §A.7).
  * :class:`~.evaluate.ContextGraphTrainer` — the windowed loss (README §A.8).
"""

import os
import random

import torch
from torch.utils.data import Sampler
from transformers import TrainingArguments

from ...utils import set_wandb_project
from ...utils.text_graph_trainer_v2 import make_compute_metrics, shift_logits_for_metrics
from ...train import select_active_params, print_trainable_parameters, get_device
from .config import EXPERIMENT_NAME
from .evaluate import ContextGraphTrainer, grid_eval
from .model import build_collator, build_model
from .process_dataset import load_split
from ._io import append_jsonl

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RUNS_JSONL = os.path.join(EXPERIMENT_DIR, "results", "train_runs.jsonl")


def _dist_context(rank=None, world_size=None):
    """``(rank, world_size)`` for this process — each explicit arg wins independently.

    Read from the environment rather than ``torch.distributed`` because the sampler is
    constructed before ``Trainer`` initialises the process group.

    The args are honoured SEPARATELY. An earlier version required both to be non-None,
    so ``_dist_context(None, 4)`` silently returned the environment's world size — which
    made every ``world_size=`` argument in the tests a no-op and would have hidden a
    real wave-width bug.
    """
    ws = max(1, int(world_size) if world_size is not None else
             int(os.environ.get("WORLD_SIZE", 1)))
    rk = int(rank) if rank is not None else int(os.environ.get("RANK", 0))
    return (rk if ws > 1 else 0), ws


def _bind_local_device():
    """Pin this rank to its own GPU before any model is built.

    ``build_model`` runs before ``TrainingArguments`` exists and ``get_device()``
    returns a bare ``cuda``, so without this every rank would materialise the model
    on cuda:0 and only then be moved by the Trainer.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


class CellGroupedSampler(Sampler):
    """Index order in which every consecutive ``batch_size`` window is one cell.

    HF's ``group_by_length`` cannot do this: ``Trainer._get_train_sampler`` only
    reads a length column from a real ``datasets.Dataset``, and ``TextGraphDataset``
    merely *wraps* one — so ``LengthGroupedSampler`` falls back to
    ``len(feature["input_ids"])``, which for our items is the **node count**, not
    the token count. It would group by N and silently mix T.

    Cells are padded up to a multiple of ``batch_size`` by resampling within the
    cell, so a partial group can never straddle two cells.

    **Under DDP this sampler does NOT shard, and must not.** HF wraps the training
    dataloader in ``accelerator.prepare``, and accelerate re-shards any dataloader
    whose sampler is not already a ``DistributedSampler`` — round-robin, batch *i*
    to rank ``i % world_size``. A rank-aware sampler therefore gets sharded twice:
    each rank ends up with ``n_train / world_size**2`` graphs and the run silently
    trains on half the dataset. Caught in flight on the first launch — HF's progress
    bar read ``0/2000`` where the recipe implies 4000 optimizer steps.

    What this class still owns is the ORDER, and one property of it matters:

      * **Every rank must see the same CELL at the same step.** Groups are emitted
        in *waves* of ``world_size`` consecutive groups drawn from one cell, so
        accelerate's round-robin hands rank r the r-th group of each wave — same
        (N, T) on every rank, every step. DDP synchronises each step, so step time
        is the max over ranks; with cell lengths spanning 960..15,936 tokens at a
        superlinear cost in length, a rank drawing N=16 against a peer drawing
        N=128 would idle for most of the step. Same-cell waves also hold the
        compiled flex kernel at one shape per step rather than ``world_size``.

    ``world_size`` is therefore only the wave WIDTH, never a slice index. At
    ``world_size == 1`` the emitted order is byte-identical to the pre-DDP version
    (the wave list is a permutation of the group list consuming the same RNG
    draws), so runs stay comparable with those already completed.
    """

    def __init__(self, cells, batch_size, seed=0, world_size=None):
        self.cells = list(cells)
        self.batch_size = max(1, batch_size)
        self.seed = seed
        self.epoch = 0
        _rank, self.world_size = _dist_context(None, world_size)
        by_cell = {}
        for idx, cell in enumerate(self.cells):
            by_cell.setdefault(cell, []).append(idx)
        self.by_cell = by_cell
        wave = self.batch_size * self.world_size
        # FULL length, not per-rank: accelerate does the per-rank split downstream.
        self._length = sum(-(-len(ix) // wave) * wave for ix in by_cell.values())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self._length

    def __iter__(self):
        # Seeded identically on every rank, so all ranks agree which wave is step t.
        rng = random.Random((self.seed, self.epoch).__hash__())
        span = self.batch_size * self.world_size
        waves = []
        for cell, indices in sorted(self.by_cell.items()):
            order = list(indices)
            rng.shuffle(order)
            remainder = (-len(order)) % span
            if remainder:
                order += [rng.choice(indices) for _ in range(remainder)]
            groups = [order[i:i + self.batch_size]
                      for i in range(0, len(order), self.batch_size)]
            waves += [groups[i:i + self.world_size]
                      for i in range(0, len(groups), self.world_size)]
        rng.shuffle(waves)
        # Flatten the wave WHOLE. Taking wave[self.rank] here is the double-shard bug.
        return iter([i for wave in waves for group in wave for i in group])


def _cells_of(dataset):
    """The (N, T) cell of every graph in a built split."""
    return [(g.graph["cell_n"], g.graph["cell_t"]) for g in dataset.graphs]


def _save_train_record(cfg, run_name, results, runs_jsonl, sweep_meta=None):
    record = {
        "mode": "train",
        **(sweep_meta or {}),
        "run_name": run_name,
        "data_config_key": cfg.data_config_key(),
        # ── hyperparameters ──
        "model_name": cfg.model_name, "k_hop": cfg.k_hop,
        "graph_attn_impl": cfg.graph_attn_impl, "dtype": cfg.dtype,
        "lora": cfg.lora, "lora_r": cfg.lora_r, "lora_dropout": cfg.lora_dropout,
        "spd": cfg.spd, "max_spd": cfg.max_spd,
        "rrwp": cfg.rrwp, "max_rw_steps": cfg.max_rw_steps,
        "magnetic": cfg.magnetic, "magnetic_dim": cfg.magnetic_dim,
        # Bias sharing granularity (0 = legacy per-layer). A first-class column:
        # a G sweep is otherwise only distinguishable by parsing the run name.
        "magnetic_groups": cfg.magnetic_groups,
        "magnetic_q": cfg.magnetic_q, "magnetic_m": cfg.magnetic_m,
        "node_counts": list(cfg.node_counts), "token_counts": list(cfg.token_counts),
        "max_train_len": cfg.max_train_len, "n_train": cfg.n_train,
        "n_dev": cfg.n_dev, "n_test": cfg.n_test, "code_len": cfg.code_len,
        "lr": cfg.lr, "bias_lr": cfg.bias_lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "max_steps": cfg.max_steps, "seed": cfg.seed, "data_seed": cfg.data_seed,
        # ── results ──
        **(results or {}),
    }
    append_jsonl(runs_jsonl, record)
    print(f"[results] appended training run to {runs_jsonl}")


def run_train_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None, resume=False):
    """Train this config on the capped (N, T) mixture; log one record."""
    runs_jsonl = runs_jsonl or _DEFAULT_RUNS_JSONL
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name
    internal_run = (f"{sweep_id}_{run_name}" if (sweep_id and run_name) else cfg.run_name())
    output_dir = f"./checkpoints/{EXPERIMENT_NAME}/{internal_run}"

    report_to = "wandb" if cfg.wandb_project else "none"
    if cfg.wandb_project:
        set_wandb_project(cfg.wandb_project)

    # Pin the intra-op thread count BEFORE anything compiles: the DataLoader's
    # pin-memory thread flips this process-global, and dynamo guards compiled flex
    # kernels on it — the flip silently doubles the recompile count per shape.
    torch.set_num_threads(1)

    _bind_local_device()
    rank, world_size = _dist_context()
    if world_size > 1:
        print(f"[ddp] rank {rank}/{world_size}  "
              f"effective batch = {cfg.batch_size} x {cfg.accumulation_steps} x {world_size} "
              f"= {cfg.batch_size * cfg.accumulation_steps * world_size} graphs/step. "
              "Divide accumulation_steps by world_size to match a single-GPU recipe.")

    device = get_device()
    model, tokenizer = build_model(cfg, device)

    print("Loading data...")
    train_dataset = load_split(cfg, "train")
    dev_dataset = load_split(cfg, "dev")
    collator = build_collator(cfg, tokenizer)
    train_cells = _cells_of(train_dataset)
    print(f"Train: {len(train_dataset)}  Dev: {len(dev_dataset)}  "
          f"cells in training distribution: {len(cfg.train_cells())}/{len(cfg.cells())}")

    model = select_active_params(model, active_params=list(cfg.active_params),
                                 lora=cfg.lora_config())
    print_trainable_parameters(model)

    # Optimizer steps, not micro-batches: under DDP the ranks step together, so the
    # per-epoch count divides by world_size too. Only `warmup_steps` reads this, but
    # a wrong value there silently changes the schedule.
    steps_per_epoch = max(1, len(train_dataset)
                          // (cfg.batch_size * cfg.accumulation_steps * world_size))
    total_steps = steps_per_epoch * cfg.num_epochs
    gc = cfg.gradient_checkpointing

    training_args = TrainingArguments(
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        output_dir=output_dir,
        logging_steps=10,
        per_device_train_batch_size=cfg.batch_size,
        # Eval at batch 1: an eval batch that mixed cells would pad every row to
        # the longest one and widen the logits window for no benefit.
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
        metric_for_best_model="eval_em_accuracy",
        greater_is_better=True,
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

    trainer = ContextGraphTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=dev_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(include_f1=False),
        preprocess_logits_for_metrics=shift_logits_for_metrics,
        active_params=list(cfg.active_params), bias_lr=cfg.bias_lr,
        bias_weight_decay=cfg.bias_weight_decay,
        train_sampler=CellGroupedSampler(train_cells, cfg.batch_size, seed=cfg.seed),
    )

    trainer.train(resume_from_checkpoint=resume)

    # Everything below is single-process work: a plain inference pass plus one
    # appended JSONL row. Running it on every rank would repeat the grid eval
    # world_size times and write world_size duplicate records.
    if not trainer.is_world_process_zero():
        return None

    # Best checkpoint is loaded (load_best_model_at_end + GraphTrainerV2's bias
    # restore). Score the dev mixture with the same metric the grid uses, so the
    # run record carries a directly comparable number.
    grid_collator = build_collator(cfg, tokenizer, for_grid=True)
    dev_metrics = grid_eval(model, dev_dataset, grid_collator, tokenizer, device=device)
    print("\n" + "=" * 50 + "\nBest model — dev mixture (teacher-forced EM):\n" + "=" * 50)
    for k, v in dev_metrics.items():
        print(f"  {k}: {v}")

    results = {f"dev_{k}": v for k, v in dev_metrics.items()}
    results["checkpoint_path"] = trainer.state.best_model_checkpoint or output_dir
    _save_train_record(cfg, internal_run, results, runs_jsonl=runs_jsonl, sweep_meta=sweep_meta)

    if report_to == "wandb":
        import wandb
        if wandb.run is not None:
            wandb.finish()
    return results

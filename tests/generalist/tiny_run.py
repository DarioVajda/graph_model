"""A whole generalist run, small enough to fit in a CPU test.

Everything a :class:`~src.generalist.trainer.GeneralistTrainer` needs, built in
memory and deterministically: a tiny GTLM-Llama with a LoRA adapter and one live
graph-bias channel, a three-task registry and its resolved mixture, an in-memory
``TaskSource`` per (task, pass), and a ``GraphCollatorV2``. No tokenizer, no
files, no adapter package — the graphs are synthesised as ``TextGraphDataset``
items directly, because what the trainer tests are about is the mixture, the
schedule and the checkpoint, not how text became tokens (that is T1's).

Two properties are load-bearing for the tests that use this module and are
therefore built in rather than left to chance:

* **Every item has the same shape** — three nodes, ``[3, 3, 4]`` tokens, an
  answer span of the prompt node's last two tokens. So a micro-batch has no
  padding whatever it holds, and two runs that group the same examples into
  differently sized micro-batches differ only by floating-point summation order.
  Padding is exercised by T3, not here; leaving it in would blur the one
  assertion the accumulation-invariance test is making.
* **The numbers divide** — ``tokens_per_step=128`` over ``mean_tokens=16`` is
  exactly 8 examples per step, so ``examples_in_step(k)`` is 8 for every *k* and
  a step splits evenly into 1, 2 or 4 micro-batches.
"""

from __future__ import annotations

import hashlib

import torch
from transformers import TrainingArguments, TrainerCallback

from src.generalist.mixture import MixtureSampler
from src.generalist.registry import Registry, TaskSpec, resolve
from src.generalist.schedule import Schedule
from src.generalist.trainer import GeneralistTrainer
from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.train.model import select_active_params
from src.utils.text_graph_collator_v2 import GraphCollatorV2
from tests.helpers.tiny_model import BASE_CONFIG

#: The graph-bias parameter-name substring for the v2 layout (``dispatch.py``'s
#: ``ACTIVE_PARAMS_V2``). Same list the real runs train.
ACTIVE_PARAMS = ["graph_bias"]

TASKS = ("t/alpha", "t/beta", "t/gamma")

NODE_TOKENS = (3, 3, 4)      # per-node token counts; the prompt node is the last
PROMPT_NODE = 2
ANSWER_TOKENS = 2            # supervised tail of the prompt node
MEAN_TOKENS = 16.0
TOKENS_PER_STEP = 128        # / MEAN_TOKENS -> exactly 8 examples per step
TRAIN_SIZE = 64
VOCAB = BASE_CONFIG["vocab_size"]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def _stream_seed(*parts) -> int:
    """A stable 32-bit seed from identifiers, so a "pass" is reproducible."""
    joined = "\x1f".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(joined).digest()[:4], "big")


class InMemorySource:
    """One task's one pass, as the ``TaskSource`` protocol D4 asks for.

    ``__len__``, ``__getitem__``, ``lengths()`` and the four attributes are the
    whole of what :class:`~src.generalist.mixture.MixtureSampler` uses, so a
    source needs neither a dataset on disk nor a tokenizer to stand in for one.
    """

    def __init__(self, task: str, pass_id: int = 0, size: int = TRAIN_SIZE,
                 split: str = "train", arm: str = "graph"):
        self.task = task
        self.split = split
        self.arm = arm
        self.pass_id = int(pass_id)
        self._items = [self._build(i) for i in range(int(size))]

    def _build(self, index: int) -> dict:
        g = torch.Generator().manual_seed(_stream_seed(self.task, self.pass_id, index))
        input_ids = [torch.randint(1, VOCAB, (n,), generator=g).tolist()
                     for n in NODE_TOKENS]
        n_nodes = len(NODE_TOKENS)
        prompt_ids = input_ids[PROMPT_NODE]
        labels = torch.full((len(prompt_ids),), -100, dtype=torch.long)
        labels[-ANSWER_TOKENS:] = torch.tensor(prompt_ids[-ANSWER_TOKENS:],
                                               dtype=torch.long)
        # A path graph, so the shortest-path table the SPD bias reads is a real
        # (and non-constant) function of the structure rather than a constant.
        spd = torch.tensor([[abs(i - j) for j in range(n_nodes)]
                            for i in range(n_nodes)], dtype=torch.long)
        return {
            "num_nodes": n_nodes,
            "prompt_node": PROMPT_NODE,
            "edges": [(i, i + 1) for i in range(n_nodes - 1)],
            "input_ids": input_ids,
            "labels": labels,
            "shortest_path_dists": spd,
            "ds_label": self.task,
        }

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> dict:
        return self._items[i]

    def lengths(self):
        nodes = [item["num_nodes"] for item in self._items]
        tokens = [sum(len(ids) for ids in item["input_ids"]) for item in self._items]
        return nodes, tokens


def make_get_source(size: int = TRAIN_SIZE):
    """``(task, pass_id) -> TaskSource``, cached so a pass is built once."""
    cache: dict = {}

    def get_source(task: str, pass_id: int) -> InMemorySource:
        key = (task, int(pass_id))
        if key not in cache:
            cache[key] = InMemorySource(task, pass_id, size=size)
        return cache[key]

    return get_source


# ─────────────────────────────────────────────────────────────────────────────
# Registry and mixture
# ─────────────────────────────────────────────────────────────────────────────

def build_registry(tasks=TASKS, passes: int = 8) -> Registry:
    return Registry([
        TaskSpec(name=name, domain="tiny", adapter="tiny", kind="corpus",
                 answer_kind="token", weight=1.0, passes=passes,
                 metric="exact_match", build_version="tiny-1",
                 mean_tokens=MEAN_TOKENS, train_size=TRAIN_SIZE)
        for name in tasks
    ])


def build_mixture(registry: Registry = None, tasks=TASKS, steps: int = 256,
                  tokens_per_step: int = TOKENS_PER_STEP, weights=None):
    """A resolved mixture over ``tasks``.

    ``steps=`` is passed so the budget comes from a step count rather than from
    the corpora's pass caps, and ``min_examples_per=0`` disables the
    "one example per 1000 steps" floor — both for the reason ``registry.resolve``
    documents: over a handful of steps neither check is measuring anything.
    """
    registry = registry if registry is not None else build_registry(tasks=tasks)
    weights = weights or {name: 1.0 for name in tasks}
    entries = [{"name": name, "weight": weights[name]} for name in tasks]
    return registry, resolve(registry, entries, tokens_per_step=tokens_per_step,
                             steps=steps, min_examples_per=0)


def build_sampler(mixture, seed: int = 0, accumulation_steps: int = 1,
                  world_size: int = 1, size: int = TRAIN_SIZE) -> MixtureSampler:
    return MixtureSampler(mixture, seed=seed, get_source=make_get_source(size),
                          accumulation_steps=accumulation_steps,
                          world_size=world_size)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(seed: int = 0):
    """A tiny GTLM-Llama with a LoRA adapter and a live SPD bias channel.

    ``spd`` alone: its table is zero-initialised and receives a gradient through
    the attention logits, so the bias norm starts at exactly 0 and moves — which
    is what the "the bias is actually trained, and actually restored" assertions
    need. LoRA dropout is 0 so two runs of the same steps are comparable without
    depending on the RNG being restored to the byte (it is, but a test that only
    passes because of that would be testing the wrong thing).
    """
    torch.manual_seed(seed)
    config = GTLMLlamaConfig(spd=True, max_spd=8, graph_attn_impl="eager",
                             **BASE_CONFIG)
    model = GTLMLlamaForCausalLM(config)
    for param in model.parameters():
        param.requires_grad = False
    model = select_active_params(
        model, active_params=ACTIVE_PARAMS,
        lora={"r": 2, "lora_alpha": 4, "lora_dropout": 0.0,
              "target_modules": ["q_proj", "v_proj"]})
    return model


def bias_tensors(model) -> dict:
    """``{name: detached tensor}`` for every graph-bias parameter."""
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if any(a in n for a in ACTIVE_PARAMS)}


def trainable_tensors(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class RecordingTrainer(GeneralistTrainer):
    """A trainer that keeps the key of every example it was handed.

    ``(task id, row index, sampler step)`` per micro-batch, in the order the
    micro-batches arrived. This is the sequence D4.1 promises a resume
    reproduces, and comparing endpoints alone would not notice two runs that saw
    the same examples in a different order.
    """

    def __init__(self, *args, **kwargs):
        self.batch_keys: list = []
        self.step_losses: list = []
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        if "task_ids" in inputs:
            self.batch_keys.append(list(zip(
                inputs["task_ids"].tolist(),
                inputs["example_index"].tolist(),
                inputs["step"].tolist())))
        out = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                   num_items_in_batch=num_items_in_batch)
        loss = out[0] if return_outputs else out
        self.step_losses.append(float(loss.detach()))
        return out


class LearningRateProbe(TrainerCallback):
    """Reads the LR the optimizer will actually apply, at the start of each step.

    ``on_step_begin`` fires with ``global_step == k`` and before
    ``optimizer.step()``, so ``param_groups[i]["lr"]`` is the value step *k* runs
    at. Reading it off the schedule object instead would only prove the schedule
    agrees with itself.
    """

    def __init__(self):
        self.seen: list = []

    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        if optimizer is None:
            return
        self.seen.append((int(state.global_step),
                          [(bool(g.get("is_bias", False)), float(g["lr"]))
                           for g in optimizer.param_groups]))

    def lrs_at(self, step: int):
        """``(lora lr, bias lr)`` recorded at ``step``."""
        for seen_step, groups in self.seen:
            if seen_step == step:
                lora = [lr for is_bias, lr in groups if not is_bias]
                bias = [lr for is_bias, lr in groups if is_bias]
                return (lora[0] if lora else None, bias[0] if bias else None)
        raise AssertionError(f"no LR recorded at step {step}; saw "
                             f"{[s for s, _ in self.seen]}")


def build_training_args(output_dir: str, *, max_steps: int = 6,
                        accumulation_steps: int = 1, lr: float = 1e-2,
                        save_steps: int = 0, seed: int = 0) -> TrainingArguments:
    """``TrainingArguments`` for a tiny CPU run.

    ``max_grad_norm=0.0`` turns clipping off. Clipping is applied to the
    accumulated gradient and so is itself accumulation-invariant, but it is a
    non-linearity right at the point the invariance test measures, and a norm
    that lands near the threshold would turn a 1e-7 float difference into a
    visible one.
    """
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=lr,
        weight_decay=0.0,
        max_grad_norm=0.0,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        save_safetensors=True,
    )


def build_trainer(output_dir: str, *, model=None, accumulation_steps: int = 1,
                  max_steps: int = 6, warmup_steps: int = 5,
                  lr_min_factor: float = 0.25, lr: float = 1e-2,
                  bias_lr: float = 5e-2, seed: int = 0, save_steps: int = 0,
                  schedule: Schedule = None, sampler: MixtureSampler = None,
                  mixture=None, registry=None, weights=None,
                  trainer_cls=RecordingTrainer, callbacks=None,
                  args_overrides=None, **trainer_kwargs):
    """One assembled run. Returns ``(trainer, model, sampler, schedule)``.

    The warmup deliberately spans the checkpoint boundary the resume tests use:
    with ``warmup_steps=5`` the LR is still climbing at step 3, so a resume that
    restored the schedule but not the optimizer's LR — or the other way round —
    diverges immediately instead of only under a decay nobody runs in a test.
    """
    if model is None:
        model = build_model(seed=seed)
    if mixture is None:
        registry, mixture = build_mixture(registry=registry, weights=weights)
    if sampler is None:
        sampler = build_sampler(mixture, seed=seed,
                                accumulation_steps=accumulation_steps)
    if schedule is None:
        schedule = Schedule.training(warmup_steps=warmup_steps,
                                     lr_min_factor=lr_min_factor)

    args = build_training_args(output_dir, max_steps=max_steps,
                               accumulation_steps=accumulation_steps, lr=lr,
                               save_steps=save_steps, seed=seed)
    # Applied after construction on purpose: some of the settings a test wants to
    # put in front of the trainer's own guardrails (load_best_model_at_end) are
    # ones `TrainingArguments.__post_init__` refuses outright, and the guardrail
    # under test is the trainer's, not HF's.
    for name, value in (args_overrides or {}).items():
        setattr(args, name, value)
    trainer = trainer_cls(
        model=model, args=args,
        data_collator=GraphCollatorV2(pad_token_id=0),
        train_dataset=None, eval_dataset=None,
        active_params=ACTIVE_PARAMS, bias_lr=bias_lr,
        sampler=sampler, schedule=schedule, registry=registry,
        callbacks=list(callbacks or []),
        **trainer_kwargs,
    )
    return trainer, model, sampler, schedule

"""End-to-end regression test for the best-checkpoint bias reload.

Unlike test_graph_trainer_load_best.py (bare trainers, HF super mocked), this
drives the REAL HF training loop: a tiny PEFT-wrapped Llama with a trainable
non-adapter parameter (name matches ``active_params``), checkpointing every
step, a scripted metric that makes the FIRST checkpoint the best, and
``load_best_model_at_end=True``. Training then drifts the bias parameter for
several more steps, so at the end HF reloads the best-step adapter — and the
override must restore the best-step bias tensor too. Before the fix, the model
ended up with the step-1 adapter and the final-step bias values (the bug that
mis-scored every post-train evaluate(); see kgqa/results/reeval_bias_bug).

Runs on CPU in seconds (64-hidden 2-layer Llama from tests/helpers/tiny_model).
"""

import os

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.train.model import select_active_params
from src.utils.text_graph_trainer_v2 import GraphTrainerV2
from tests.helpers.tiny_model import BASE_CONFIG

ACTIVE_PARAMS = ["graph_bias"]


class BiasedTinyLlama(LlamaForCausalLM):
    """Tiny Llama plus one trainable logit-shift param named like a graph bias.

    The shift feeds the loss, so the optimizer moves it every step — giving the
    checkpoints genuinely different ``bias_parameters.pt`` contents, exactly
    like the drifting graph-bias modules in the real GTLM model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.graph_bias_shift = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = out.logits + self.graph_bias_shift
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, n=16, seq_len=8, vocab=BASE_CONFIG["vocab_size"]):
        g = torch.Generator().manual_seed(0)
        self.rows = torch.randint(1, vocab, (n, seq_len), generator=g)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return {"input_ids": row, "attention_mask": torch.ones_like(row), "labels": row}


def _stack_collator(features):
    return {k: torch.stack([f[k] for f in features]) for k in features[0]}


class BiasSnapshot(TrainerCallback):
    """Record the live bias tensor after every optimizer step (the last snapshot
    is the true end-of-training state — checkpoint rotation can't erase it, and
    on_step_end runs before HF's _load_best_model)."""

    def __init__(self, model):
        self._model = model
        self.last = None

    def on_step_end(self, args, state, control, **kwargs):
        [(_, p)] = [(n, p) for n, p in self._model.named_parameters() if "graph_bias" in n]
        self.last = p.detach().clone()


def _run_tiny_training(tmp_path):
    torch.manual_seed(0)
    model = BiasedTinyLlama(LlamaConfig(**BASE_CONFIG))
    for p in model.parameters():
        p.requires_grad = False
    model = select_active_params(
        model, active_params=ACTIVE_PARAMS,
        lora={"r": 2, "target_modules": ["q_proj", "v_proj"]})

    # Scripted selection metric: step-1 checkpoint is best, then monotone decay.
    scores = iter([1.0, 0.8, 0.6, 0.4])
    args = TrainingArguments(
        output_dir=str(tmp_path), max_steps=4,
        per_device_train_batch_size=4, per_device_eval_batch_size=4,
        learning_rate=1.0,  # huge on purpose: every step visibly moves the bias
        eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
        metric_for_best_model="fake", greater_is_better=True,
        load_best_model_at_end=True, save_total_limit=1,
        logging_strategy="no", report_to=[], use_cpu=True, seed=0,
    )
    snapshot = BiasSnapshot(model)
    trainer = GraphTrainerV2(
        model=model, args=args,
        train_dataset=TokenDataset(), eval_dataset=TokenDataset(n=4),
        data_collator=_stack_collator,
        compute_metrics=lambda _: {"fake": next(scores)},
        active_params=ACTIVE_PARAMS,
        callbacks=[snapshot],
    )
    trainer.train()

    best_ckpt = trainer.state.best_model_checkpoint
    assert best_ckpt is not None and best_ckpt.endswith("checkpoint-1")
    best_saved = torch.load(os.path.join(best_ckpt, "bias_parameters.pt"),
                            map_location="cpu", weights_only=True)
    [(name, best_bias)] = [(n, t) for n, t in best_saved.items() if "graph_bias" in n]
    live_bias = dict(trainer.model.named_parameters())[name].detach()
    final_bias = snapshot.last

    # Preconditions, or the assertions below would pass vacuously: the bias
    # moved off init by the best step, and kept drifting after it.
    assert best_bias.abs().sum() > 0
    assert not torch.allclose(final_bias, best_bias)
    return live_bias, best_bias, final_bias


def test_load_best_model_at_end_restores_bias(tmp_path):
    live, best, final = _run_tiny_training(tmp_path)
    # THE regression: post-train model must carry the BEST-step bias, not the final.
    assert torch.allclose(live, best)
    assert not torch.allclose(live, final)


def test_without_fix_the_bug_reappears(tmp_path, monkeypatch):
    """Sanity check on the test itself: with the override neutered (pre-fix
    behavior) the post-train model keeps the drifted end-of-training bias —
    i.e. the assertion above genuinely detects the bug."""
    import transformers

    monkeypatch.setattr(GraphTrainerV2, "_load_best_model",
                        transformers.Trainer._load_best_model)
    live, best, final = _run_tiny_training(tmp_path)
    assert torch.allclose(live, final)       # stale end-of-training bias
    assert not torch.allclose(live, best)    # best NOT restored -> the bug

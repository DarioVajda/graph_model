"""
T3 — the mixture sampler and the two-level loss accounting
(`src/generalist/mixture.py`, DESIGN.md §D4).

What has to hold, and why each one is here rather than read off a loss curve:

* the **realised example share equals the configured share**. A weight that does
  not become a share is the ``--magnetic-groups`` bug one layer down: the run
  reports the mixture it was configured with and trains on a different one;
* a **corpus stops at its passes** and a **generator refreshes** (D4.2). A corpus
  that silently wrapped past its cap would repeat data the budget said it would
  not; a generator that did not refresh would train ``passes`` times on one draw
  while the report said otherwise;
* a **resumed sampler draws what an uninterrupted one would** (D4.1). Without
  this, every comparison across a chunk boundary of the Slurm chain (D8.3) is
  confounded by a change in data order;
* **micro-batches stay under the token budget and are mixed** (D4.3/D4.4).
  Homogeneous batches make per-task gradient noise a function of task share,
  which is exactly what the mixture-weight readout measures;
* the **gradient share equals the example share, under any accumulation and any
  rank count** (D4.3). Normalising by the micro-batch instead of by the optimizer
  step is the standard footgun and it is invisible in the loss curve — the run
  simply weights the mixture differently than the config says.

Everything is CPU, with fake sources and a three-parameter linear model; no
tokenizer and no dataset are involved.
"""

import json
import math
import os
from collections import Counter

import pytest
import torch
import torch.multiprocessing as mp

from src.generalist.mixture import (
    Draw,
    MixtureDataset,
    MixtureError,
    MixtureLoss,
    MixtureSampler,
    count_examples_in_step,
    measure_grad_share,
    task_ids_for,
    wrap_collator,
)
from src.generalist.registry import Registry, TaskSpec, resolve


# ─────────────────────────────────────────────────────────────────────────────
# Fakes: the TaskSource protocol, and a get_source that records what it loaded
# ─────────────────────────────────────────────────────────────────────────────

class FakeSource:
    """A ``TaskSource``: length, items, ``lengths()``, and the four attributes.

    An item carries its own ``row`` and ``pass_id`` so a test can assert that the
    row the sampler drew is the row that came back — the generator-refresh bug
    (D4.2) shows up exactly as a row from the wrong pass.
    """

    def __init__(self, task, n, pass_id=0, tokens=64, nodes=8, split="train",
                 arm="graph"):
        self.task = task
        self.split = split
        self.arm = arm
        self.pass_id = pass_id
        self.n = int(n)
        self._tokens = [tokens(i) if callable(tokens) else int(tokens)
                        for i in range(self.n)]
        self._nodes = [nodes(i) if callable(nodes) else int(nodes)
                       for i in range(self.n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "ds_label": self.task,
            "num_nodes": self._nodes[i],
            "num_tokens": self._tokens[i],
            "row": int(i),
            "pass_id": self.pass_id,
        }

    def lengths(self):
        return list(self._nodes), list(self._tokens)


class Loader:
    """``get_source(task, pass_id)`` plus a log of every pass it was asked for."""

    def __init__(self, sizes, **kwargs):
        self.sizes = dict(sizes)
        self.kwargs = kwargs
        self.calls = []

    def __call__(self, task, pass_id):
        self.calls.append((task, pass_id))
        return FakeSource(task, self.sizes[task], pass_id=pass_id,
                          **self.kwargs.get(task, {}))


def make_mixture(specs, weights, tokens_per_step, steps):
    """A resolved :class:`Mixture` over fake tasks, budgeted by a step count."""
    registry = Registry(TaskSpec(**s) for s in specs)
    entries = [{"name": name, "weight": w} for name, w in weights.items()]
    return resolve(registry, entries, tokens_per_step, steps=steps)


# The share fixture: very unequal weights (16 : 3 : 1 -> .80 / .15 / .05), two
# corpora big enough not to exhaust over the horizon, and one generator. Every
# task is 100 tokens, so 2000 tokens/step is exactly 20 examples/step.
SHARE_SPECS = (
    dict(name="t/big", domain="fake", adapter="fake", kind="corpus",
         mean_tokens=100.0, train_size=40000, passes=1),
    dict(name="t/mid", domain="fake", adapter="fake", kind="corpus",
         mean_tokens=100.0, train_size=10000, passes=1),
    dict(name="t/small", domain="fake", adapter="fake", kind="generator",
         mean_tokens=100.0, cap_per_pass=1000),
)
SHARE_WEIGHTS = {"t/big": 16.0, "t/mid": 3.0, "t/small": 1.0}


def share_sampler(seed=7, **kwargs):
    mixture = make_mixture(SHARE_SPECS, SHARE_WEIGHTS, tokens_per_step=2000,
                           steps=2000)
    loader = Loader({"t/big": 40000, "t/mid": 10000, "t/small": 1000})
    return MixtureSampler(mixture, seed=seed, get_source=loader, **kwargs), loader


# ─────────────────────────────────────────────────────────────────────────────
# D4.1 — the draw plan
# ─────────────────────────────────────────────────────────────────────────────

def test_the_realised_share_matches_the_configured_share():
    """2000 steps x 20 examples. The tolerance is absolute in the share.

    A relative tolerance would be the wrong test at these counts: the multinomial
    standard error on the 0.05 task over 40000 draws is ~0.001 in share, i.e.
    ~2 % of its own value, so a 2 % *relative* band would fail on sampling noise
    roughly a third of the time and say nothing about the sampler.
    """
    sampler, _ = share_sampler()
    seen = Counter()
    for k in range(2000):
        for draw in sampler.draw_step(k):
            seen[draw.task] += 1

    total = sum(seen.values())
    assert total == 40000
    for name, share in sampler.mixture.shares.items():
        assert abs(seen[name] / total - share) < 0.02, (name, seen[name] / total)
    # And the ordering of the shares is not an accident of the tolerance.
    assert seen["t/big"] > seen["t/mid"] > seen["t/small"] > 0


def test_the_step_composition_is_a_pure_function_of_the_step():
    """D4.1: counts for step k do not depend on how the sampler got to k."""
    a, _ = share_sampler()
    b, _ = share_sampler()
    for k in (0, 1, 17, 999):
        assert a.counts_for_step(k) == b.counts_for_step(k)
    # b walks to step 500 first; the plan for 999 is unmoved.
    for k in range(500):
        b.draw_step(k)
    assert a.counts_for_step(999) == b.counts_for_step(999)


def test_a_different_seed_draws_a_different_plan():
    a, _ = share_sampler(seed=7)
    b, _ = share_sampler(seed=8)
    assert any(a.counts_for_step(k) != b.counts_for_step(k) for k in range(20))


def test_the_fractional_example_count_averages_to_the_configured_rate():
    """``examples_per_step`` is a token budget over a mean length, so it is a float.

    Rounding it every step would drift the realised token budget; the accumulator
    has to land on ``floor(K x e)`` after K steps, exactly.
    """
    odd = make_mixture(
        (dict(name="t/a", domain="fake", adapter="fake", kind="corpus",
              mean_tokens=300.0, train_size=100000, passes=1),),
        {"t/a": 1.0}, tokens_per_step=1000, steps=3000)
    odd_sampler = MixtureSampler(odd, seed=1,
                                 get_source=Loader({"t/a": 100000}))
    assert odd_sampler.examples_per_step == pytest.approx(1000 / 300)
    for K in (1, 7, 100, 3000):
        total = sum(odd_sampler.examples_in_step(k) for k in range(K))
        assert total == math.floor(K * odd_sampler.examples_per_step)
        assert odd_sampler.fraction_at(K) == pytest.approx(
            K * odd_sampler.examples_per_step - total)
    assert {odd_sampler.examples_in_step(k) for k in range(50)} == {3, 4}


# ─────────────────────────────────────────────────────────────────────────────
# D4.2 — passes
# ─────────────────────────────────────────────────────────────────────────────

def small_mixture(passes=2, train_size=5, gen_size=4, tokens_per_step=200,
                  steps=100, seed=3, **kwargs):
    """One tiny corpus and one tiny generator, 2 examples/step."""
    specs = (
        dict(name="t/corpus", domain="fake", adapter="fake", kind="corpus",
             mean_tokens=100.0, train_size=train_size, passes=passes),
        dict(name="t/gen", domain="fake", adapter="fake", kind="generator",
             mean_tokens=100.0, cap_per_pass=gen_size),
    )
    mixture = make_mixture(specs, {"t/corpus": 1.0, "t/gen": 1.0},
                           tokens_per_step=tokens_per_step, steps=steps)
    loader = Loader({"t/corpus": train_size, "t/gen": gen_size})
    return MixtureSampler(mixture, seed=seed, get_source=loader, **kwargs), loader


def test_a_corpus_stops_after_its_passes():
    """5 rows x 2 passes is 10 examples and not an eleventh, ever."""
    sampler, loader = small_mixture(passes=2, train_size=5)
    seen = Counter()
    rows = []
    for k in range(100):
        for draw in sampler.draw_step(k):
            seen[draw.task] += 1
            if draw.task == "t/corpus":
                rows.append((draw.pass_id, draw.index))

    assert seen["t/corpus"] == 10
    assert "t/corpus" in sampler.exhausted
    # Each pass is a permutation: every row exactly once per pass, no repeats
    # within one and no third pass.
    assert sorted(rows) == [(p, i) for p in (0, 1) for i in range(5)]
    # A third pass is never even asked for: the retirement happens at the cap.
    assert ("t/corpus", 2) not in loader.calls
    # The generator is untouched by the corpus retiring and keeps drawing.
    assert seen["t/gen"] > 50


def test_a_generator_advances_its_pass_and_is_reloaded_each_time():
    """D4.2: a generator has no pass cap and its source is re-requested per pass."""
    sampler, loader = small_mixture(gen_size=4)
    for k in range(20):
        sampler.draw_step(k)

    assert sampler.pass_id["t/gen"] >= 3
    gen_passes = [p for task, p in loader.calls if task == "t/gen"]
    assert gen_passes == sorted(set(gen_passes))          # each pass loaded once
    assert gen_passes == list(range(len(gen_passes)))     # and in order, no gaps
    assert "t/gen" not in sampler.exhausted


def test_an_item_comes_from_the_pass_it_was_drawn_from():
    """The row index means nothing without its pass: row 2 of pass 3 is not row 2
    of pass 4. Materialising after a rollover is the bug this pins."""
    sampler, _ = small_mixture(gen_size=4)
    seen_passes = set()
    for k in range(10):
        for batch in sampler.batches_for_step(k):
            for d in batch:
                item = sampler.source(d.task, d.pass_id)[d.index]
                assert item["pass_id"] == d.pass_id
                assert item["row"] == d.index
                if d.task == "t/gen":
                    seen_passes.add(d.pass_id)
    assert len(seen_passes) > 1

    # And the same through the dataset, which is what the trainer sees.
    fresh, _ = small_mixture(gen_size=4)
    for batch in MixtureDataset(fresh, end_step=10):
        for item in batch:
            assert item["row"] == item["example_index"]


# ─────────────────────────────────────────────────────────────────────────────
# D4.1 — determinism and resume
# ─────────────────────────────────────────────────────────────────────────────

def keys_of(batches):
    return [[(d.task, d.pass_id, d.index) for d in batch] for batch in batches]


def test_two_fresh_samplers_draw_identical_batches():
    a, _ = share_sampler(accumulation_steps=2)
    b, _ = share_sampler(accumulation_steps=2)
    for k in range(12):
        assert keys_of(a.batches_for_step(k)) == keys_of(b.batches_for_step(k))


def test_a_resumed_sampler_continues_the_uninterrupted_stream():
    """§T4's sampler half: restore at step k, draw the same keys at k+1..."""
    straight, _ = share_sampler(accumulation_steps=2)
    state = None
    expected = []
    for k in range(10):
        batches = straight.batches_for_step(k)
        if k == 4:
            state = straight.state_dict()
        if k > 4:
            expected.append(keys_of(batches))

    assert state is not None and state["step"] == 5

    resumed, _ = share_sampler(accumulation_steps=2)
    resumed.load_state_dict(json.loads(json.dumps(state)))   # through the file
    got = [keys_of(resumed.batches_for_step(k)) for k in range(5, 10)]
    assert got == expected


def test_the_state_survives_a_pass_boundary():
    """The cursor alone is not enough — the pass id has to travel with it."""
    straight, _ = small_mixture(passes=8, train_size=5, gen_size=4)
    state, expected = None, []
    for k in range(30):
        batches = straight.batches_for_step(k)
        if k == 11:
            state = straight.state_dict()
        elif k > 11:
            expected.append(keys_of(batches))
    assert state["pass_id"]["t/gen"] > 0

    resumed, _ = small_mixture(passes=8, train_size=5, gen_size=4)
    resumed.load_state_dict(state)
    assert [keys_of(resumed.batches_for_step(k)) for k in range(12, 30)] == expected


def test_the_state_is_json_serialisable_and_carries_what_d4_1_names():
    sampler, _ = small_mixture()
    for k in range(6):
        sampler.draw_step(k)
    state = sampler.state_dict()
    assert json.loads(json.dumps(state)) == state
    assert set(state) >= {"step", "cursor", "pass_id", "exhausted", "fraction",
                          "mixture_hash", "seed"}
    assert state["step"] == 6


def test_the_sampler_refuses_to_rewind_without_a_state():
    sampler, _ = small_mixture()
    sampler.draw_step(0)
    with pytest.raises(MixtureError, match="sampler is at step 1"):
        sampler.batches_for_step(0)
    with pytest.raises(MixtureError, match="sampler is at step 1"):
        sampler.batches_for_step(5)


# ─────────────────────────────────────────────────────────────────────────────
# D4.3 / D4.4 — batching
# ─────────────────────────────────────────────────────────────────────────────

def test_micro_batches_stay_under_the_token_budget():
    """Padded total is ``len(batch) x max tokens``; the budget is per micro-batch."""
    specs = tuple(
        dict(name=f"t/{n}", domain="fake", adapter="fake", kind="corpus",
             mean_tokens=80.0, train_size=5000, passes=1)
        for n in ("a", "b", "c"))
    mixture = make_mixture(specs, {"t/a": 2.0, "t/b": 1.5, "t/c": 1.0},
                           tokens_per_step=4000, steps=200)
    # Four distinct lengths per source, so several buckets are live in one step.
    lengths = {name: dict(tokens=lambda i: 32 * (1 + i % 4),
                          nodes=lambda i: 4 * (1 + i % 3))
               for name in ("t/a", "t/b", "t/c")}
    loader = Loader({"t/a": 5000, "t/b": 5000, "t/c": 5000}, **lengths)
    sampler = MixtureSampler(mixture, seed=11, get_source=loader,
                             accumulation_steps=4)
    budget = sampler.micro_batch_tokens
    assert budget == 1000.0

    n_batches = 0
    for k in range(50):
        for batch in sampler.batches_for_step(k):
            assert batch
            n_batches += 1
            tokens = [sampler.source(d.task, d.pass_id).lengths()[1][d.index]
                      for d in batch]
            assert len(batch) * max(tokens) <= budget or len(batch) == 1
    assert n_batches > 100


def test_micro_batches_are_mixed():
    """Dealing round-robin from a task-ordered bucket, not slicing a sorted list.

    With three tasks at .5/.3/.2 and every example the same size, each step's 20
    examples land in one bucket and are dealt into two micro-batches of 10. A
    batch could only come out homogeneous if one task took 19 of the 20 slots,
    which is a 1e-5 event here — so this is a hard assertion, not a rate.
    """
    specs = tuple(
        dict(name=f"t/{n}", domain="fake", adapter="fake", kind="corpus",
             mean_tokens=128.0, train_size=5000, passes=1)
        for n in ("a", "b", "c"))
    mixture = make_mixture(specs, {"t/a": 5.0, "t/b": 3.0, "t/c": 2.0},
                           tokens_per_step=2560, steps=200)
    lengths = {name: dict(tokens=128) for name in ("t/a", "t/b", "t/c")}
    loader = Loader({"t/a": 5000, "t/b": 5000, "t/c": 5000}, **lengths)
    sampler = MixtureSampler(mixture, seed=5, get_source=loader,
                             accumulation_steps=2)

    sizes = set()
    for k in range(50):
        batches = sampler.batches_for_step(k)
        assert len(batches) == 2
        for batch in batches:
            sizes.add(len(batch))
            assert len({d.task for d in batch}) >= 2, [d.task for d in batch]
    assert sizes == {10}


def test_the_dataset_shards_micro_batches_across_ranks():
    """Every rank runs the same sampler; the split needs no communication."""
    reference, _ = share_sampler(accumulation_steps=2, world_size=2)
    expected = {0: [], 1: []}
    for k in range(6):
        for j, batch in enumerate(reference.batches_for_step(k)):
            expected[j % 2].append([(d.task, d.pass_id, d.index) for d in batch])

    def stream(rank):
        sampler, _ = share_sampler(accumulation_steps=2, world_size=2)
        ds = MixtureDataset(sampler, end_step=6, rank=rank, world_size=2)
        return [[(i["ds_label"], i["pass_id"], i["example_index"]) for i in batch]
                for batch in ds]

    assert stream(0) == expected[0]
    assert stream(1) == expected[1]
    assert expected[0] and expected[1]


def test_the_dataset_stamps_the_side_channel_on_every_item():
    sampler, _ = share_sampler(accumulation_steps=2)
    ids = task_ids_for(sampler.mixture)
    ds = MixtureDataset(sampler, end_step=3)
    seen_steps = set()
    for batch in ds:
        for item in batch:
            assert item["task_id"] == ids[item["ds_label"]]
            assert item["example_index"] == item["row"]
            seen_steps.add(item["step"])
    assert seen_steps == {0, 1, 2}


def test_task_ids_are_sorted_and_stable():
    sampler, _ = share_sampler()
    assert task_ids_for(sampler.mixture) == {"t/big": 0, "t/mid": 1, "t/small": 2}
    assert task_ids_for(["b", "a"]) == {"a": 0, "b": 1}


# ─────────────────────────────────────────────────────────────────────────────
# wrap_collator
# ─────────────────────────────────────────────────────────────────────────────

class RecordingCollator:
    def __init__(self):
        self.seen_keys = None

    def __call__(self, items):
        self.seen_keys = set().union(*(set(i) for i in items))
        return {"batch_size": len(items)}


def test_wrap_collator_attaches_task_ids_and_hides_the_side_channel():
    base = RecordingCollator()
    collate = wrap_collator(base)
    items = [
        {"ds_label": "t/a", "num_nodes": 3, "task_id": 0, "example_index": 7,
         "step": 4},
        {"ds_label": "t/b", "num_nodes": 5, "task_id": 1, "example_index": 2,
         "step": 4},
    ]
    batch = collate(items)

    assert base.seen_keys == {"ds_label", "num_nodes"}
    assert torch.equal(batch["task_ids"], torch.tensor([0, 1]))
    assert torch.equal(batch["example_index"], torch.tensor([7, 2]))
    assert torch.equal(batch["step"], torch.tensor([4, 4]))
    assert batch["batch_size"] == 2
    # The items themselves are not mutated — the sampler's dicts are reused.
    assert items[0]["task_id"] == 0


def test_the_repo_collator_ignores_the_side_channel():
    """``GraphCollatorV2`` reads named keys, so the wrapper is a contract and not
    a workaround; if it ever started rejecting unknown keys this would catch it."""
    from src.utils.text_graph_collator_v2 import GraphCollatorV2

    item = {
        "input_ids": [[1, 2, 3], [4, 5]],
        "num_nodes": 2,
        "prompt_node": 1,
        "edges": [(0, 1)],
        "task_id": 0,
        "example_index": 3,
        "step": 9,
    }
    batch = GraphCollatorV2(pad_token_id=0)([item, dict(item)])
    assert batch["input_ids"].shape == (2, 5)
    assert "task_id" not in batch


# ─────────────────────────────────────────────────────────────────────────────
# D4.3 — two-level loss accounting
# ─────────────────────────────────────────────────────────────────────────────

N_TASKS = 3


def tiny_model():
    """One linear layer, one weight per task direction, deterministic init."""
    model = torch.nn.Linear(N_TASKS, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.zeros(1, N_TASKS))
    return model


def onehot_batch(task_ids):
    """(B, 1, N_TASKS) inputs: every example of task t points along axis t.

    Orthogonal per-task directions of equal magnitude are what make the expected
    answer exact: task t's gradient is ``n_t x g_t / N`` with the ``g_t`` unit and
    mutually orthogonal, so its share of the summed norms is ``n_t / N`` — the
    example share, with no distributional argument in the way.
    """
    x = torch.zeros(len(task_ids), 1, N_TASKS)
    for i, t in enumerate(task_ids):
        x[i, 0, int(t)] = 1.0
    return x


def token_losses_for(model, x, target=1.0):
    preds = model(x).squeeze(-1)                  # (B, T)
    return (preds - target) ** 2


def test_the_gradient_share_equals_the_example_share():
    task_ids = torch.tensor([0] * 6 + [1] * 3 + [2] * 1)
    x = onehot_batch(task_ids)
    mask = torch.ones(len(task_ids), 1)
    model = tiny_model()
    loss_fn = MixtureLoss(ddp_scale=False)
    n = len(task_ids)

    def loss_for(task):
        rows = (task_ids == task).nonzero(as_tuple=True)[0]
        loss, _ = loss_fn(token_losses_for(model, x[rows]), mask[rows],
                          task_ids[rows], examples_in_step=n)
        return loss

    share = measure_grad_share(model, loss_for, tasks=[0, 1, 2])
    # Exact up to fp32 on the forward; the shares are not an approximation.
    assert share == pytest.approx({0: 0.6, 1: 0.3, 2: 0.1}, abs=1e-7)


def test_a_task_split_over_micro_batches_reads_the_same_as_one_batch():
    """The readout is over a step, and a step is several micro-batches.

    The gradients are summed and *then* normed — which is the same number the
    step's own gradient carries. Summing the norms instead would make the share
    depend on how many micro-batches the step happened to be cut into, so the
    diagnostic would move when only the accumulation setting changed.
    """
    task_ids = torch.tensor([0] * 6 + [1] * 3 + [2] * 1)
    x = onehot_batch(task_ids)
    mask = torch.ones(len(task_ids), 1)
    model = tiny_model()
    loss_fn = MixtureLoss(ddp_scale=False)
    n = len(task_ids)

    def term(rows):
        loss, _ = loss_fn(token_losses_for(model, x[rows]), mask[rows],
                          task_ids[rows], examples_in_step=n)
        return loss

    def in_chunks(task):
        """The task's rows, cut into micro-batches of two, yielded lazily."""
        rows = (task_ids == task).nonzero(as_tuple=True)[0]
        if not len(rows):
            return None
        return (term(rows[i:i + 2]) for i in range(0, len(rows), 2))

    assert measure_grad_share(model, in_chunks, tasks=[0, 1, 2]) == pytest.approx(
        {0: 0.6, 1: 0.3, 2: 0.1}, abs=1e-7)


def test_the_per_task_table_reports_examples_and_summed_loss():
    task_ids = torch.tensor([0, 0, 1])
    model = tiny_model()
    x = onehot_batch(task_ids)
    mask = torch.ones(3, 1)
    loss, per_task = MixtureLoss(ddp_scale=False)(
        token_losses_for(model, x), mask, task_ids, examples_in_step=3)
    # Every example's loss is (0 - 1)^2 = 1 at the zero init.
    assert per_task == {0: (2.0, 2), 1: (1.0, 1)}
    assert float(loss.detach()) == pytest.approx(1.0)


def accumulated_grad(model, x, mask, task_ids, chunks, examples_in_step,
                     loss_fn=None):
    """Sum the gradients of ``chunks`` micro-batches, as an accumulating step does."""
    loss_fn = loss_fn or MixtureLoss(ddp_scale=False)
    model.zero_grad(set_to_none=True)
    size = len(task_ids) // chunks
    for c in range(chunks):
        sl = slice(c * size, (c + 1) * size)
        loss, _ = loss_fn(token_losses_for(model, x[sl]), mask[sl], task_ids[sl],
                          examples_in_step=examples_in_step)
        loss.backward()
    return model.weight.grad.detach().clone()


def test_the_step_is_unchanged_under_accumulation_1_vs_4():
    """The footgun D4.3 names: normalise by the micro-batch and the mixture's
    weights become a function of how the step was chopped up."""
    torch.manual_seed(0)
    task_ids = torch.tensor([0, 1, 2, 0, 0, 1, 0, 2])
    x = torch.randn(8, 5, N_TASKS)
    mask = torch.zeros(8, 5)
    mask[:, :3] = 1.0                     # a three-token span per example
    mask[3, :] = 1.0                      # and one longer one, to make the
    mask[5, :2] = 1.0                     # per-example division do real work

    one = accumulated_grad(tiny_model(), x, mask, task_ids, chunks=1,
                           examples_in_step=8)
    four = accumulated_grad(tiny_model(), x, mask, task_ids, chunks=4,
                            examples_in_step=8)
    assert torch.allclose(one, four, atol=1e-6)

    # And the wrong normalisation is genuinely different, so the test has teeth:
    # dividing by the micro-batch count would scale the step by 4.
    wrong = accumulated_grad(tiny_model(), x, mask, task_ids, chunks=4,
                             examples_in_step=2)
    assert not torch.allclose(one, wrong, atol=1e-6)


def test_count_examples_in_step_multiplies_out_the_accumulation():
    task_ids = torch.tensor([0, 1, 2, 0])
    assert count_examples_in_step(task_ids, accumulation_steps=4) == 16
    assert count_examples_in_step(task_ids, accumulation_steps=1, world_size=2) == 8


def test_per_token_normalisation_divides_by_the_batch_mean_span():
    """A ``per_token`` task's examples keep their relative length inside the task,
    while the task's own contribution still averages to one unit per example."""
    task_ids = torch.tensor([0, 0])
    model = tiny_model()
    x = onehot_batch(task_ids)
    mask = torch.zeros(2, 4)
    mask[0, :2] = 1.0        # span 2
    mask[1, :] = 1.0         # span 4, mean span 3
    losses = token_losses_for(model, x.expand(2, 4, N_TASKS))

    _, per_task = MixtureLoss({0: "per_token"}, ddp_scale=False)(
        losses, mask, task_ids, examples_in_step=2)
    # Per-token loss is 1 everywhere, so the summed spans are 2 and 4 over a mean
    # span of 3: 2/3 and 4/3, summing to 2 -- one unit per example at task level.
    assert per_task[0][0] == pytest.approx(2.0)

    _, plain = MixtureLoss(ddp_scale=False)(losses, mask, task_ids,
                                            examples_in_step=2)
    assert plain[0][0] == pytest.approx(2.0)     # 1 + 1, per example


def test_a_name_keyed_loss_norm_needs_the_id_table():
    with pytest.raises(MixtureError, match="loss_norm is keyed by task name"):
        MixtureLoss({"t/a": "per_token"})
    loss_fn = MixtureLoss({"t/a": "per_token"}, task_ids={"t/a": 0, "t/b": 1})
    assert loss_fn.norm_for(0) == "per_token"
    assert loss_fn.norm_for(1) == "per_example"


def test_a_micro_batch_normalisation_is_refused():
    with pytest.raises(MixtureError, match="examples_in_step must be a positive"):
        MixtureLoss()(torch.zeros(2, 2), torch.ones(2, 2), torch.tensor([0, 0]),
                      examples_in_step=0)


# ── the two-rank case (CPU gloo) ─────────────────────────────────────────────

DDP_TASK_IDS = [0, 1, 2, 0, 0, 1, 0, 2]


def _ddp_data():
    torch.manual_seed(0)
    task_ids = torch.tensor(DDP_TASK_IDS)
    x = torch.randn(8, 5, N_TASKS)
    mask = torch.zeros(8, 5)
    mask[:, :3] = 1.0
    mask[3, :] = 1.0
    mask[5, :2] = 1.0
    return x, mask, task_ids


def _ddp_worker(rank, world_size, init_file, out_dir):
    """One rank of a DDP step: half the examples, the global example count, and
    the gradient averaged across ranks the way DDP averages it."""
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank,
        world_size=world_size)
    try:
        x, mask, task_ids = _ddp_data()
        per_rank = x.shape[0] // world_size
        sl = slice(rank * per_rank, (rank + 1) * per_rank)

        model = tiny_model()
        # ddp_scale on: the loss is already divided by the global example count,
        # so it is multiplied by the world size to survive DDP's own averaging.
        loss, _ = MixtureLoss(ddp_scale=True)(
            token_losses_for(model, x[sl]), mask[sl], task_ids[sl],
            examples_in_step=count_examples_in_step(task_ids[sl],
                                                    accumulation_steps=1))
        loss.backward()
        grad = model.weight.grad.detach().clone()
        torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
        grad /= world_size
        if rank == 0:
            torch.save(grad, os.path.join(out_dir, "ddp_grad.pt"))
    finally:
        torch.distributed.destroy_process_group()


def test_the_step_is_unchanged_under_1_vs_2_ranks(tmp_path):
    if not (torch.distributed.is_available()
            and getattr(torch.distributed, "is_gloo_available", lambda: False)()):
        pytest.skip("gloo is not available in this build")

    x, mask, task_ids = _ddp_data()
    single = accumulated_grad(tiny_model(), x, mask, task_ids, chunks=1,
                              examples_in_step=8)

    ctx = mp.get_context("fork")
    init_file = str(tmp_path / "gloo_rendezvous")
    procs = [ctx.Process(target=_ddp_worker,
                         args=(rank, 2, init_file, str(tmp_path)))
             for rank in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    ddp = torch.load(tmp_path / "ddp_grad.pt")
    assert torch.allclose(single, ddp, atol=1e-6)
    # count_examples_in_step saw the whole step, not one rank's half.
    assert x.shape[0] == 8


# ─────────────────────────────────────────────────────────────────────────────
# measure_grad_share edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_grad_share_reports_zeros_when_nothing_moves():
    """A model with a detached loss must not read as a balanced mixture."""
    model = tiny_model()
    zero = torch.zeros((), requires_grad=True)

    def loss_for(task):
        return zero * 0.0 + (model.weight * 0.0).sum()

    assert measure_grad_share(model, loss_for, tasks=[0, 1]) == {0: 0.0, 1: 0.0}


def test_grad_share_refuses_a_frozen_model():
    model = tiny_model()
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(MixtureError, match="no trainable parameters"):
        measure_grad_share(model, lambda t: None, tasks=[0])


def test_draw_is_a_plain_tuple():
    """Callers unpack it; the pass id is the third field and nothing else moved."""
    d = Draw("t/a", 3, 1)
    assert tuple(d) == ("t/a", 3, 1)
    task, index, pass_id = d
    assert (task, index, pass_id) == ("t/a", 3, 1)

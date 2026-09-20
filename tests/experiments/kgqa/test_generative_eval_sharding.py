"""
The generative eval's rank sharding (E5, TODO_cwq).

Under torchrun every rank used to generate the *whole* split and throw away all
but one copy of the answer. That is N times the wall clock and N times the host
memory at the run's peak, and CWQ's single-rank headline already peaked near
183 GB inside this loop — which is what makes the redundancy a correctness
problem for the job, not just a waste.

The sharded loop is only admissible if it moves no number: the metrics are
macro-means over per-question scores, so scoring each question on exactly one
rank and averaging over the gathered union has to reproduce the single-rank
value exactly. That equality is what these tests pin, alongside the partition
property it rests on.
"""

import types

import pytest
import torch

from src.experiments.kgqa import evaluate as ev


QUESTION_END = [99]


class _Graph:
    def __init__(self, i):
        self.graph = {"gold_answers": [f"a{i}"]}


class _Dataset:
    """Ten questions whose prompt node carries the question index in token 0."""

    def __init__(self, n=10):
        self.n = n
        self.graphs = [_Graph(i) for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"prompt_node": 0, "input_ids": [[i, 2, 99]], "labels": [[0, 0, 0]]}


class _Tokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        i = int(ids[0]) - 1000
        # Right on every third question: 4 of 10, so the mean is not 0 or 1 and
        # a shard boundary that dropped or double-counted one would show.
        return f"a{i}" if i % 3 == 0 else "somethingelse"


class _Model:
    """Answers from the index in token 0; no collectives, as the real one has none."""

    training = False
    config = types.SimpleNamespace(graph_attn_impl="eager")

    def eval(self):
        pass

    def train(self):
        pass

    def generate(self, input_ids=None, **kw):
        i = int(input_ids[0][0])
        return torch.tensor([[i, 2, 99, 1000 + i]])


def _collate(items):
    return {"input_ids": torch.tensor(items[0]["input_ids"])}


def _run(rank, world, gather, max_samples=None):
    """One rank's generative_eval, with the distributed surface faked out."""
    patches = {
        "is_available": lambda: world > 1,
        "is_initialized": lambda: world > 1,
        "get_rank": lambda: rank,
        "get_world_size": lambda: world,
        "all_gather_object": gather,
    }
    saved = {k: getattr(torch.distributed, k) for k in patches}
    for k, v in patches.items():
        setattr(torch.distributed, k, v)
    saved_group = ev._score_gather_group
    ev._score_gather_group = lambda: None            # no real process group here
    try:
        return ev.generative_eval(
            _Model(), _Dataset(), _Tokenizer(), _collate, QUESTION_END,
            device=torch.device("cpu"), max_samples=max_samples,
            prefix="eval", answer_sep=",")
    finally:
        ev._score_gather_group = saved_group
        for k, v in saved.items():
            setattr(torch.distributed, k, v)


def _sharded(world, max_samples=None):
    """Run every rank, then re-run rank 0 with the real gathered buckets."""
    collected = []

    def collect(parts, buckets, group=None):
        collected.append([list(b) for b in buckets])
        parts[:] = [buckets] * len(parts)

    for r in range(world):
        _run(r, world, collect, max_samples)

    def replay(parts, buckets, group=None):
        parts[:] = collected

    return _run(0, world, replay, max_samples)


def test_single_rank_scores_every_question():
    single = _run(0, 1, None)

    assert single["eval_f1"] == pytest.approx(0.4)


@pytest.mark.parametrize("world", [2, 3, 4, 8])
def test_sharded_metrics_equal_the_single_rank_metrics(world):
    """The whole premise: sharding moves no number, at any world size.

    World 3 and 8 leave ragged shards over ten questions (4/3/3 and 2/1/1/...),
    which is the case a naive equal-split would get wrong.
    """
    single = _run(0, 1, None)

    assert _sharded(world) == single


def test_the_shards_partition_the_index_list_exactly_once():
    indices = list(ev.eval_indices(10, None))

    for world in (2, 3, 4, 8):
        covered = sorted(i for r in range(world) for i in indices[r::world])
        assert covered == indices


def test_sharding_respects_the_seeded_subsample():
    """The in-training path caps the split; ranks must shard that same draw."""
    single = _run(0, 1, None, max_samples=6)

    assert _sharded(3, max_samples=6) == single


def test_the_gather_group_is_built_once_and_reused(monkeypatch):
    """A group per eval call would leak one per checkpoint over a long run."""
    ev._SCORE_GATHER_GROUP = None
    calls = []
    monkeypatch.setattr(torch.distributed, "new_group",
                        lambda **kw: calls.append(kw) or "group")
    try:
        assert ev._score_gather_group() == "group"
        assert ev._score_gather_group() == "group"
        assert calls == [{"backend": "gloo"}]
    finally:
        ev._SCORE_GATHER_GROUP = None

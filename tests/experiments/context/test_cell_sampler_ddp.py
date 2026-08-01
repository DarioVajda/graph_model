"""Pin ``CellGroupedSampler``'s two contracts, both of which fail silently when broken.

**Cell homogeneity.** Every consecutive ``batch_size`` window must be one (N, T) cell,
or the collator pads to the longest row and the compiled flex kernel sees a shape per
batch instead of a shape per cell.

**It must NOT shard.** This is the correction to an earlier version of this file, which
asserted the opposite and was wrong. HF wraps the training dataloader in
``accelerator.prepare``, and accelerate re-shards any dataloader whose sampler is not
already a ``DistributedSampler`` — round-robin, batch *i* to rank ``i % world_size``. A
rank-aware sampler is therefore sharded TWICE: each rank gets ``n_train / world_size**2``
graphs and the run silently trains on a fraction of the dataset.

That shipped. It was caught only because HF's progress bar read ``0/2000`` against a
recipe implying 4000 optimizer steps — at ``n_train=16000``, ``accum=4``, 2 ranks,
2 epochs, the halved figure is exactly what double sharding predicts. Nothing else
would have flagged it; the run would have completed and reported plausible numbers
from half the data.

What the sampler still owns is ORDER: waves of ``world_size`` consecutive groups from
one cell, so accelerate's round-robin lands the same (N, T) on every rank at every step.
"""

import random

import pytest

from src.experiments.context.train import CellGroupedSampler


CELLS = ([(16, 64)] * 7 + [(32, 128)] * 5 + [(128, 64)] * 4)


def _legacy_order(cells, batch_size, seed=0, epoch=0):
    """The pre-DDP implementation, verbatim — the reference for world_size == 1."""
    by_cell = {}
    for idx, cell in enumerate(cells):
        by_cell.setdefault(cell, []).append(idx)
    rng = random.Random((seed, epoch).__hash__())
    groups = []
    for _cell, indices in sorted(by_cell.items()):
        order = list(indices)
        rng.shuffle(order)
        remainder = (-len(order)) % batch_size
        if remainder:
            order += [rng.choice(indices) for _ in range(remainder)]
        groups += [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(groups)
    return [i for g in groups for i in g]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_world_size_one_is_byte_identical_to_the_legacy_sampler(batch_size):
    """Completed runs used the pre-DDP order; adding DDP must not perturb it."""
    s = CellGroupedSampler(CELLS, batch_size, seed=3, world_size=1)
    assert list(s) == _legacy_order(CELLS, batch_size, seed=3)


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_sampler_emits_the_whole_epoch_regardless_of_world_size(world_size, batch_size):
    """THE regression. Sharding here would be double-sharding once accelerate runs.

    The emitted length must not depend on world_size beyond wave padding — if it
    shrinks by a factor of world_size, the sampler is slicing by rank again.
    """
    s = CellGroupedSampler(CELLS, batch_size, seed=1, world_size=world_size)
    emitted = list(s)
    assert len(emitted) == len(s)
    span = batch_size * world_size
    expected = sum(-(-c // span) * span
                   for c in {cell: CELLS.count(cell) for cell in set(CELLS)}.values())
    assert len(emitted) == expected
    assert len(emitted) >= len(CELLS), "sampler dropped part of the epoch"


def test_every_graph_appears_at_least_once():
    """Padding may repeat within a cell, but nothing may be dropped."""
    for ws in (1, 2, 4):
        assert set(CellGroupedSampler(CELLS, 1, seed=5, world_size=ws)) == set(range(len(CELLS)))


@pytest.mark.parametrize("world_size", [2, 4])
def test_round_robin_over_the_stream_gives_every_rank_the_same_cell(world_size):
    """Simulates accelerate: batch i -> rank i % world_size, batch_size 1.

    DDP syncs per step, so a rank drawing N=16 against a peer drawing N=128 idles for
    most of it — cell lengths here span 16x and cost is superlinear in length.
    """
    order = list(CellGroupedSampler(CELLS, 1, seed=2, world_size=world_size))
    per_rank = [order[r::world_size] for r in range(world_size)]
    for step in range(min(len(p) for p in per_rank)):
        drawn = {CELLS[p[step]] for p in per_rank}
        assert len(drawn) == 1, f"step {step}: ranks got different cells {drawn}"


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_consecutive_batch_windows_are_cell_homogeneous(world_size, batch_size):
    order = list(CellGroupedSampler(CELLS, batch_size, seed=4, world_size=world_size))
    for pos in range(0, len(order), batch_size):
        window = {CELLS[i] for i in order[pos:pos + batch_size]}
        assert len(window) == 1, f"batch at {pos} mixes cells: {window}"


def test_world_size_defaults_to_the_torchrun_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    s = CellGroupedSampler(CELLS, 1, seed=6)
    assert s.world_size == 2
    assert not hasattr(s, "rank"), "a rank attribute invites slicing by it again"


def test_no_torchrun_env_means_single_process(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    assert CellGroupedSampler(CELLS, 1, seed=6).world_size == 1

"""`tools/reports/shapes.py`: the fitted ladder is the optimum it claims to be."""

import itertools

import numpy as np
import pytest

from src.generalist.tools.reports.shapes import _apply, fit_ladder


def brute_force(values, weights, k, grid, floor, cost):
    snapped = np.maximum(floor, grid * np.ceil(values / grid).astype(np.int64))
    points = sorted(set(int(v) for v in snapped))
    top = points[-1]
    best = None
    for inner in itertools.combinations(points[:-1], min(k, len(points)) - 1):
        bounds = sorted(inner) + [top]
        padded = _apply(bounds, values)
        total = float((weights * np.array([cost(int(p)) for p in padded])).sum())
        if best is None or total < best[0] - 1e-9:
            best = (total, bounds)
    return best


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("power", [1, 2])
def test_the_fit_matches_brute_force(k, power):
    rng = np.random.default_rng(k * 10 + power)
    values = rng.lognormal(5.5, 0.9, size=300).astype(np.int64) + 1
    weights = rng.uniform(0.2, 1.0, size=300)

    def cost(v):
        return v ** power

    bounds = fit_ladder(values, weights, k, grid=128, floor=128, cost=cost)
    got = float((weights * np.array([cost(int(p)) for p in _apply(bounds, values)])).sum())
    want, _ = brute_force(values, weights, k, 128, 128, cost)
    assert got == pytest.approx(want)
    assert len(bounds) == min(k, len(set(bounds)))
    assert bounds == sorted(bounds) and bounds[-1] >= values.max()
    assert all(b % 128 == 0 for b in bounds)


def test_nothing_falls_off_the_top_and_the_floor_holds():
    values = np.array([3, 17, 40, 41, 900])
    bounds = fit_ladder(values, np.ones(5), 3, grid=16, floor=32, cost=lambda v: v * v)
    padded = _apply(bounds, values)
    assert (padded >= values).all() and min(bounds) >= 32 and bounds[-1] == 912

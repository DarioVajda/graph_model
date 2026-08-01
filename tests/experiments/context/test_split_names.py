"""Pin that every split a grid loop asks for is a split ``split_paths`` defines.

This is a boring-looking invariant that cost a real run. When ``hop_counts`` made the
evaluation grid a 3-axis product, ``grid.py`` was updated to request
``cell_split_name(n, t, k)`` but the two loops in ``flat.py`` were not — they kept
asking for ``test_n128_t512``, which no longer exists under a k mixture.

Nothing caught it. The name is only resolved at the *end* of ``run_flat_train_mode``,
after training completes, so the calibration run trained for six minutes and then died
with ``KeyError: 'test_n128_t512'``. In the real sweep that would have been hours of
training discarded at the last step.

The fix is cheap to verify without a GPU or any built data: the set of names the
consumers construct must equal the set ``split_paths`` provides.
"""

import pytest

from src.experiments.context.config import RunConfig
from src.experiments.context.process_dataset import cell_split_name, split_paths


def _requested(cfg):
    """Exactly what grid.py and both flat.py loops construct."""
    mixed = bool(cfg.hop_counts)
    return {cell_split_name(n, t, k if mixed else None)
            for (n, t) in cfg.cells() for k in cfg.hops_list()}


MIXTURE = RunConfig(hop_counts=(1, 2, 3, 4), fan_out=2,
                    node_counts=(16, 32, 64, 128), token_counts=(64, 128, 256, 512))
SINGLE_K = RunConfig(hops=4, fan_out=2, node_counts=(16, 32), token_counts=(64,))
LOOKUP = RunConfig()          # hops=0, the original star build


@pytest.mark.parametrize("cfg", [MIXTURE, SINGLE_K, LOOKUP],
                         ids=["k_mixture", "single_k", "lookup"])
def test_every_requested_split_is_defined(cfg):
    missing = _requested(cfg) - set(split_paths(cfg))
    assert not missing, f"grid loops would request undefined splits: {sorted(missing)}"


@pytest.mark.parametrize("cfg", [MIXTURE, SINGLE_K, LOOKUP],
                         ids=["k_mixture", "single_k", "lookup"])
def test_no_test_split_is_left_unrequested(cfg):
    """The converse: a built split nothing asks for is data paid for and never used."""
    defined = {n for n in split_paths(cfg) if n.startswith("test_")}
    assert not defined - _requested(cfg)


def test_mixture_grid_is_the_full_three_axis_product():
    assert len(_requested(MIXTURE)) == 16 * 4


def test_single_k_names_are_unchanged_by_the_hops_argument():
    """Single-k builds on disk must keep resolving under their historical names."""
    assert cell_split_name(32, 64) == "test_n32_t64"
    assert _requested(SINGLE_K) == {"test_n16_t64", "test_n32_t64"}


def test_mixture_names_carry_k_and_do_not_collide():
    names = _requested(MIXTURE)
    assert "test_n128_t512_h4" in names
    # The bug in one line: the un-suffixed name must NOT be what a mixture asks for.
    assert "test_n128_t512" not in names

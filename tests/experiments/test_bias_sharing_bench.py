"""CPU-side invariants of the `bias_sharing` speed benchmark.

The benchmark itself needs a GPU, but everything that decides whether its numbers
*mean* anything is CPU-checkable: that each source rebuilds the recipe its sweep
actually ran, that the `magnetic_groups` axis reaches the model config, and that
the synthetic batches carry WebQSP's token profile rather than a stand-in.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.experiments.bias_experiments.bias_sharing.bench.speed import (  # noqa: E402
    SOURCES, _backend, load_run_config, sweep_argv,
)
from src.experiments.bias_experiments.bias_sharing.bench.synth import (  # noqa: E402
    SynthSpec, build_batch, build_items, verify_against_webqsp,
)

SOURCE_NAMES = sorted(SOURCES)


# ── recipe fidelity ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_config_comes_from_the_sweep_command(source):
    """The recipe is replayed from the sweep's own argv, not a second copy."""
    module, argv = sweep_argv(SOURCES[source]["sweep"])
    assert module.startswith("src.experiments.")
    # Bookkeeping flags must be stripped: they would send this benchmark's output
    # into the training sweep's runs.jsonl.
    for flag in ("--runs-jsonl", "--run-name", "--sweep-id"):
        assert flag not in argv


@pytest.mark.parametrize("source", SOURCE_NAMES)
@pytest.mark.parametrize("groups", [0, 1, 4, 16])
def test_magnetic_groups_reaches_bias_params(source, groups):
    """The §5 wiring bug, re-pinned on the benchmark's own config path."""
    cfg = load_run_config(source, groups)
    assert cfg.magnetic_groups == groups
    bias = cfg.bias_params()
    if groups:
        assert bias["magnetic_groups"] == groups
        assert "magnetic" not in bias, "magnetic and magnetic_groups are exclusive"
    else:
        assert bias.get("magnetic") is True
        assert "magnetic_groups" not in bias


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_backend_matches_the_sweep(source):
    expected = {"synth": "flex", "webqsp": "flex", "context": "flex", "graphqa": "eager"}
    assert _backend(load_run_config(source, 0)) == expected[source]


def test_llm_floor_uses_the_matching_backend():
    """sdpa where GTLM runs flex; eager where GTLM runs eager (GraphQA)."""
    for source in SOURCE_NAMES:
        gtlm = _backend(load_run_config(source, 0))
        assert SOURCES[source]["llm_attn"] == ("sdpa" if gtlm == "flex" else "eager")


# ── synthetic batch fidelity ──────────────────────────────────────────────────

def test_token_profile_matches_webqsp():
    """Synthetic per-node token counts track WebQSP's mean and median.

    Tolerances are loose on the tail (a few thousand draws cannot reproduce a
    max-127 tail) and tight on the centre, which is what sets sequence length.
    """
    v = verify_against_webqsp(SynthSpec(n_nodes=2048), n_graphs=4)
    assert abs(v["drift"]["mean"]) < 0.10, v
    assert v["synthetic"]["median"] == v["webqsp"]["median"]
    assert abs(v["drift"]["std"]) < 0.40, v


@pytest.mark.parametrize("n", [512, 1024])
def test_batch_shapes_scale_with_node_count(n):
    spec = SynthSpec(n_nodes=n, batch_size=1, seed=0)
    batch, meta = build_batch(spec, torch.device("cpu"))

    assert meta["n_nodes"] == n
    # magnetic_m=128 truncation is what the bias einsums contract over.
    assert meta["magnetic_m"] == min(128, n)
    assert meta["node_slots"] == n                      # power-of-two node bucket
    # Flex requires block alignment; the collator's ladder must deliver it.
    assert meta["seq_len"] % 128 == 0
    assert meta["seq_len"] >= meta["real_tokens"]
    # ~3 tokens/node is the WebQSP profile this benchmark exists to reproduce.
    assert 2.5 < meta["tokens_per_node"] < 3.5


def test_every_node_and_the_prompt_are_represented():
    """node_ids must cover all N nodes, or the bias would be gathered for nodes
    that no token belongs to and the timing would measure the wrong shape."""
    spec = SynthSpec(n_nodes=512, batch_size=2, seed=1)
    batch, _ = build_batch(spec, torch.device("cpu"))
    real = batch["attention_mask"].bool()
    for row in range(batch["node_ids"].shape[0]):
        seen = torch.unique(batch["node_ids"][row][real[row]])
        assert seen.numel() == 512
    assert (batch["num_nodes"] == 512).all()


def test_spd_is_a_real_tree_metric():
    """SPD comes from the generated topology, not from noise: a tree has exactly
    one path between any two nodes, so distances are symmetric with a zero
    diagonal and no unreachable pairs."""
    item = build_items(SynthSpec(n_nodes=256, batch_size=1, seed=3))[0]
    spd = item["shortest_path_dists"]
    assert spd.shape == (256, 256)
    assert torch.equal(spd, spd.T)
    assert (torch.diagonal(spd) == 0).all()
    assert (spd[~torch.eye(256, dtype=torch.bool)] > 0).all()
    assert len(item["edges"]) == 255                    # n-1 edges = a tree


def test_seeds_give_different_graphs_but_the_same_profile():
    a = build_items(SynthSpec(n_nodes=512, seed=0))[0]
    b = build_items(SynthSpec(n_nodes=512, seed=1))[0]
    assert a["edges"] != b["edges"]
    lens = [np.mean([len(x) for x in it["input_ids"]]) for it in (a, b)]
    assert abs(lens[0] - lens[1]) < 0.5


# ── the plain-LLM floor ───────────────────────────────────────────────────────

def test_llm_causal_drops_only_the_attention_mask():
    """`llm_causal` must feed input_ids/labels but NOT attention_mask.

    Padding masks force transformers down the explicit-4D-mask path, where sdpa
    cannot use is_causal; that handicaps the floor by exactly the block skipping
    flex gets for free. This pins the one input difference between the two floors.
    """
    from src.experiments.bias_experiments.bias_sharing.bench.speed import LLM_KEYS, time_arm
    import inspect

    src = inspect.getsource(time_arm)
    assert "drop_attention_mask" in src
    assert "attention_mask" in LLM_KEYS

    captured = {}

    class FakeLoss:
        def backward(self):
            pass

        def detach(self):
            return torch.tensor(0.0)

    class FakeModel:
        config = type("C", (), {})()

        def zero_grad(self, set_to_none=True):
            pass

        def __call__(self, **kw):
            captured.setdefault("keys", set()).update(kw)
            return type("O", (), {"loss": FakeLoss()})()

    batch = {"input_ids": torch.zeros(1, 8, dtype=torch.long),
             "attention_mask": torch.ones(1, 8, dtype=torch.long),
             "labels": torch.zeros(1, 8, dtype=torch.long),
             "extra_graph_tensor": torch.zeros(1)}

    if not torch.cuda.is_available():
        pytest.skip("time_arm uses CUDA events")

    time_arm(FakeModel(), [batch], plain_llm=True, warmup_passes=0, passes=1,
             drop_attention_mask=True)
    assert captured["keys"] == {"input_ids", "labels"}


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_llm_causal_shares_the_llm_recipe(source):
    """Both floors are the same model on the same backend — only the mask differs."""
    from src.experiments.bias_experiments.bias_sharing.bench.speed import ARMS
    assert "llm_causal" in ARMS
    assert SOURCES[source]["llm_attn"] in ("sdpa", "eager")


# ── bias_modes: gather vs scatter decomposition ───────────────────────────────

def test_bias_modes_runs_each_cell_in_a_fresh_subprocess():
    """Pins the fix for job 121672.

    Driving the grid in one process let VRAM fragmentation accumulate across
    cells: flex[none] forwards of 0.71 ms at N=1024 vs 90.6 ms at N=2048, and OOM
    at N>=2048 where speed.py runs a whole 16-layer model. run_sweep.py:5 requires
    a fresh subprocess per cell. If someone re-inlines run_isolation to share the
    flex compile cache, this fails.
    """
    import inspect
    from src.experiments.bias_experiments.bias_sharing.bench import bias_modes

    src = inspect.getsource(bias_modes._one_cell)
    assert "subprocess.run" in src
    assert "bench_isolation" in src
    assert "run_isolation" not in inspect.getsource(bias_modes.run_grid)


def test_bias_modes_matches_the_production_recipe():
    """The decomposition must be measured at the settings training actually uses."""
    from src.experiments.bias_experiments.bias_sharing.bench import bias_modes

    assert bias_modes.K_HOP == 0                    # all three sweeps
    assert bias_modes.MAGNETIC_M == 128             # 002_webqsp_g_sweep
    # WebQSP measures 2.99 tokens/node; scatter contention goes as tokens-per-node²,
    # so this is the input that must not drift to a convenient round number.
    assert bias_modes.TOKENS_PER_NODE == 3
    assert bias_modes.MODES == ("none", "frozen", "full")


def test_bias_modes_defaults_to_production_node_id_dtype():
    """int64, not the flex_attn package's tuned int32.

    The collator emits long node_ids and src/models/flex_kernel.py has no cast, so
    measuring at int32 would price an optimization the model path never adopted.
    """
    from src.experiments.bias_experiments.bias_sharing.bench.bias_modes import main
    import inspect
    assert '"--node-id-dtype", default="int64"' in inspect.getsource(main)

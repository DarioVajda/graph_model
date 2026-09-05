"""
The command line, the config and the shipped run configs
(`src/generalist/config.py`, `src/generalist/__main__.py`, DESIGN.md §D8).

D8 is the layer a mistake is cheapest to catch in and most expensive to miss in:
every one of these failures is silent at submission and only visible hours later
in a log, or — worse — not visible at all.

* **``validate`` has to run on a login node.** It is the check that stands
  between a typo and a queued GPU job, so it must resolve a whole config with no
  GPU and without importing torch. That is asserted in a subprocess, because by
  the time this file runs under the full suite torch is long since in
  ``sys.modules`` and an in-process check would pass for the wrong reason.
* **The config hash is the resume's discontinuity test.** It has to be blind to
  the fields two jobs of *one* run differ in — the run name, the output
  directory, the partition a chunk landed on — and sensitive to the ones that
  make two runs different. Both directions are asserted: a hash that never moves
  is as bad as one that always does.
* **Every mode's arguments.** A missing ``--from`` must fail at parse time
  naming the flag, not at the first checkpoint read.
* **The shipped configs pass ``validate``.** They are the files that will
  actually be submitted; a config in ``configs/runs/`` or ``configs/probes/``
  that does not resolve is a broken run waiting for someone to have GPU time.
  Discovery is asserted too: a directory split is only worth having if the
  thing that walks it cannot quietly walk half of it.
* **A selection key naming ``test`` is refused** wherever it can be written —
  a training run refuses selection at all (D7.4), and a fork's own config is
  checked before the fork writes anything.

No molecule data is built here. Everything below reads the raw CSVs at most for
their digests (``build_version``), which is what ``validate`` itself does.
"""

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys

import pytest

from src.generalist import __main__ as cli
from src.generalist.config import (
    CONFIGS_DIR,
    MIXTURES,
    VALIDATOR_SETS,
    ConfigError,
    RunConfig,
    load_config_file,
    runnable_configs,
    shell_assignments,
    write_template,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The configs that ship in the repo — the ones a run is actually launched from,
#: across both `configs/runs/` and `configs/probes/`. Discovered rather than
#: listed: a config that nobody remembered to register here is exactly the one
#: that stops resolving unnoticed.
SHIPPED = runnable_configs()


def _config(**overrides) -> RunConfig:
    """A config that resolves without touching the adapter.

    ``validate()`` is deliberately not called: most tests here are about the
    hash and the parser, and the adapter's own ``validate`` pulls RDKit for no
    gain in those.
    """
    base = dict(run_name="t", mixture="smoke", validators="smoke",
                results_dir="/tmp/gen-test-results")
    base.update(overrides)
    return RunConfig(**base)


def _args(argv):
    """argv through the real parser, exactly as ``main`` builds it."""
    return cli.build_parser().parse_args(cli.normalise_argv(argv))


# ─────────────────────────────────────────────────────────────────────────────
# validate: resolves the shipped configs, on a login node, without torch
# ─────────────────────────────────────────────────────────────────────────────

def test_discovery_covers_both_directories_and_skips_the_fork_overlays():
    """The split is only safe if the walk sees all of it and none of ``forks/``.

    A fork overlay is not a ``RunConfig`` — it is a patch applied to one — so a
    walk that picked it up would fail the whole suite; a walk that missed
    ``runs/`` would pass it while validating nothing that matters.
    """
    assert SHIPPED, "no shipped configs found — the walk lost its directories"
    parents = {os.path.basename(os.path.dirname(p)) for p in SHIPPED}
    assert parents == {"runs", "probes"}
    assert any(os.path.basename(os.path.dirname(p)) == "runs" for p in SHIPPED)
    forks = os.path.join(CONFIGS_DIR, "forks")
    assert os.path.isdir(forks), "the fork overlays moved; this test is stale"
    assert not [p for p in SHIPPED if p.startswith(forks + os.sep)]


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: os.path.basename(p))
def test_shipped_configs_validate(path):
    """Every runnable shipped config resolves and passes ``RunConfig.validate``."""
    config = RunConfig(**load_config_file(path)).validate()
    assert config.mixture in MIXTURES
    assert config.validators in VALIDATOR_SETS
    # Property 2, on the file rather than on a constructed object: a flat arm
    # carrying a bias would advertise a comparison that is not happening.
    if config.arm == "flat":
        assert config.bias.strip() == "none"


def test_the_campaign_cells_differ_only_where_they_are_meant_to():
    """Six files, one recipe. This is what keeps them from drifting apart.

    There is no config inheritance, so the arm-2 campaign is six complete files
    that are copies of one another everywhere except the axes it varies: the run
    name, the seed, and — between arms — `arm`, `bias` and `tokens_per_step`.
    Any other field that comes to differ is a silent recipe change in one cell of
    a six-cell comparison, which is precisely the failure that would be read as a
    seed effect.

    Two fields are on the allowed list and both are allowed for the same reason —
    they are the knobs that *hold* the recipe equal rather than vary it, and the
    two arms need different values to arrive at the same place:

    * `tokens_per_step`, because matching the arms in examples requires it to
      differ — a flat example is ~3.5x shorter (`..._flat_s0.jsonc`).
    * `accumulation_steps`, because it only sets `micro_batch_tokens` and so
      changes nothing about which examples a step draws or what gradient it
      produces (D4.4). The graph arm needs 16 to fit one card; the flat arm has
      no such problem and 8 keeps its micro-batches from getting pointlessly
      small.

    Both are still asserted single-valued *within* an arm, which is where a
    genuine drift between seeds would show up.
    """
    cells = {os.path.basename(p): RunConfig(**load_config_file(p))
             for p in SHIPPED
             if os.path.basename(p).startswith("001_molecule_generalist_")}
    assert len(cells) == 6, f"expected six campaign cells, found {sorted(cells)}"

    varies = {"run_name", "seed", "arm", "bias", "tokens_per_step",
              "accumulation_steps"}
    reference = next(iter(cells.values()))
    for name, config in cells.items():
        for spec in dataclasses.fields(RunConfig):
            if spec.name in varies:
                continue
            assert getattr(config, spec.name) == getattr(reference, spec.name), (
                f"{name} differs from the campaign recipe in {spec.name!r}")

    seeds = {(c.arm, c.seed) for c in cells.values()}
    assert seeds == {(a, s) for a in ("graph", "flat") for s in (0, 1, 2)}
    # Within an arm the token budget is one number; across arms it must not be,
    # because the arms are matched in examples and a flat example is ~3.5x shorter.
    for arm in ("graph", "flat"):
        for field in ("tokens_per_step", "accumulation_steps"):
            values = {getattr(c, field) for c in cells.values() if c.arm == arm}
            assert len(values) == 1, f"{arm} cells disagree on {field}: {values}"
    assert ({c.tokens_per_step for c in cells.values() if c.arm == "graph"} !=
            {c.tokens_per_step for c in cells.values() if c.arm == "flat"})

    # The arms must still land on the same micro-batch after their two knobs are
    # combined — that is the quantity the OOM was about, and the only reason
    # `accumulation_steps` is allowed to differ at all.
    micro = {c.arm: c.tokens_per_step / c.accumulation_steps for c in cells.values()}
    assert micro["graph"] == 1024, (
        f"the graph arm's micro-batch is {micro['graph']} tokens; 2048 is the "
        "value that OOMed a 178 GB card at step 20")


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: os.path.basename(p))
def test_validate_mode_prints_a_mixture_table(path, capsys):
    assert cli.main(["validate", "--config", path]) == 0
    out = capsys.readouterr().out
    config = RunConfig(**load_config_file(path))
    assert config.run_name in out
    assert config.config_hash() in out
    assert "mixture" in out
    # Every task the config will train on is named, whether or not the build
    # manifest exists yet (before `data_prep` the shares print, not the budget).
    for entry in config.mixture_entries():
        assert entry["name"] in out
    # And every validator, with the cadence it will actually fire at.
    for spec in config.validator_specs():
        assert spec["name"] in out


def test_validate_imports_neither_torch_nor_transformers(tmp_path):
    """The login-node property, checked where it is checkable.

    In-process this would pass for the wrong reason — torch is already imported
    by the rest of the suite — so it runs in a fresh interpreter and asserts on
    that interpreter's ``sys.modules``.
    """
    code = (
        "import sys, json;"
        "sys.argv = ['x'];"
        "from src.generalist.__main__ import main;"
        "rc = main(['validate', '--config', %r]);"
        "print(json.dumps({'rc': rc, 'heavy': sorted("
        "m for m in ('torch', 'transformers', 'peft', 'accelerate')"
        " if m in sys.modules)}))" % SHIPPED[0]
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["rc"] == 0
    assert report["heavy"] == [], (
        f"validate imported {report['heavy']}; it has to resolve a config on a "
        "login node, and it is the check that stands between a typo and a queued "
        "GPU job")


def test_validate_print_shell_is_the_chain_scripts_only_python_call(tmp_path):
    """``--print-shell`` emits assignments a shell can eval, and nothing else."""
    config = _config(run_name="r", partition="frida", chunk_time="06:00:00",
                     chunks=4).validate()
    text = shell_assignments(config)
    values = {}
    for line in text.splitlines():
        key, _, raw = line.partition("=")
        assert key.startswith("GEN_")
        assert raw.startswith("'") and raw.endswith("'")
        values[key] = raw[1:-1]
    assert values["GEN_RUN_NAME"] == "r"
    assert values["GEN_TIME"] == "06:00:00"
    assert values["GEN_CHUNKS"] == "4"
    assert values["GEN_RUN_DIR"] == config.run_dir()
    assert values["GEN_CONFIG_HASH"] == config.config_hash()

    # Single-quoted, so nothing in a config can become a command.
    hostile = _config(run_name="r'; touch /tmp/gen_pwned; echo '").validate()
    assert "'\"'\"'" in shell_assignments(hostile)

    # The mode prints those lines and stops — no registry, no adapter, no table.
    out = subprocess.run(
        [sys.executable, "-m", "src.generalist", "validate",
         "--config", SHIPPED[0], "--print-shell"],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-4000:]
    assert out.stdout.strip().splitlines()[0].startswith("GEN_RUN_NAME=")
    assert "mixture" not in out.stdout


# ─────────────────────────────────────────────────────────────────────────────
# The config hash (D8.2)
# ─────────────────────────────────────────────────────────────────────────────

#: Fields that must not move the hash, with a value that differs from the
#: default. Every one of them differs between two jobs of the *same* run: a
#: chain's second chunk, a re-submission on another partition, a rename.
INVARIANT = {
    "run_name": "some_other_name",
    "output_dir": "/tmp/somewhere/else",
    "results_dir": "/tmp/another/results",
    "partition": "dev",
    "account": "other",
    "gpus": "H100",
    "gpus_per_config": 4,
    "cpus": 32,
    "mem": "256G",
    "chunk_time": "01:00:00",
    "chunks": 12,
    "chain_dependency": "afterok",
    "container": "/shared/other.sqsh",
    "inductor_cache": "/tmp/cache",
}

#: Fields that must move it: two runs differing in any of these are two runs.
SENSITIVE = {
    "lr": 1e-4,
    "bias_lr": 5e-3,
    "lr_min": 1e-5,
    "tokens_per_step": 8192,
    "task_weights": "mol/bace=0.9",
    "mixture": "molecule_generalist",
    "validators": "default",
    "arm": "flat",
    "seed": 7,
    "data_seed": 3,
    "warmup_steps": 17,
    "accumulation_steps": 2,
    "max_spd": 16,
    "lora_r": 8,
    "encoding": "levi",
    "loss_norm": "per_token",
}


@pytest.mark.parametrize("field,value", sorted(INVARIANT.items()))
def test_config_hash_ignores_the_fields_two_jobs_of_one_run_differ_in(field, value):
    base = _config()
    moved = _config(**{field: value})
    assert getattr(moved, field) != getattr(base, field)
    assert moved.config_hash() == base.config_hash(), (
        f"{field} moved the config hash; a resume would read it as a "
        "discontinuity and append a re-warm for a change in nothing")


@pytest.mark.parametrize("field,value", sorted(SENSITIVE.items()))
def test_config_hash_moves_with_what_makes_a_different_run(field, value):
    base = _config()
    # The flat arm carries no bias (Property 2); the point here is the hash, so
    # the pairing is made rather than asserted about.
    extra = {"bias": "none"} if field == "arm" else {}
    moved = _config(**{field: value}, **extra)
    assert moved.config_hash() != base.config_hash(), (
        f"{field} left the config hash where it was; two runs differing in it "
        "would share a lineage and a resume would not notice the change")


def test_config_hash_sees_the_resolved_weights_not_the_preset_name():
    """An override that changes nothing does not move the hash; a real one does."""
    base = _config()
    entries = {e["name"]: e["weight"] for e in base.mixture_entries()}
    name, weight = sorted(entries.items())[0]
    same = _config(task_weights=f"{name}={weight!r}")
    assert same.config_hash() == base.config_hash()
    assert _config(task_weights=f"{name}={weight * 2}").config_hash() \
        != base.config_hash()


def test_config_hash_is_stable_across_processes():
    """It goes into ``state.json``; a per-process hash would make resume noise."""
    code = ("import sys;"
            "from src.generalist.config import RunConfig;"
            "print(RunConfig(run_name='t', mixture='smoke', validators='smoke')"
            ".config_hash())")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    expected = RunConfig(run_name="t", mixture="smoke",
                         validators="smoke").config_hash()
    assert proc.stdout.strip() == expected


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing: every mode, and every required flag
# ─────────────────────────────────────────────────────────────────────────────

def test_every_mode_parses_its_arguments():
    assert _args(["validate", "--config", "c.jsonc"]).mode == "validate"
    assert _args(["data_prep", "--arms", "graph,flat"]).arms == "graph,flat"
    assert _args(["train", "--lr", "1e-4"]).lr == 1e-4
    assert _args(["resume", "--from", "latest"]).from_ == "latest"
    forked = _args(["fork", "--from", "ckpt", "--mode", "anneal"])
    assert (forked.from_, forked.fork_mode) == ("ckpt", "anneal")
    assert _args(["eval", "--checkpoint", "ckpt"]).checkpoint == "ckpt"
    # Every mode DESIGN.md D8.1 lists has a subparser and a function.
    assert set(cli.MODE_FUNCTIONS) == set(cli.MODES)


@pytest.mark.parametrize("argv,flag", [
    (["resume"], "--from"),
    (["fork", "--mode", "anneal"], "--from"),
    (["fork", "--from", "ckpt"], "--mode"),
    (["eval"], "--checkpoint"),
])
def test_a_missing_required_flag_is_refused_by_name(argv, flag, capsys):
    with pytest.raises(SystemExit) as exc:
        _args(argv)
    assert exc.value.code != 0
    assert flag in capsys.readouterr().err


def test_the_mode_defaults_to_train_so_the_sweep_runner_works():
    """``python -m sweep src.generalist <cfg>`` passes flags and no subcommand."""
    assert cli.normalise_argv(["--lr", "1e-4"]) == ["train", "--lr", "1e-4"]
    assert cli.normalise_argv(["resume", "--from", "x"]) == ["resume", "--from", "x"]
    assert cli.normalise_argv([]) == []
    args = _args(["--config", "c.jsonc", "--lr", "1e-4"])
    assert args.mode == "train" and args.lr == 1e-4


def test_a_flag_nobody_typed_does_not_overwrite_the_config_file(tmp_path):
    path = tmp_path / "run.jsonc"
    path.write_text(json.dumps({
        "name": "from_file", "mixture": "smoke", "validators": "smoke",
        "lr": 1.5e-4, "tokens_per_step": 4096, "max_steps": 10,
        "min_examples_per": 0, "warmup_steps": 1, "rewarm_steps": 1,
    }))
    config = cli.config_from_args(_args(["train", "--config", str(path)]))
    assert (config.run_name, config.lr, config.tokens_per_step) == \
        ("from_file", 1.5e-4, 4096)
    # An explicit flag wins over the file; everything else survives it.
    config = cli.config_from_args(
        _args(["train", "--config", str(path), "--lr", "9e-5"]))
    assert (config.lr, config.tokens_per_step) == (9e-5, 4096)
    # …and --run-id is how the sweep runner names a run.
    config = cli.config_from_args(
        _args(["train", "--config", str(path), "--run-id", "sweep_0003"]))
    assert config.run_name == "sweep_0003"


def test_an_unknown_key_in_a_config_file_is_refused_by_name(tmp_path):
    path = tmp_path / "typo.jsonc"
    path.write_text(json.dumps({"name": "t", "tokens_per_setp": 4096}))
    with pytest.raises(ConfigError) as exc:
        load_config_file(str(path))
    assert "tokens_per_setp" in str(exc.value)


def test_the_sbatch_block_is_folded_onto_the_slurm_fields(tmp_path):
    """A config states how it is submitted once, and the run record shows it."""
    path = tmp_path / "run.jsonc"
    path.write_text(json.dumps({
        "name": "t",
        "execution": {"sbatch": {"partition": "dev", "cpus": 4, "mem": "8G",
                                 "time": "02:00:00", "gpus": ["B200", "H100"],
                                 "inductor_cache": ".inductor_cache/x"}},
        "chain": {"chunks": 5, "dependency": "afterok"},
        "cpus": 12,
    }))
    config = RunConfig(**load_config_file(str(path)))
    assert config.partition == "dev"
    assert config.chunk_time == "02:00:00"
    assert config.gpus == "B200|H100"
    assert config.chunks == 5 and config.chain_dependency == "afterok"
    # An explicit top-level field wins over the block it duplicates.
    assert config.cpus == 12


# ─────────────────────────────────────────────────────────────────────────────
# --init
# ─────────────────────────────────────────────────────────────────────────────

def test_init_writes_a_config_that_validate_then_accepts(tmp_path, capsys):
    path = write_template("my_run", str(tmp_path))
    assert os.path.basename(path) == "my_run.jsonc"
    values = load_config_file(path)
    assert values["run_name"] == "my_run"
    RunConfig(**values).validate()
    assert cli.main(["validate", "--config", path]) == 0
    assert "my_run" in capsys.readouterr().out


def test_init_is_reachable_from_the_command_line(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROBES_DIR", str(tmp_path))
    assert cli.main(["--init", "generated"]) == 0
    assert os.path.exists(tmp_path / "generated.jsonc")


# ─────────────────────────────────────────────────────────────────────────────
# The end-of-training record
# ─────────────────────────────────────────────────────────────────────────────

class _FakeRun:
    """Just the two attributes ``_write_log_history`` reaches through."""

    def __init__(self, history):
        state = type("S", (), {"log_history": history})()
        self.trainer = type("T", (), {"state": state})()


def test_the_end_events_metrics_reach_a_file(tmp_path):
    """`end` fires after the last checkpoint, so `log_history` is its only carrier.

    HF runs ``on_train_end`` after the final save and after the progress bar is
    closed, which is how a 200-step smoke computed ``perm_spread`` and
    ``per_example`` in full and persisted neither. This is the file that keeps
    them.
    """
    config = _config(results_dir=str(tmp_path))
    os.makedirs(config.run_dir(), exist_ok=True)
    history = [{"step": 200, "loss": 0.4},
               {"step": 200, "perm_spread/mol/bace/margin_spread_max": 0.0}]
    cli._write_log_history(config, _FakeRun(history))

    with open(os.path.join(config.run_dir(), "log_history.json")) as fh:
        written = json.load(fh)
    assert written == history


def test_an_empty_history_writes_nothing(tmp_path):
    """A chunk killed before its first log leaves the parent's file alone."""
    config = _config(results_dir=str(tmp_path))
    os.makedirs(config.run_dir(), exist_ok=True)
    cli._write_log_history(config, _FakeRun([]))
    assert not os.path.exists(os.path.join(config.run_dir(), "log_history.json"))


# ─────────────────────────────────────────────────────────────────────────────
# What validate refuses
# ─────────────────────────────────────────────────────────────────────────────

def test_a_training_run_refuses_a_selection_at_all():
    with pytest.raises(Exception) as exc:
        _config(selection={"metric": "eval/mol/bace/val/roc_auc"}).validate()
    assert "select" in str(exc.value).lower()


@pytest.mark.parametrize("selection", [
    {"metric": "eval/mol/bace/test/roc_auc"},
    {"metric": "roc_auc", "split": "test"},
    {"metric": "test_roc_auc"},
])
def test_a_fork_selection_naming_test_is_refused(selection, tmp_path):
    """D7.4, checked before the fork writes anything.

    A fork *may* select — that is what an anneal is for — but never on a key
    naming the test split, wherever in the key it sits.
    """
    path = tmp_path / "fork.jsonc"
    path.write_text(json.dumps({"selection": selection}))
    args = argparse.Namespace(decay_steps=None, fork_mode="anneal",
                              fork_config=str(path))
    with pytest.raises(Exception) as exc:
        cli.load_fork_config(str(path), args, _config())
    assert "test" in str(exc.value)


def test_a_fork_may_select_on_val(tmp_path):
    path = tmp_path / "fork.jsonc"
    path.write_text(json.dumps(
        {"selection": {"metric": "eval/mol/bace/val/roc_auc", "split": "val"}}))
    args = argparse.Namespace(decay_steps=None, fork_mode="anneal",
                              fork_config=str(path))
    out = cli.load_fork_config(str(path), args, _config())
    assert out["selection"]["split"] == "val"


def test_a_fork_inherits_the_recipes_anneal_floor(tmp_path):
    """§7: an anneal decays to ``lr/10``, and that is a property of the recipe."""
    args = argparse.Namespace(decay_steps=None, fork_mode="anneal",
                              fork_config=None)
    config = _config(lr=3e-4, lr_min=3e-5, tokens_per_step=4096, seed=5)
    out = cli.load_fork_config(None, args, config)
    assert out["min_factor"] == pytest.approx(0.1)
    assert out["tokens_per_step"] == 4096 and out["seed"] == 5


@pytest.mark.parametrize("overrides,needle", [
    ({"arm": "flat"}, "flat arm"),
    ({"arm": "sideways"}, "arm"),
    ({"bias": "spd+wormhole"}, "wormhole"),
    ({"bias": "spd+spd"}, "duplicate"),
    ({"bias": "magnetic+magnetic_shared"}, "pick one"),
    ({"tokens_per_step": 0}, "tokens_per_step"),
    ({"lr_min": 3e-4}, "lr_min"),
    ({"rewarm_steps": 0}, "rewarm_steps"),
    ({"save_total_limit": 0}, "save_total_limit"),
    ({"mixture": "nope"}, "preset"),
    ({"validators": "nope"}, "preset"),
    ({"task_weights": "mol/nonexistent=0.5"}, "mol/nonexistent"),
    ({"task_weights": "mol/bace"}, "task_weights"),
    ({"chunks": 0}, "chunks"),
    ({"loss_norm": "per_molecule"}, "loss_norm"),
])
def test_validate_refuses_what_cannot_produce_a_defensible_number(overrides, needle):
    with pytest.raises(Exception) as exc:
        _config(**overrides).validate()
    assert needle in str(exc.value)


def test_the_flat_arm_is_configurable_with_no_bias():
    config = _config(arm="flat", bias="none").validate()
    assert config.bias_tokens() == []
    assert config.model_bias_config() == {}


def test_mixture_entries_carry_the_documented_block_shares():
    """`MOLECULE_GENERALIST.md` §2, computed from its own rule rather than typed."""
    entries = {e["name"]: e["weight"]
               for e in _config(mixture="molecule_generalist").mixture_entries()}
    tier_b = {n: w for n, w in entries.items()
              if n in {f"mol/{s}" for s in ("bace", "bbbp", "hiv", "tox21", "sider")}}
    assert sum(tier_b.values()) == pytest.approx(0.40)
    assert entries["mol/chebi20"] == pytest.approx(0.20)
    assert entries["mol/g2s"] == pytest.approx(0.15)
    assert sum(entries.values()) == pytest.approx(1.0)
    # §2's "roughly HIV 27 %, Tox21 37 %" of the Tier-B block.
    assert tier_b["mol/hiv"] / 0.40 == pytest.approx(0.27, abs=0.01)
    assert tier_b["mol/tox21"] / 0.40 == pytest.approx(0.37, abs=0.01)
    # Finite sources are capped at six passes; generators declare none. Six and
    # not §2's original three because the budget rule takes its horizon from the
    # *smallest* corpus — `available / share` goes as `size ** 0.5` — so at three
    # BBBP's 1,244 molecules ended the run while HIV was at 0.52 epochs.
    passes = {e["name"]: e.get("passes")
              for e in _config(mixture="molecule_generalist").mixture_entries()}
    assert passes["mol/chebi20"] == 6 and passes["mol/g2s"] is None
    assert {n: p for n, p in passes.items() if p is not None} == {
        f"mol/{s}": 6 for s in ("bace", "bbbp", "hiv", "tox21", "sider", "chebi20")
    }, "every finite corpus carries the cap, or the smallest one still binds"


# ─────────────────────────────────────────────────────────────────────────────
# The chain script (D8.3)
# ─────────────────────────────────────────────────────────────────────────────

CHAIN = os.path.join(REPO, "src", "generalist", "tools", "chain.sh")


def test_chain_writes_one_script_per_chunk_under_shared(tmp_path):
    """A dry run: the scripts are written and nothing is submitted.

    The chunk bodies are the assertion. Chunk 1 trains, every chunk after it
    resumes from the last complete checkpoint, and the dependency is ``afterany``
    — a chunk killed by the time limit exits non-zero, and that is the expected
    end of a chunk rather than a failure to stop the chain on.

    The config is a *renamed copy* of the shipped one, so the chain directory the
    script writes into is this test's and not a live run's. Running against the
    shipped name meant this test rewrote the scripts of whatever chain was
    queued under it, and once — with a chunk mid-flight — it rewrote the file
    that chunk's `bash` was still reading, which resumed at its old byte offset
    in the new text and exited 127. The directory is still the real one under
    `results/chain/`, because where the scripts live is part of what is tested.
    """
    if not os.path.exists("/usr/bin/env"):                    # pragma: no cover
        pytest.skip("no shell")
    settings = load_config_file(SHIPPED[0])
    settings["run_name"] = f"chain_selftest_{os.getpid()}"
    config_path = tmp_path / "chain_selftest.jsonc"
    config_path.write_text(json.dumps(settings))

    run_name = RunConfig(**settings).run_name
    chain_dir = os.path.join(REPO, "src", "generalist", "results", "chain", run_name)
    assert chain_dir.startswith("/shared"), (
        "job scripts live under /shared; node-local scratch is gone by the time "
        "the next chunk starts")

    env = dict(os.environ, DRY_RUN="1", CHUNKS="3",
               PYTHON=sys.executable)
    try:
        proc = subprocess.run(["bash", CHAIN, str(config_path)], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-4000:]

        first = open(os.path.join(chain_dir, "chunk_1.sh")).read()
        assert "-m src.generalist train" in first
        # A requeued first chunk falls through to a resume: `train` refuses to
        # start a second schedule beside a checkpoint that already exists.
        assert "resume --from latest" in first
        for i in (2, 3):
            body = open(os.path.join(chain_dir, f"chunk_{i}.sh")).read()
            assert "resume --from latest" in body
            assert "-m src.generalist train" not in body
    finally:
        shutil.rmtree(chain_dir, ignore_errors=True)

    assert proc.stdout.count("(dry run)") == 3
    assert "--dependency afterany:" in proc.stdout
    assert proc.stdout.count("--dependency") == 2      # not on the first chunk


def test_chain_replaces_a_chunk_script_rather_than_truncating_it(tmp_path):
    """A second invocation must not corrupt a chunk that is running.

    `bash` reads a script by byte offset as it executes, so rewriting the same
    inode under a running chunk makes it resume mid-line. Writing to a temporary
    file and renaming leaves the running shell on the old inode; the check is
    that the script's inode number changes across two invocations.
    """
    if not os.path.exists("/usr/bin/env"):                    # pragma: no cover
        pytest.skip("no shell")
    settings = load_config_file(SHIPPED[0])
    settings["run_name"] = f"chain_inode_{os.getpid()}"
    config_path = tmp_path / "chain_inode.jsonc"
    config_path.write_text(json.dumps(settings))
    chain_dir = os.path.join(REPO, "src", "generalist", "results", "chain",
                             RunConfig(**settings).run_name)
    env = dict(os.environ, DRY_RUN="1", CHUNKS="1", PYTHON=sys.executable)

    try:
        inodes = []
        for _ in range(2):
            proc = subprocess.run(["bash", CHAIN, str(config_path)], cwd=REPO,
                                  env=env, capture_output=True, text=True,
                                  timeout=600)
            assert proc.returncode == 0, proc.stderr[-4000:]
            inodes.append(os.stat(os.path.join(chain_dir, "chunk_1.sh")).st_ino)
        assert inodes[0] != inodes[1], (
            "the chunk script was rewritten in place; a chunk running off that "
            "inode would resume mid-line in the new text")
        assert not [n for n in os.listdir(chain_dir) if ".tmp." in n]
    finally:
        shutil.rmtree(chain_dir, ignore_errors=True)


def test_chain_refuses_a_config_that_does_not_validate(tmp_path):
    """Nothing is queued on a config a training job would then die on."""
    path = tmp_path / "bad.jsonc"
    path.write_text(json.dumps({"name": "bad", "lr_min": 1.0, "lr": 3e-4}))
    env = dict(os.environ, DRY_RUN="1", PYTHON=sys.executable)
    proc = subprocess.run(["bash", CHAIN, str(path)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode != 0
    assert "nothing submitted" in proc.stderr

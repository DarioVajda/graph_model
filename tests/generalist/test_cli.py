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
    config_cells,
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

#: Every *cell* of every shipped config, as ``(path, cell name)``. A campaign is
#: one file holding one cell per (arm, seed) (`configs/README.md`), so the file
#: is no longer the unit a test can check — resolving a multi-cell config once
#: would leave eleven of its twelve runs unvalidated.
SHIPPED_CELLS = [(path, cell) for path in SHIPPED for cell in config_cells(path)]


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


@pytest.mark.parametrize("path,cell", SHIPPED_CELLS, ids=lambda v: os.path.basename(v))
def test_shipped_configs_validate(path, cell):
    """Every cell of every shipped config resolves and passes ``RunConfig.validate``."""
    config = RunConfig(**load_config_file(path, cell)).validate()
    assert config.run_name == cell
    assert config.mixture in MIXTURES
    assert config.validators in VALIDATOR_SETS
    # Property 2, on the file rather than on a constructed object: a flat arm
    # carrying a bias would advertise a comparison that is not happening.
    if config.arm == "flat":
        assert config.bias.strip() == "none"


def test_task_passes_overrides_one_corpus_and_refuses_a_generator():
    """The repeat cap is per task, and only a corpus has one.

    ``task_passes`` is how a budget the mixture cannot otherwise sustain gets
    bought, so it has to be exact about what it is buying: a generator draws a
    fresh pass every time (D4.2) and has no repeats to raise, and silently
    accepting the override would put a number in the config that does nothing.
    """
    config = _config(mixture="molecule_generalist", validators="default",
                     task_passes="mol/chebi20=7")
    entries = {e["name"]: e for e in config.mixture_entries()}
    assert entries["mol/chebi20"]["passes"] == 7
    assert entries["mol/sider"]["passes"] == 6          # untouched

    with pytest.raises(ConfigError, match="generator"):
        _config(mixture="molecule_generalist", validators="default",
                task_passes="mol/g2s=7").mixture_entries()
    with pytest.raises(ConfigError, match="task_passes names"):
        _config(mixture="molecule_generalist", validators="default",
                task_passes="mol/nope=7").mixture_entries()
    with pytest.raises(ConfigError, match="not an integer"):
        _config(task_passes="mol/chebi20=seven").pass_overrides()


@pytest.mark.parametrize("scale", [0, -1.0])
def test_a_bad_budget_scale_is_refused_at_validate(scale):
    with pytest.raises(ConfigError, match="budget_scale"):
        _config(budget_scale=scale).validate()


def test_a_multi_cell_config_refuses_to_pick_a_cell_for_the_caller():
    """Naming no cell is an error, not a default.

    A campaign file holds twelve runs and resolving it without a cell would have
    to guess which one — and a wrong guess is a job that trains the wrong arm
    under the right name. The error names the cells, so the fix is in the message.
    """
    path = os.path.join(CONFIGS_DIR, "runs", "molecule_generalist.jsonc")
    cells = config_cells(path)
    assert len(cells) > 1
    with pytest.raises(ConfigError) as excinfo:
        load_config_file(path)
    assert "molecule_generalist_graph_s0" in str(excinfo.value)
    with pytest.raises(ConfigError):
        load_config_file(path, "no_such_cell")


#: The campaign's legs — one arm each — and the three seeds each runs at. The
#: extended-horizon rerun of two of them lives in `probes/` until it reports.
CAMPAIGN_LEGS = ("molecule_generalist_graph", "molecule_generalist_flat",
                 "notation_selfies", "notation_inchi")
CAMPAIGN_SEEDS = (0, 1, 2)

#: What a campaign cell may differ from its siblings in. Everything else is the
#: recipe, and a recipe that drifts in one cell of a twelve-cell comparison reads
#: as a seed effect rather than as the mistake it is.
#:
#: * `run_name`, `seed`, `arm`, `bias` — the experiment's own axes.
#: * `tokens_per_step`, because matching the arms in EXAMPLES requires it to
#:   differ: a SMILES example is ~3.5x shorter than a graph one.
#: * `accumulation_steps`, `gpus_per_config`, `chunks` — execution shape. The
#:   first only sets `micro_batch_tokens` and so changes nothing about which
#:   examples a step draws or what gradient it produces (D4.4); the other two are
#:   unhashed Slurm fields.
#: * `validators`, because the notation arms cannot run `perm_spread` — SELFIES
#:   and InChI are canonical-only, so there is no re-ordered string to rewrite.
CAMPAIGN_VARIES = {"run_name", "seed", "arm", "bias", "tokens_per_step",
                   "accumulation_steps", "gpus_per_config", "chunks",
                   "validators"}


#: What each campaign cell resolved to when its numbers were measured. A config
#: in `runs/` is a result artifact: reorganising the files is fine, but changing
#: what a cell resolves to breaks the tie between a quoted number and the run that
#: produced it, and `config_hash` is the only thing that would notice.
CAMPAIGN_HASHES = {
    "molecule_generalist_graph_s0": "297fc38e3f200f10",
    "molecule_generalist_graph_s1": "5a97f979a0edcce7",
    "molecule_generalist_graph_s2": "b0a9b083dda3be97",
    "molecule_generalist_flat_s0": "1ab2083371fb2235",
    "molecule_generalist_flat_s1": "18cbc9c64bdc8e09",
    "molecule_generalist_flat_s2": "63645809bc0e8303",
    "notation_selfies_s0": "70b360fc773a73a6",
    "notation_selfies_s1": "96b1730fea93c71d",
    "notation_selfies_s2": "1e5e797ea2228b58",
    "notation_inchi_s0": "456f0137e9bc64fb",
    "notation_inchi_s1": "6d2edee2c6459bdf",
    "notation_inchi_s2": "c2a8b450703e4c3d",
}


#: The same pin for the campaign the write-up now reports — `probes/008`, six
#: cells on instruct weights in chat formatting (`MOLECULE_GENERALIST.md` §7).
#: Every digest here is read off that cell's own line in `results/runs.jsonl`,
#: not off the config as it stands today, which is the only way the pin means
#: anything.
#:
#: A probe config rather than a `runs/` one, and pinned all the same: what makes a
#: config a result artifact is that a quoted number came out of it, not which
#: directory it sits in.
INSTRUCT_HASHES = {
    "molecule_generalist_instruct_graph_s0": "d9c37cf143837e32",
    "molecule_generalist_instruct_graph_s1": "e9696da0e0eb5e07",
    "molecule_generalist_instruct_graph_s2": "a21e0ae903631f67",
    "molecule_generalist_instruct_flat_s0": "bf34e59c2c8fd378",
    "molecule_generalist_instruct_flat_s1": "8f4cca0fffe08b5b",
    "molecule_generalist_instruct_flat_s2": "0ebe079768037ad0",
}


def test_the_instruct_campaign_still_resolves_to_the_runs_that_were_measured():
    """The six reportable cells hash to what their run records carry.

    This is the campaign `MOLECULE_GENERALIST.md` §7 reports and the one the document's headline
    numbers come from, so it gets the same guard the notation ladder has. It also
    catches the specific way a validator gets added wrong: ``validator_specs`` is
    inside `hash_payload`, so putting a new validator into an existing preset
    renames every cell that used it — which is why `text_behaviour` ships as its
    own set.
    """
    cells = config_cells(os.path.join(
        CONFIGS_DIR, "probes", "008_molecule_generalist_instruct.jsonc"))
    assert set(INSTRUCT_HASHES) <= set(cells)
    for name, digest in INSTRUCT_HASHES.items():
        assert RunConfig(**cells[name]).config_hash()[:16] == digest, (
            f"{name} no longer resolves to the run that was measured")


def test_the_campaign_still_resolves_to_the_runs_that_were_measured():
    """The twelve cells hash to what their run records carry.

    Every number in `configs/runs/molecule_generalist.md` was measured from a
    checkpoint whose record names one of these digests. Editing the file is
    allowed — merging twelve configs into it did not move a single one — but an
    edit that changes what a cell *resolves to* silently detaches the write-up
    from the runs, and nothing else in the suite would catch it.
    """
    cells = config_cells(os.path.join(CONFIGS_DIR, "runs", "molecule_generalist.jsonc"))
    # A subset, not an equality: cells added for a campaign that has not run yet
    # have no measured digest to pin, and pinning one before the run would be
    # pinning a guess.
    assert set(CAMPAIGN_HASHES) <= set(cells)
    for name, digest in CAMPAIGN_HASHES.items():
        assert RunConfig(**cells[name]).config_hash()[:16] == digest, (
            f"{name} no longer resolves to the run that was measured")


def test_the_campaign_cells_differ_only_where_they_are_meant_to():
    """One file, twelve cells, one recipe — asserted rather than assumed.

    Merging the campaign into a single config makes most of this structural: a
    field written once at the top level cannot drift between cells. What is still
    worth asserting is the other half — that the bundle and the seed axis vary
    *only* what they are meant to, that the arms are matched where the comparison
    depends on it, and that the file really does hold the twelve cells the
    write-up quotes.
    """
    path = os.path.join(CONFIGS_DIR, "runs", "molecule_generalist.jsonc")
    cells = {name: RunConfig(**values)
             for name, values in config_cells(path).items()}
    assert set(cells) == {f"{leg}_s{seed}" for leg in CAMPAIGN_LEGS
                          for seed in CAMPAIGN_SEEDS}

    def leg_of(name):
        return name.rsplit("_s", 1)[0]

    reference = next(iter(cells.values()))
    for name, config in cells.items():
        for spec in dataclasses.fields(RunConfig):
            if spec.name in CAMPAIGN_VARIES:
                continue
            assert getattr(config, spec.name) == getattr(reference, spec.name), (
                f"{name} differs from the campaign recipe in {spec.name!r}")

    # Within a leg every varying field is one number, which is where a genuine
    # drift between seeds would show up.
    for leg in CAMPAIGN_LEGS:
        members = [c for n, c in cells.items() if leg_of(n) == leg]
        assert len(members) == len(CAMPAIGN_SEEDS)
        for field in ("tokens_per_step", "accumulation_steps", "validators",
                      "arm", "budget_scale", "max_steps"):
            # (budget_scale and max_steps are uniform here; asserted anyway, so
            #  a horizon change to one cell of a leg cannot pass unnoticed.)
            values = {getattr(c, field) for c in members}
            assert len(values) == 1, f"{leg} cells disagree on {field}: {values}"
        assert {c.seed for c in members} == set(CAMPAIGN_SEEDS)

    # Across the arms of ONE horizon the token budget must not be shared, because
    # the arms are matched in examples and no two of these representations
    # measure the same tokens/example. Between horizons it may repeat: the graph
    # arm's 16384 is the same at both, and it is the flat arm's that moves,
    # because the water-filled shares shift the mean example length.
    for scale in {c.budget_scale for c in cells.values()}:
        legs = {leg_of(n): c for n, c in cells.items() if c.budget_scale == scale}
        budgets = {c.arm: c.tokens_per_step for c in legs.values()}
        assert len(set(budgets.values())) == len(budgets), (
            f"two arms at budget_scale {scale} share a token budget: {budgets}")

    # Every cell lands on the same micro-batch after its knobs are combined, to
    # within the rounding the integer budgets force. That is the quantity the OOM
    # was about, and the only reason `accumulation_steps` may differ at all.
    for name, config in cells.items():
        micro = config.tokens_per_step / (config.accumulation_steps
                                          * config.gpus_per_config)
        if config.arm == "graph":
            assert micro == 1024, (
                f"{name}'s micro-batch is {micro} tokens; 2048 is the value that "
                "OOMed a 178 GB card at step 20")
        else:
            assert 512 <= micro <= 1088, (
                f"{name} runs at {micro} micro-batch tokens, off the ~1024 every "
                "other cell holds")

    # Property 2 on every flat arm, not just the SMILES one.
    for config in cells.values():
        if config.arm != "graph":
            assert config.bias.strip() == "none"


@pytest.mark.parametrize("path,cell", SHIPPED_CELLS, ids=lambda v: os.path.basename(v))
def test_validate_mode_prints_a_mixture_table(path, cell, capsys):
    assert cli.main(["validate", "--config", path, "--cell", cell]) == 0
    out = capsys.readouterr().out
    config = RunConfig(**load_config_file(path, cell))
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


def test_the_batching_budgets_leave_an_unset_config_hash_alone():
    """`micro_batch_tokens`, `micro_batch_node_pairs` and `allow_exhaustion`
    arrived after runs were made; unset, they hash as those runs did, and
    `allow_exhaustion` never hashes since it only decides whether a run starts."""
    base = _config()
    payload = base.hash_payload()
    for name in ("micro_batch_tokens", "micro_batch_node_pairs", "allow_exhaustion"):
        assert name not in payload
    assert _config(allow_exhaustion=True).config_hash() == base.config_hash()
    assert _config(micro_batch_tokens=8192).config_hash() != base.config_hash()


def test_a_pair_budget_needs_a_token_budget():
    with pytest.raises(ConfigError, match="micro_batch_node_pairs needs"):
        _config(micro_batch_node_pairs=1 << 20).validate()


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


def _adapt_args(path, **overrides):
    """A `fork --mode adapt` argv, with every override off by default."""
    base = dict(decay_steps=None, fork_mode="adapt", fork_config=str(path),
                task=None, target_metric=None, target_value=None,
                starts=None, held_out_by=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _adapt_file(tmp_path, target, **extra):
    path = tmp_path / "adapt.jsonc"
    path.write_text(json.dumps(
        dict({"task": "mol/bace", "budget_steps": 1000, "eval_steps": 25,
              "target": target}, **extra)))
    return path


TIER_B_TARGET = {"metric": "in_mixture/mol/bace/test/roc_auc",
                 "value": 0.7792, "direction": "max", "consecutive": 3,
                 "on_test": True}


class TestAdaptTargetOverrides:
    """`--task` rewrites the metric key's task segment and nothing else, so the
    other two thirds of a target travel on their own flags. Both mismatches
    guarded here were live in `KFOLD_TRANSFER.md`'s fork configs."""

    def test_task_rewrites_the_metrics_task_segment(self, tmp_path):
        path = _adapt_file(tmp_path, TIER_B_TARGET)
        out = cli.load_fork_config(
            str(path), _adapt_args(path, task="mol/bbbp",
                                   target_value=0.6703), _config())
        assert out["target"]["metric"] == "in_mixture/mol/bbbp/test/roc_auc"
        assert out["target"]["value"] == 0.6703
        assert out["task"] == "mol/bbbp"

    def test_switching_task_without_a_threshold_is_refused(self, tmp_path):
        """A wrong threshold is worse than a missing one: it still crosses."""
        path = _adapt_file(tmp_path, TIER_B_TARGET)
        with pytest.raises(Exception) as exc:
            cli.load_fork_config(str(path),
                                 _adapt_args(path, task="mol/bbbp"), _config())
        assert "per task" in str(exc.value)

    def test_a_deferred_target_needs_no_threshold_to_switch_task(self, tmp_path):
        """An `anchor` has no number to inherit wrongly, so `--task` alone is
        enough for the eleven tasks whose threshold is still owed."""
        path = _adapt_file(tmp_path, {
            "metric": "in_mixture/mol/ring_membership/test/em_accuracy",
            "anchor": "95% of the fold B trunk", "consecutive": 3,
            "on_test": True}, task="mol/ring_membership")
        out = cli.load_fork_config(
            str(path), _adapt_args(path, task="mol/fg_count"), _config())
        assert out["target"]["metric"] == \
            "in_mixture/mol/fg_count/test/em_accuracy"

    def test_target_metric_replaces_the_whole_key(self, tmp_path):
        """g2s is `smiles` and scores roundtrip_match; ChEBI-20 is `text` and
        scores bleu2. Rewriting the task segment alone names a key ChEBI never
        emits."""
        path = _adapt_file(tmp_path, {
            "metric": "in_mixture/mol/g2s/test/roundtrip_match",
            "anchor": "owed", "consecutive": 3, "on_test": True},
            task="mol/g2s")
        out = cli.load_fork_config(str(path), _adapt_args(
            path, task="mol/chebi20",
            target_metric="in_mixture/mol/chebi20/test/bleu2"), _config())
        assert out["target"]["metric"] == "in_mixture/mol/chebi20/test/bleu2"

    def test_starts_and_held_out_by_override_the_config(self, tmp_path):
        path = _adapt_file(tmp_path, TIER_B_TARGET,
                           starts=["parent", "base"], held_out_by="C")
        out = cli.load_fork_config(
            str(path), _adapt_args(path, starts="parent", held_out_by="A"),
            _config())
        assert out["starts"] == ["parent"]
        assert out["held_out_by"] == "A"


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

CHAIN = os.path.join(REPO, "src", "generalist", "tools", "launch", "chain.sh")


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

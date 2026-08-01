"""How a resolved sweep is rendered into sbatch arguments and job scripts.

These are pure string builders, but they decide whether a submitted job runs at
all, and the two failure modes are silent rather than loud:

  * a gres count that disagrees with ``--nproc_per_node`` either wastes a card or
    hangs in NCCL waiting on a rank that has no GPU, so ``gpus_per_config`` is the
    single source of truth and a count embedded in ``gpus`` must be rejected;
  * a shared inductor cache that is exported when it does not exist would make a
    fresh clone fail to compile rather than fall back.
"""

import os

import pytest

from sweep import execute
from sweep.expand import SweepError


def sb(**kw):
    base = {"partition": "frida", "gpus": "B200"}
    base.update(kw)
    return base


# ── gres rendering ────────────────────────────────────────────────────────────

def test_single_type_renders_type_and_count():
    assert execute._gpu_args("B200", 1) == ["--gres", "gpu:B200:1"]
    assert execute._gpu_args("B200", 4) == ["--gres", "gpu:B200:4"]


def test_bare_count_renders_without_a_type():
    assert execute._gpu_args(2, 2) == ["--gres", "gpu:2"]


def test_a_list_of_types_becomes_a_count_plus_a_feature_constraint():
    """Slurm gres cannot express an OR, so a type list is a constraint."""
    assert execute._gpu_args(["B200", "B300"], 2) == [
        "--gres", "gpu:2", "--constraint", "GPU_BRD:B200|GPU_BRD:B300"]


def test_count_always_comes_from_per_config_not_from_the_type_string():
    assert execute._gpu_args("H100:1", 2) == ["--gres", "gpu:H100:2"]


@pytest.mark.parametrize("entry,expected", [
    ("H100:2", ("H100", 2)), ("H100", ("H100", None)),
    ("2", (None, 2)), ("A100_80GB:1", ("A100_80GB", 1)),
])
def test_gpu_entry_parsing(entry, expected):
    assert execute._parse_gpu_entry(entry) == expected


# ── gpus_per_config is the single source of truth ─────────────────────────────

def test_defaults_to_one():
    assert execute._gpus_per_config(sb()) == 1


def test_a_matching_count_in_gpus_is_accepted():
    """The historical "B200:1" form with the default must keep working."""
    assert execute._gpus_per_config(sb(gpus="B200:1")) == 1
    assert execute._gpus_per_config(sb(gpus=["B200:1", "B300:1"])) == 1


def test_a_conflicting_count_in_gpus_is_an_error():
    with pytest.raises(SweepError, match="gpus_per_config"):
        execute._gpus_per_config(sb(gpus="H100:2", gpus_per_config=1))
    with pytest.raises(SweepError, match="gpus_per_config"):
        execute._gpus_per_config(sb(gpus=["H100:1"], gpus_per_config=2))


def test_zero_or_negative_is_an_error():
    with pytest.raises(SweepError, match=">= 1"):
        execute._gpus_per_config(sb(gpus_per_config=0))


def test_sbatch_argv_uses_the_resolved_count(tmp_path):
    argv = execute._sbatch_argv("job", "log", "wrap", sb(gpus=["H100"], gpus_per_config=2))
    assert argv[argv.index("--gres") + 1] == "gpu:2"


def test_partition_and_gpus_are_required():
    with pytest.raises(SweepError, match="partition"):
        execute._sbatch_argv("j", "l", "w", {"gpus": "B200"})
    with pytest.raises(SweepError, match="gpus"):
        execute._sbatch_argv("j", "l", "w", {"partition": "frida"})


# ── job scripts: python vs torchrun ───────────────────────────────────────────

def test_one_gpu_invokes_plain_python(tmp_path):
    path = execute._write_job_script(str(tmp_path), "lbl", [["-m", "pkg", "--flag", "v"]], 1)
    body = open(path).read()
    assert "python -m pkg --flag v" in body
    assert "torchrun" not in body


def test_more_than_one_gpu_invokes_torchrun_with_matching_ranks(tmp_path):
    path = execute._write_job_script(str(tmp_path), "lbl", [["-m", "pkg"]], 4)
    assert "torchrun --standalone --nproc_per_node 4 -m pkg" in open(path).read()


def test_every_invocation_in_a_sequential_job_gets_the_launcher(tmp_path):
    path = execute._write_job_script(str(tmp_path), "lbl", [["-m", "a"], ["-m", "b"]], 2)
    lines = [ln for ln in open(path).read().splitlines() if "nproc_per_node" in ln]
    assert len(lines) == 2


def test_arguments_are_quoted(tmp_path):
    path = execute._write_job_script(str(tmp_path), "lbl", [["-m", "pkg", "--t", "a b"]], 1)
    assert "'a b'" in open(path).read()


# ── the shared inductor cache ─────────────────────────────────────────────────

def test_cache_is_not_exported_when_unset():
    assert not any(e.startswith("SWEEP_INDUCTOR_CACHE") for e in execute._job_env(sb()))


def test_cache_is_exported_as_an_absolute_path(tmp_path):
    env = execute._job_env(sb(inductor_cache=str(tmp_path)))
    entry = next(e for e in env if e.startswith("SWEEP_INDUCTOR_CACHE="))
    assert os.path.isabs(entry.split("=", 1)[1])


def test_relative_cache_paths_are_resolved_at_submit_time():
    env = execute._job_env(sb(inductor_cache=".inductor_cache/shared"))
    entry = next(e for e in env if e.startswith("SWEEP_INDUCTOR_CACHE="))
    assert entry.split("=", 1)[1] == os.path.abspath(".inductor_cache/shared")


def test_both_wrap_builders_carry_the_same_env(tmp_path):
    """`_srun_wrap` and `_array_wrap` must not drift apart."""
    conf = sb(inductor_cache=str(tmp_path), container="img.sqsh")
    single = execute._srun_wrap("lbl", "job.sh", conf)
    array = execute._array_wrap(["lbl"], ["job.sh"], conf)
    for token in ("PYTHONUNBUFFERED=1", "SWEEP_PROJECT_ROOT=", "SWEEP_INDUCTOR_CACHE="):
        assert token in single, token
        assert token in array, token

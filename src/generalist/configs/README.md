# Generalist run configs

Three directories, and which one a file goes in is a statement about what the
file is *for*, not about how big it is.

| directory | what lives here | the test |
|---|---|---|
| `runs/` | the campaign runs — a real generalist model, trained to its budget | someone reproducing a published number runs this file, unedited |
| `probes/` | smokes, cross-checks, hyperparameter screens, one-off measurements | it answered a question, and once answered nobody re-runs it |
| `forks/` | fork overlays (`--fork-config`), applied on top of a base config | it is not a run on its own and does not resolve as a `RunConfig` |

**Why the split.** Both kinds were sitting in one flat directory numbered in
submission order, which made the directory a chronological log rather than a
map. The two kinds have opposite lifecycles: a probe is disposable the moment
it has reported, and its value is entirely in what it wrote into the plan
documents; a campaign config is a *result artifact* — it has to keep resolving,
byte-identical, for as long as the numbers it produced are quoted. Keeping them
apart makes the second kind's obligations visible, and stops a reader from
having to guess which of nine files is the one that produced a table.

**Probes are numbered; campaigns are named.** A probe's number is part of the
record and appears throughout `MOLECULE_GENERALIST.md`, `molecules/PLAN.md` and
`results/BUILD_LOG.md` — `002` means the BACE cross-check everywhere those files
are read — and the gaps in `probes/` are the history, not an accident. A campaign
is named for what it trains rather than for when it was submitted, because it
outlives its submission order and its cells are addressed by name.

**Rules for `runs/`.** One file per **campaign**, holding one *cell* per (arm,
seed). A campaign is one experiment — the whole result is a difference between
arms measured at a recipe every arm holds fixed — and writing it as one file is
what makes that structural: the shared recipe is written once at the top level,
and a cell can differ from its siblings only in what the file's bundle and axes
actually vary. Twelve files that must stay identical everywhere except three
fields is twelve chances to edit one and not the others, and a recipe that drifts
in one cell of a twelve-cell comparison reads as a seed effect rather than as the
mistake it is.

The expansion is the sweep runner's (`sweep/README.md`): a list of scalars is an
**axis**, a list of objects is a **bundle** of keys that vary together, and the
run set is their product. **A cell's name is its `run_name` plus `_s<seed>` when
the file sweeps the seed** (`config.py::config_cells`), which is the convention
every run directory on disk already follows.

A cell is addressed by name, and one chain is one cell:

```bash
python3 -m src.generalist validate --config <file> --cells        # the names
src/generalist/tools/launch/chain.sh <file> <cell>                # submit one
```

A config holding a single run takes no `--cell`; one holding several refuses to
guess, because picking a cell for the caller is picking which run the numbers
came from. Every file in here carries its own `execution.sbatch` and `chain`
blocks, so those two commands are the whole of the launch — and where a cell
genuinely needs a different execution shape, the Slurm fields (`gpus_per_config`,
`chunks`, `accumulation_steps`) are `RunConfig` fields and go in the bundle,
because `execution` belongs to the runner and cannot vary per cell.

**What a run config owes.** It is a *result artifact*: it has to keep resolving
to the `config_hash` its run records carry, for as long as its numbers are
quoted. Reorganising the files is fine; changing what a cell resolves to is not.

**What a probe still owes.** `tests/generalist/test_cli.py` resolves every *cell*
of every file in both directories and asserts it passes `RunConfig.validate`, so
a probe that has stopped resolving is a failing test rather than a surprise at
submission time. A probe that is genuinely dead gets deleted; it does not get
left behind broken.

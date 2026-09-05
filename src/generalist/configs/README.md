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

**Numbering does not restart per directory.** The numbers are the campaign's
own record and appear throughout `MOLECULE_GENERALIST.md`,
`molecules/PLAN.md` and `results/BUILD_LOG.md`; `002` means the BACE
cross-check everywhere those files are read. The gaps in each directory are the
history, not an accident.

**Rules for `runs/`.** One file per run — per (arm, seed) — because a run is
reproduced by naming a file, never by remembering a flag. Every file in here
carries its own `execution.sbatch` and `chain` blocks, so `chain.sh <file>` is
the whole of the launch. The reasoning behind the recipe is written once, in the
seed-0 graph file; the siblings state their delta and point at it, so there is
one place to correct if a decision changes and no chance of two files disagreeing
about why.

**What a probe still owes.** `tests/generalist/test_cli.py` resolves every file
in both directories and asserts it passes `RunConfig.validate`, so a probe that
has stopped resolving is a failing test rather than a surprise at submission
time. A probe that is genuinely dead gets deleted; it does not get left behind
broken.

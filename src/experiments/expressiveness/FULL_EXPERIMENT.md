# Full expressiveness scaling experiment (multi-seed, B200)

Personal run-book / reference. **Not committed.** Written 2026-06-26.

## Why this run exists

Earlier large-graph runs (N≥500) looked like they "failed to learn" — eval accuracy
stuck at the ~0.5 marginal. Investigation showed this was **under-training**, not a
capability wall: the loss collapses to the 50/50 marginal and sits on a long plateau
before a delayed-generalization *escape*, and the plateau gets longer with graph size.

Evidence that settled it:
- N=10–25 learns cleanly to ~0.80 (sanity check).
- Controlled eager sweep (m=0, bias_lr=1e-2), 3 epochs: N=17→0.80, 50→0.72, 100→0.66,
  200→0.71, 300→0.69 — graceful degradation, **no collapse** through N=300.
- `magnetic_m=128` vs `m=0` was ~identical at N=200/300 → truncation is **not** the cause.
- N=500 with the production config (flex, m=128, bias_lr=1e-3, LoRA) flatlined at 0.54 in
  3 epochs but reached **0.754 in 12 epochs**, escaping the plateau at ~epoch 4
  (just past where the 3-epoch budget ended). See memory `project_expressiveness_plateau`.

So this experiment runs the **full size × seed × k matrix at 15 epochs** to get
multi-seed scaling curves and confirm the escape holds at 1000 and 2000 nodes.

## The matrix (18 training runs)

`sizes {500, 1000, 2000} × seeds {0,1,2} × k_hop {0,1}` — each a **single-k** run.

Fixed hyperparameters (every run):
```
--impls v2-flex --num-epochs 20 --magnetic-m 128 --bias-lr 1e-3
--lora --lora-r 8 --lr 1e-4 --train-dataset-size 2000 --eval-dataset-size 200
--eval-steps 75 --report-to GraphLLM
```
Per-size micro-batch (effective batch = bs*acc = 32 everywhere): 500/1000 use
`--batch-size 4 --accumulation-steps 8`; **2000 uses `--batch-size 2 --accumulation-steps 16`**
(the (B,H,N,N) graph bias is ~3 GB/layer/4-samples at N=2400, so batch 4 risks GPU OOM).
Eval is 200 examples every 75 steps (small + infrequent → eval no longer dominates wall-time).
Per-size flex buckets (`--len-buckets` / `--node-buckets`):
| size | len-buckets | node-buckets | node range |
|------|-------------|--------------|------------|
| 500  | 640,768 | 512,640 | 400–600 |
| 1000 | 1408,1664,1792 | 1024,1280 | 800–1200 |
| 2000 | 3200,3712,4096,4224 | 2048,2432 | 1600–2400 |

**Why single-k per process:** running `--k-hops 0,1` in one process accumulates both k's
flex variants (for 2000: 4 L × 2 N × {fwd,bwd} × {k0,k1} ≈ 32+, over the cache limit →
recompile thrash). Single-k keeps each process at ≤ ~16 variants → no thrash, no code
change to the cache limit needed.

## GPU split (8× B200 on node `ixb3`, partition `frida`)

One sbatch job per GPU; each job runs its whole sequence so the GPU is held throughout
(no chance of losing it between runs). Makespan-balanced (~13–14 h, gated by the 2000-node
runs). The six 2000-runs are spread one-per-GPU (pairing two would double a GPU to ~24 h);
their slack absorbs the six 500-runs (run **first** as a fast pipeline check). GPUs 6–7
take all six 1000-runs and reuse compiled kernels across seeds (same shape).

Each job: `--mem=224G` (host RAM — see OOM note below), `-t 36:00:00` (≈2× the ~15–17 h
estimated makespan), `--gres=gpu:B200:1`, names `gtlm_train_{i}`.

| job (jobid) | name / GPU label | sequence |
|------|-----------|----------|
| 107190 | gtlm_train_0 / gpu0 | `500/s0/k0` → `2000/s0/k0` |
| 107191 | gtlm_train_1 / gpu1 | `500/s0/k1` → `2000/s0/k1` |
| 107192 | gtlm_train_2 / gpu2 | `500/s1/k0` → `2000/s1/k0` |
| 107193 | gtlm_train_3 / gpu3 | `500/s1/k1` → `2000/s1/k1` |
| 107194 | gtlm_train_4 / gpu4 | `500/s2/k0` → `2000/s2/k0` |
| 107195 | gtlm_train_5 / gpu5 | `500/s2/k1` → `2000/s2/k1` |
| 107196 | gtlm_train_6 / gpu6 | `PREP:2000` → `1000/s0/k0` → `1000/s1/k0` → `1000/s2/k0` |
| 107197 | gtlm_train_7 / gpu7 | `1000/s0/k1` → `1000/s1/k1` → `1000/s2/k1` |

(jobids are from the 2026-06-27 relaunch; resubmitting will change them.)

## 2000-node gotchas (took several iterations)

Two distinct failures at N=2000, both now fixed in `src/utils/text_graph_dataset.py`:

1. **Host-RAM OOM** during dataset gen (`--mem=64G` → SIGKILL). The 2400×2400 SPD
   feature needs ~80–100 GB to build → `--mem=224G` for gen (see below).
2. **Arrow 2³¹-element-per-array limit** on the flattened N×N SPD column. At N=2400
   that's ~5.8M elems/row, so the default 1000-row write batch (~5.8e9 elems) overflows
   at *every* cast and at `save_to_disk`. Fix in `cast_float_features_to_fp32()` +
   `save()`: store SPD as **`large_list`** (64-bit offsets) and cap the write batch
   (`writer_batch_size=128` in the cast; `datasets.config.DEFAULT_MAX_BATCH_SIZE=128`
   for save) so no single written array exceeds 2³¹. (500/1000 were unaffected: 1000-node
   was ~1.4e9/batch, just under the limit.)

**2000 runs are split out as their own jobs** (the 500s for those seeds already
completed under 107190–197). Generated the 2000 dataset once (20 GB train + 2 GB eval,
`n1600-2400`), then 6 single-spec 2000 jobs reuse it:

| jobid | name | spec |
|---|---|---|
| 107227–107232 | gtlm_train_0..5 | `2000/{s0,s1,s2}/{k0,k1}` |

2000 trains at **batch 2 / accum 16** (fits B200 ~comfortably, no OOM) but is heavy:
**~100 s/step × 1240 steps ≈ 34h + eval ≈ ~38h/run**. Walltime set to **`-t 80:00:00`**
(≈2× estimate). All 6 run in parallel, so wall-clock to finish ≈ one run (~38h).

## Host-RAM OOM (the crash that reset this batch)

An earlier batch ran with `--mem=64G` and every 2000-node run died with `rc=137` (SIGKILL,
no CUDA-OOM message → **host** RAM). Cause: a 2400-node graph's shortest-path-distance
feature is 2400×2400 per graph, so building the 2000-example train dataset needs ~80–100 GB
RAM — far over 64 G. That also meant the 2000 `.gtds` never persisted (gen was killed mid-build),
so each training job re-tried generation and re-OOM'd. Fixes: `--mem=224G`, and the runner now
checks the dataset-gen exit code (`ensure_dataset` returns non-zero + skips instead of
falsely logging "ready"). 500/1000 datasets are small and were unaffected.

## Run naming

`n{num_nodes}_k{k}_s{seed}_lora{r}_{suffix}`, e.g.
`n500_k0_s0_lora8_spd(32)+magnetic(dim=32,q=0.25,m=128)` — size leads, no backend/difficulty
prefix (so this batch is easy to tell apart from earlier `HARD_v2-flex_…` / `v2_n…` runs).

**Note (gotcha that bit on first submit):** the runner must use the venv interpreter
directly (`.venv/bin/python3`), NOT `source .venv/bin/activate`. Under `set -u` the
activate script trips on unbound vars; if that failure is swallowed the job silently
runs on the system python (`/opt/deepops/venv`, no `transformers`) and dies instantly.
`run_matrix_gpu.sh` already does this via `$PY`.

## Datasets

- 500 (`n400-600`) and 1000 (`n800-1200`) train/eval datasets already exist on disk with
  `magnetic_m=128` stored → reused directly.
- 2000 (`n1600-2400`) does **not** exist and is generated on first use (GPU spectral
  decomposition, ~1–2 h). `gpu6` generates it first (`PREP:2000`) so it overlaps the 500
  runs; the other 2000-jobs wait on a **flock** (`/tmp/expr_ds_n1600-2400.lock`) so they
  never race on the same `.gtds` file — one writes, the rest load. No codebase change; the
  lock lives entirely in the runner script.

## Environment (important — the B200 gotcha)

The B200 nodes (`ixb*`) run a **py3.12** host image, but the project `.venv` is **py3.10**
(its `python3` symlinks to `/usr/bin/python3`, which is 3.12 on B200 → `.venv` unusable on
the bare host; jobs die instantly on `import transformers`). The fix is to run inside the
project's container, which provides a py3.10 base so `.venv` resolves:

- container: `/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh`
- launch via pyxis: `srun --container-image=<sqsh> --container-mounts=/shared:/shared …`
  (an `srun` *step inside* an `sbatch` allocation — bare interactive `srun` is blocked on `frida`)
- inside: `export HOME=/shared/home/dario.vajda` (HF/wandb creds + cached Llama-3.2-1B),
  `source .venv/bin/activate`, `bash login.sh` (HF + wandb), then `python -m …`.

Validated end-to-end on B200 before launch (flex-attention autotunes/compiles, 2 steps,
~2.7 s/step — ~2× faster than A100). All of this is baked into `run_matrix_gpu.sh`.

## Files & mechanism

- **`run_matrix_gpu.sh`** — the single reusable runner (runs *inside* the container).
  `run_matrix_gpu.sh <label> <spec>...` where `<spec>` is `PREP:<size>` or `<size>:<seed>:<k>`.
  Exports HOME, activates `.venv`, runs `login.sh`, sets a per-GPU
  `TORCHINDUCTOR_CACHE_DIR=.inductor_cache/<label>` (within-GPU kernel reuse), flock-ensures
  datasets, then runs each training spec. (Kept; the 8 sbatch lines below are throwaway.)
- The 8 submissions (reproduce by re-running this loop):
  ```bash
  HELPER=/shared/workspace/povejmo/graph_model/src/experiments/expressiveness/run_matrix_gpu.sh
  OUT=/shared/workspace/povejmo/graph_model/src/experiments/expressiveness/job_logs
  SQSH=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh
  declare -A ASSIGN=(
    [gpu0]="500:0:0 2000:0:0"  [gpu1]="500:0:1 2000:0:1"
    [gpu2]="500:1:0 2000:1:0"  [gpu3]="500:1:1 2000:1:1"
    [gpu4]="500:2:0 2000:2:0"  [gpu5]="500:2:1 2000:2:1"
    [gpu6]="PREP:2000 1000:0:0 1000:1:0 1000:2:0"
    [gpu7]="1000:0:1 1000:1:1 1000:2:1" )
  for g in gpu0 gpu1 gpu2 gpu3 gpu4 gpu5 gpu6 gpu7; do
    sbatch -p frida -A povejmo -w ixb3 --gres=gpu:B200:1 -c 16 --mem=64G -t 24:00:00 \
      -J "exp_$g" -o "$OUT/slurm_${g}_%j.out" \
      --wrap "srun --container-image=$SQSH --container-mounts=/shared:/shared bash $HELPER $g ${ASSIGN[$g]}"
  done
  ```

## Logs & results

- Per-GPU log:  `job_logs/<label>.log`        (START/END markers, dataset gen)
- Per-run log:  `job_logs/run_n<size>_s<seed>_k<k>.log`  (full training output)
- Slurm stdout: `job_logs/slurm_<label>_<jobid>.out`
- Durable metrics: `results/train_runs.jsonl` (one JSON line per run; grows to 18 rows)
- wandb project: **GraphLLM**

## Monitoring

```bash
squeue -u $USER -o "%.8i %.10j %.8T %.10M %.20R"      # job states
tail -f src/experiments/expressiveness/job_logs/run_n500_s0_k0.log
grep -c . src/experiments/expressiveness/results/train_runs.jsonl   # progress: x/18
# escape check for any run: eval_loss should cross below ln(2)=0.693
grep "eval_em_accuracy" src/experiments/expressiveness/job_logs/run_n2000_s0_k0.log
```

## Final report

Once `results/train_runs.jsonl` reaches 18 rows, generate the comprehensive report:
accuracy vs size with per-seed mean±std, k=0 vs k=1, whether 2000 escapes the plateau,
and step-time / peak-memory / sparsity scaling. (Claude will produce this on completion.)

## Expected outcome

If the under-training thesis holds, 1000 and 2000 should also escape the marginal within
15 epochs and land well above 0.5 (500 reached ~0.75 at 12 epochs). A flat-0.5 result at
2000 despite 15 epochs would instead point to a genuine scale limit and warrant the
prompt-bridge fix (compute SPD/magnetic features on the graph *without* the prompt-node
bridge — see `project_expressiveness_plateau`).

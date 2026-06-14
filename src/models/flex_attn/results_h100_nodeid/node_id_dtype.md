# Experiment #10 — node_id dtype (int64 / int32)

device: NVIDIA H100 PCIe, torch 2.6.0+cu124, compile_mode=max-autotune-no-cudagraphs

int16 excluded: torch requires index tensors to be long/int/byte/bool, and node_ids is used as an index.

### fwd (ms, and %Δ vs int64)

| config | L | blkSp | int64 | int32 |
|---|---|---|---|---|
| k0_512x32_dense | 17593 | 0.01 | 46.32 | 43.01 (-7.1%) |
| k2_512x32_op | 17593 | 0.93 | 4.98 | 4.88 (-2.0%) |
| k2_2048x8_largeN | 17392 | 0.95 | 4.06 | 3.99 (-1.6%) |

### bwd (ms, and %Δ vs int64)

| config | L | blkSp | int64 | int32 |
|---|---|---|---|---|
| k0_512x32_dense | 17593 | 0.01 | 278.77 | 256.34 (-8.0%) |
| k2_512x32_op | 17593 | 0.93 | 22.13 | 24.27 (+9.7%) |
| k2_2048x8_largeN | 17392 | 0.95 | 12.72 | 12.07 (-5.1%) |

### fwd_bwd (ms, and %Δ vs int64)

| config | L | blkSp | int64 | int32 |
|---|---|---|---|---|
| k0_512x32_dense | 17593 | 0.01 | 321.10 | 299.75 (-6.7%) |
| k2_512x32_op | 17593 | 0.93 | 27.08 | 29.14 (+7.6%) |
| k2_2048x8_largeN | 17392 | 0.95 | 16.88 | 16.10 (-4.7%) |

# FlexAttention sweep — `isolation`

_Generated 2026-06-12 18:32. 12 configs, methods: flash, eager, flex-{0,2}._

**Legend.** Two tables over the same configs: **latency** (median forward+backward, milliseconds; `OOM` and error tags show up here, in bold) and **peak memory** (during forward+backward, GB; `—` for failed runs). `flash`/`flash_nc`/`eager` are K-independent (one column; run once per config); `flex-{K}` = flex with the K-hop-{K} mask. `tokSp-{K}` / `blkSp-{K}` = token-level / block-level sparsity of that mask (fraction of attention masked out; block-level is what flex actually skips — latency table only). `L` = packed sequence length.

## Latency — forward+backward (ms)

| nodes | tpn | order | L | flash | eager | tokSp-0 | blkSp-0 | flex-0 | tokSp-2 | blkSp-2 | flex-2 |
| --: | --: | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 128 | 2 | rcm | 402 | 0.6 | 6.2 | 0.28 | 0.19 | 0.9 | 0.86 | 0.31 | 2.9 |
| 128 | 8 | rcm | 1216 | 0.5 | 27.9 | 0.10 | 0.09 | 2.4 | 0.89 | 0.53 | 4.5 |
| 128 | 32 | rcm | 4478 | 2.3 | 341.0 | 0.03 | 0.03 | 27.5 | 0.90 | 0.67 | 12.9 |
| 128 | 128 | rcm | 17506 | 29.0 | **OOM** | 0.01 | 0.01 | 682.2 | 0.90 | 0.82 | 135.9 |
| 512 | 2 | rcm | 1206 | 0.6 | 35.5 | 0.10 | 0.09 | 4.9 | 0.97 | 0.67 | 2.1 |
| 512 | 8 | rcm | 4493 | 2.5 | 361.8 | 0.03 | 0.03 | 24.1 | 0.97 | 0.80 | 7.9 |
| 512 | 32 | rcm | 17593 | 29.1 | **OOM** | 0.01 | 0.01 | 372.1 | 0.97 | 0.90 | 54.7 |
| 512 | 128 | rcm | 69949 | 437.3 | **OOM** | 0.00 | 0.00 | 7614.4 | 0.97 | 0.95 | 469.3 |
| 2048 | 2 | rcm | 4457 | 2.4 | 475.8 | 0.03 | 0.03 | 35.1 | 0.99 | 0.87 | 9.7 |
| 2048 | 8 | rcm | 17392 | 28.4 | **OOM** | 0.01 | 0.01 | 323.3 | 0.99 | 0.92 | 44.3 |
| 2048 | 32 | rcm | 69121 | 425.9 | **OOM** | 0.00 | 0.00 | 5858.3 | 0.99 | 0.97 | 252.3 |
| 2048 | 128 | rcm | 276067 | 7097.9 | **OOM** | 0.00 | 0.00 | **TIMEOUT** | 0.99 | 0.99 | 1974.9 |

## Peak memory — forward+backward (GB)

| nodes | tpn | order | L | flash | eager | flex-0 | flex-2 |
| --: | --: | :-- | --: | --: | --: | --: | --: |
| 128 | 2 | rcm | 402 | 0.05 | 0.28 | 0.04 | 0.04 |
| 128 | 8 | rcm | 1216 | 0.08 | 2.27 | 0.06 | 0.06 |
| 128 | 32 | rcm | 4478 | 0.23 | 30.05 | 0.14 | 0.14 |
| 128 | 128 | rcm | 17506 | 0.80 | — | 0.47 | 0.47 |
| 512 | 2 | rcm | 1206 | 0.36 | 2.53 | 0.40 | 0.40 |
| 512 | 8 | rcm | 4493 | 0.51 | 30.55 | 0.48 | 0.48 |
| 512 | 32 | rcm | 17593 | 1.09 | — | 0.81 | 0.81 |
| 512 | 128 | rcm | 69949 | 3.40 | — | 2.11 | 2.11 |
| 2048 | 2 | rcm | 4457 | 4.98 | 34.80 | 5.73 | 5.74 |
| 2048 | 8 | rcm | 17392 | 5.55 | — | 6.06 | 6.06 |
| 2048 | 32 | rcm | 69121 | 7.84 | — | 7.35 | 7.35 |
| 2048 | 128 | rcm | 276067 | 16.97 | — | — | 12.55 |

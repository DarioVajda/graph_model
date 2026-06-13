# FlexAttention sweep — `full_model`

_Generated 2026-06-12 18:32. 12 configs, methods: flash, eager, flex-{0,2}._

**Legend.** Two tables over the same configs: **latency** (median forward+backward, milliseconds; `OOM` and error tags show up here, in bold) and **peak memory** (during forward+backward, GB; `—` for failed runs). `flash`/`flash_nc`/`eager` are K-independent (one column; run once per config); `flex-{K}` = flex with the K-hop-{K} mask. `tokSp-{K}` / `blkSp-{K}` = token-level / block-level sparsity of that mask (fraction of attention masked out; block-level is what flex actually skips — latency table only). `L` = packed sequence length.

## Latency — forward+backward (ms)

| nodes | tpn | order | L | flash | eager | tokSp-0 | blkSp-0 | flex-0 | tokSp-2 | blkSp-2 | flex-2 |
| --: | --: | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 128 | 2 | rcm | 402 | 61.1 | 152.7 | 0.28 | 0.19 | 106.8 | 0.86 | 0.31 | 122.3 |
| 128 | 8 | rcm | 1216 | 74.2 | 518.8 | 0.10 | 0.09 | 148.1 | 0.89 | 0.53 | 114.9 |
| 128 | 32 | rcm | 4478 | 265.5 | 5691.7 | 0.03 | 0.03 | 670.4 | 0.90 | 0.67 | 442.3 |
| 128 | 128 | rcm | 17506 | 1320.1 | **OOM** | 0.01 | 0.01 | 11483.9 | 0.90 | 0.82 | 3072.1 |
| 512 | 2 | rcm | 1206 | 79.9 | 677.4 | 0.10 | 0.09 | 153.9 | 0.97 | 0.67 | 152.6 |
| 512 | 8 | rcm | 4493 | 267.3 | 6058.2 | 0.03 | 0.03 | 641.2 | 0.97 | 0.80 | 385.1 |
| 512 | 32 | rcm | 17593 | 1330.2 | **OOM** | 0.01 | 0.01 | 6965.8 | 0.97 | 0.90 | 1759.8 |
| 512 | 128 | rcm | 69949 | **OOM** | **OOM** | 0.00 | 0.00 | **OOM** | 0.97 | 0.95 | **OOM** |
| 2048 | 2 | rcm | 4457 | 262.9 | **OOM** | 0.03 | 0.03 | 1346.4 | 0.99 | 0.87 | 919.3 |
| 2048 | 8 | rcm | 17392 | 1301.8 | **OOM** | 0.01 | 0.01 | **OOM** | 0.99 | 0.92 | **OOM** |
| 2048 | 32 | rcm | 69121 | **OOM** | **OOM** | 0.00 | 0.00 | **OOM** | 0.99 | 0.97 | **OOM** |
| 2048 | 128 | rcm | 276067 | **OOM** | **OOM** | 0.00 | 0.00 | **OOM** | 0.99 | 0.99 | **OOM** |

## Peak memory — forward+backward (GB)

| nodes | tpn | order | L | flash | eager | flex-0 | flex-2 |
| --: | --: | :-- | --: | --: | --: | --: | --: |
| 128 | 2 | rcm | 402 | 5.61 | 5.63 | 5.63 | 5.63 |
| 128 | 8 | rcm | 1216 | 6.34 | 8.61 | 6.43 | 6.43 |
| 128 | 32 | rcm | 4478 | 16.79 | 58.30 | 16.32 | 16.31 |
| 128 | 128 | rcm | 17506 | 58.73 | — | 56.37 | 56.37 |
| 512 | 2 | rcm | 1206 | 6.21 | 10.93 | 8.85 | 8.84 |
| 512 | 8 | rcm | 4493 | 16.88 | 61.06 | 19.04 | 19.04 |
| 512 | 32 | rcm | 17593 | 59.02 | — | 59.14 | 59.14 |
| 512 | 128 | rcm | 69949 | — | — | — | — |
| 2048 | 2 | rcm | 4457 | 16.78 | — | 56.86 | 56.87 |
| 2048 | 8 | rcm | 17392 | 58.40 | — | — | — |
| 2048 | 32 | rcm | 69121 | — | — | — | — |
| 2048 | 128 | rcm | 276067 | — | — | — | — |

# FlexAttention sweep — `isolation`

_Generated 2026-06-12 18:32. 12 configs, methods: flash, flash_nc, eager, flex-{0,2}._

**Legend.** Two tables over the same configs: **latency** (median forward+backward, milliseconds; `OOM` and error tags show up here, in bold) and **peak memory** (during forward+backward, GB; `—` for failed runs). `flash`/`flash_nc`/`eager` are K-independent (one column; run once per config); `flex-{K}` = flex with the K-hop-{K} mask. `tokSp-{K}` / `blkSp-{K}` = token-level / block-level sparsity of that mask (fraction of attention masked out; block-level is what flex actually skips — latency table only). `L` = packed sequence length.

## Latency — forward+backward (ms)

| nodes | tpn | order | L | flash | flash_nc | eager | tokSp-0 | blkSp-0 | flex-0 | tokSp-2 | blkSp-2 | flex-2 |
| --: | --: | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 128 | 2 | rcm | 402 | 0.5 | 0.5 | 4.7 | 0.28 | 0.19 | 1.5 | 0.89 | 0.25 | 1.6 |
| 128 | 8 | rcm | 1216 | 0.5 | 0.5 | 21.5 | 0.10 | 0.09 | 3.0 | 0.91 | 0.55 | 2.1 |
| 128 | 32 | rcm | 4478 | 2.0 | 3.0 | 263.8 | 0.03 | 0.03 | 26.9 | 0.90 | 0.69 | 15.3 |
| 128 | 128 | rcm | 17506 | 22.4 | 43.3 | **OOM** | 0.01 | 0.01 | 411.6 | 0.89 | 0.82 | 110.0 |
| 512 | 2 | rcm | 1206 | 0.5 | 0.5 | 23.0 | 0.10 | 0.09 | 4.2 | 0.97 | 0.67 | 2.5 |
| 512 | 8 | rcm | 4493 | 2.0 | 3.4 | 268.3 | 0.03 | 0.03 | 24.8 | 0.97 | 0.81 | 8.3 |
| 512 | 32 | rcm | 17593 | 22.8 | 43.3 | **OOM** | 0.01 | 0.01 | 383.5 | 0.97 | 0.90 | 60.4 |
| 512 | 128 | rcm | 69949 | 337.9 | 672.8 | **OOM** | 0.00 | 0.00 | 6437.0 | 0.97 | 0.95 | 441.3 |
| 2048 | 2 | rcm | 4457 | 1.9 | 3.1 | 302.8 | 0.03 | 0.03 | 46.1 | 0.99 | 0.86 | 12.6 |
| 2048 | 8 | rcm | 17392 | 22.0 | 42.6 | **OOM** | 0.01 | 0.01 | 310.9 | 0.99 | 0.92 | 54.7 |
| 2048 | 32 | rcm | 69121 | 330.6 | 654.2 | **OOM** | 0.00 | 0.00 | 5511.5 | 0.99 | 0.97 | 282.7 |
| 2048 | 128 | rcm | 276067 | 5404.4 | 10627.3 | **OOM** | 0.00 | 0.00 | **TIMEOUT** | 0.99 | 0.99 | 1834.0 |

## Peak memory — forward+backward (GB)

| nodes | tpn | order | L | flash | flash_nc | eager | flex-0 | flex-2 |
| --: | --: | :-- | --: | --: | --: | --: | --: | --: |
| 128 | 2 | rcm | 402 | 0.07 | 0.07 | 0.30 | 0.07 | 0.07 |
| 128 | 8 | rcm | 1216 | 0.11 | 0.11 | 2.29 | 0.09 | 0.09 |
| 128 | 32 | rcm | 4478 | 0.25 | 0.25 | 30.06 | 0.17 | 0.17 |
| 128 | 128 | rcm | 17506 | 0.83 | 0.83 | — | 0.50 | 0.50 |
| 512 | 2 | rcm | 1206 | 0.39 | 0.39 | 2.55 | 0.42 | 0.42 |
| 512 | 8 | rcm | 4493 | 0.54 | 0.54 | 30.56 | 0.50 | 0.50 |
| 512 | 32 | rcm | 17593 | 1.11 | 1.11 | — | 0.83 | 0.83 |
| 512 | 128 | rcm | 69949 | 3.43 | 3.43 | — | 2.13 | 2.13 |
| 2048 | 2 | rcm | 4457 | 5.00 | 5.00 | 34.81 | 5.75 | 5.76 |
| 2048 | 8 | rcm | 17392 | 5.57 | 5.57 | — | 6.08 | 6.08 |
| 2048 | 32 | rcm | 69121 | 7.86 | 7.86 | — | 7.37 | 7.37 |
| 2048 | 128 | rcm | 276067 | 16.99 | 16.99 | — | — | 12.57 |

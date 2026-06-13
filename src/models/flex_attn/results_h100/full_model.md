# FlexAttention sweep — `full_model`

_Generated 2026-06-12 18:32. 12 configs, methods: flash, eager, flex-{0,2}._

**Legend.** Two tables over the same configs: **latency** (median forward+backward, milliseconds; `OOM` and error tags show up here, in bold) and **peak memory** (during forward+backward, GB; `—` for failed runs). `flash`/`flash_nc`/`eager` are K-independent (one column; run once per config); `flex-{K}` = flex with the K-hop-{K} mask. `tokSp-{K}` / `blkSp-{K}` = token-level / block-level sparsity of that mask (fraction of attention masked out; block-level is what flex actually skips — latency table only). `L` = packed sequence length.

## Latency — forward+backward (ms)

| nodes | tpn | order | L | flash | eager | tokSp-0 | blkSp-0 | flex-0 | tokSp-2 | blkSp-2 | flex-2 |
| --: | --: | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 128 | 2 | rcm | 402 | 32.0 | 113.9 | 0.28 | 0.19 | 77.0 | 0.89 | 0.25 | 77.6 |
| 128 | 8 | rcm | 1216 | 48.2 | 396.5 | 0.10 | 0.09 | 104.2 | 0.91 | 0.55 | 93.0 |
| 128 | 32 | rcm | 4478 | 170.8 | 4450.1 | 0.03 | 0.03 | 569.5 | 0.90 | 0.69 | 347.3 |
| 128 | 128 | rcm | 17506 | 886.5 | **OOM** | 0.01 | 0.01 | 7245.3 | 0.89 | 0.82 | 2260.8 |
| 512 | 2 | rcm | 1206 | 49.4 | 442.4 | 0.10 | 0.09 | 164.2 | 0.97 | 0.67 | 120.5 |
| 512 | 8 | rcm | 4493 | 173.4 | 4549.6 | 0.03 | 0.03 | 560.6 | 0.97 | 0.81 | 326.3 |
| 512 | 32 | rcm | 17593 | 895.2 | **OOM** | 0.01 | 0.01 | 6665.6 | 0.97 | 0.90 | 1563.7 |
| 512 | 128 | rcm | 69949 | **OOM** | **OOM** | 0.00 | 0.00 | **OOM** | 0.97 | 0.95 | **OOM** |
| 2048 | 2 | rcm | 4457 | 171.3 | **OOM** | 0.03 | 0.03 | 1796.4 | 0.99 | 0.86 | 1213.5 |
| 2048 | 8 | rcm | 17392 | 875.9 | **OOM** | 0.01 | 0.01 | **OOM** | 0.99 | 0.92 | **OOM** |
| 2048 | 32 | rcm | 69121 | **OOM** | **OOM** | 0.00 | 0.00 | **OOM** | 0.99 | 0.97 | **OOM** |
| 2048 | 128 | rcm | 276067 | **OOM** | **OOM** | 0.00 | 0.00 | **OOM** | 0.99 | 0.99 | **OOM** |

## Peak memory — forward+backward (GB)

| nodes | tpn | order | L | flash | eager | flex-0 | flex-2 |
| --: | --: | :-- | --: | --: | --: | --: | --: |
| 128 | 2 | rcm | 402 | 5.66 | 5.68 | 5.67 | 5.67 |
| 128 | 8 | rcm | 1216 | 6.39 | 8.66 | 6.47 | 6.47 |
| 128 | 32 | rcm | 4478 | 16.83 | 58.34 | 16.36 | 16.36 |
| 128 | 128 | rcm | 17506 | 58.78 | — | 56.42 | 56.42 |
| 512 | 2 | rcm | 1206 | 6.25 | 10.98 | 8.89 | 8.89 |
| 512 | 8 | rcm | 4493 | 16.93 | 61.09 | 19.08 | 19.08 |
| 512 | 32 | rcm | 17593 | 59.07 | — | 59.19 | 59.19 |
| 512 | 128 | rcm | 69949 | — | — | — | — |
| 2048 | 2 | rcm | 4457 | 16.83 | — | 56.91 | 56.91 |
| 2048 | 8 | rcm | 17392 | 58.45 | — | — | — |
| 2048 | 32 | rcm | 69121 | — | — | — | — |
| 2048 | 128 | rcm | 276067 | — | — | — | — |

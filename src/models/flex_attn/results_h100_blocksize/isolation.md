# FlexAttention sweep — `isolation`

_Generated 2026-06-12 21:12. 6 configs, methods: flash, flash_nc, eager, flex-{0,2}, flex@64-{0,2}._

**Legend.** Two tables over the same configs: **latency** (median forward+backward, milliseconds; `OOM` and error tags show up here, in bold) and **peak memory** (during forward+backward, GB; `—` for failed runs). `flash`/`flash_nc`/`eager` are K-independent (one column; run once per config); `flex-{K}` = flex with the K-hop-{K} mask. `tokSp-{K}` / `blkSp-{K}` = token-level / block-level sparsity of that mask (fraction of attention masked out; block-level is what flex actually skips — latency table only). `L` = packed sequence length.

## Latency — forward+backward (ms)

| nodes | tpn | order | L | flash | flash_nc | eager | tokSp-0 | blkSp-0 | flex-0 | flex@64-0 | tokSp-2 | blkSp-2 | flex-2 | flex@64-2 |
| --: | --: | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 512 | 2 | rcm | 1206 | 0.5 | 0.5 | 23.0 | 0.10 | 0.09 | 3.5 | 3.5 | 0.97 | 0.67 | 1.6 | 1.6 |
| 512 | 8 | rcm | 4493 | 2.0 | 3.4 | 268.3 | 0.03 | 0.03 | 23.7 | 20.7 | 0.97 | 0.81 | 7.1 | 4.2 |
| 512 | 32 | rcm | 17593 | 22.8 | 43.3 | **OOM** | 0.01 | 0.01 | 320.5 | 343.6 | 0.97 | 0.90 | 48.2 | 31.0 |
| 2048 | 2 | rcm | 4457 | 1.9 | 3.1 | 302.8 | 0.03 | 0.03 | 42.3 | 41.7 | 0.99 | 0.86 | 8.8 | 5.7 |
| 2048 | 8 | rcm | 17392 | 22.0 | 42.6 | **OOM** | 0.01 | 0.01 | 288.1 | 291.6 | 0.99 | 0.92 | 32.3 | 20.5 |
| 2048 | 32 | rcm | 69121 | 330.6 | 654.2 | **OOM** | 0.00 | 0.00 | 5082.4 | 5090.9 | 0.99 | 0.97 | 170.3 | 115.5 |

## Peak memory — forward+backward (GB)

| nodes | tpn | order | L | flash | flash_nc | eager | flex-0 | flex@64-0 | flex-2 | flex@64-2 |
| --: | --: | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 512 | 2 | rcm | 1206 | 0.39 | 0.39 | 2.55 | 0.42 | 0.42 | 0.42 | 0.42 |
| 512 | 8 | rcm | 4493 | 0.54 | 0.54 | 30.56 | 0.50 | 0.50 | 0.50 | 0.50 |
| 512 | 32 | rcm | 17593 | 1.11 | 1.11 | — | 0.83 | 0.82 | 0.83 | 0.82 |
| 2048 | 2 | rcm | 4457 | 5.00 | 5.00 | 34.81 | 5.75 | 5.75 | 5.76 | 5.76 |
| 2048 | 8 | rcm | 17392 | 5.57 | 5.57 | — | 6.08 | 6.08 | 6.08 | 6.08 |
| 2048 | 32 | rcm | 69121 | 7.86 | 7.86 | — | 7.37 | 7.38 | 7.37 | 7.38 |

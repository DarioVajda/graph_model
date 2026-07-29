# graphqa — real inputs, eager vs flex vs plain-LLM sdpa

GPU: NVIDIA H100 80GB HBM3

### graphqa / standard/node_count — 24 real batches (B=4, train split)

L: mean 512, range 512–512, 1 distinct (L,N) shapes (ladder step 512); N max 20; padding 94%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 925.1 ± 8.6 | 1.00× | 17.04 | 23 | 0.06 h |
| `flex` | 355.0 ± 27.7 | 2.61× | 14.92 | 200 | 0.02 h |
| `sdpa` | 271.3 ± 0.9 | 3.41× | 15.29 | 7 | 0.02 h |

### graphqa / standard/shortest_path — 24 real batches (B=4, train split)

L: mean 128, range 128–128, 1 distinct (L,N) shapes (ladder step 128); N max 20; padding 70%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 129.4 ± 4.4 | 1.00× | 7.42 | 3 | 0.01 h |
| `flex` | 113.4 ± 2.5 | 1.14× | 7.27 | 186 | 0.01 h |
| `flex-nobias` | 93.4 ± 1.6 | 1.39× | 7.27 | 228 | 0.01 h |
| `sdpa` | 79.4 ± 1.8 | 1.63× | 7.36 | 2 | 0.01 h |

### graphqa / standard/node_count — 24 real batches (B=4, train split)

L: mean 128, range 128–128, 1 distinct (L,N) shapes (ladder step 128); N max 20; padding 77%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 128.5 ± 4.0 | 1.00× | 7.42 | 3 | 0.01 h |
| `flex` | 114.0 ± 3.9 | 1.13× | 7.27 | 187 | 0.01 h |
| `flex-nobias` | 93.3 ± 1.6 | 1.38× | 7.27 | 228 | 0.01 h |
| `sdpa` | 79.0 ± 1.8 | 1.63× | 7.36 | 2 | 0.01 h |

### graphqa / standard/shortest_path — 24 real batches (B=4, train split)

L: mean 44, range 33–53, 17 distinct (L,N) shapes (ladder step 512); N max 20; padding 12%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 101.8 ± 1.4 | 1.00× | 5.81 | 3 | 0.01 h |
| `sdpa` | 44.0 ± 1.9 | 2.31× | 5.81 | 1 | 0.00 h |

### graphqa / standard/node_count — 24 real batches (B=4, train split)

L: mean 33, range 25–36, 9 distinct (L,N) shapes (ladder step 512); N max 20; padding 12%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 101.5 ± 2.3 | 1.00× | 5.47 | 3 | 0.01 h |
| `sdpa` | 43.1 ± 1.9 | 2.35× | 5.48 | 1 | 0.00 h |

### graphqa / incidence/shortest_path — 24 real batches (B=4, train split)

L: mean 325, range 128–512, 7 distinct (L,N) shapes (ladder step 128); N max 172; padding 55%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 523.8 ± 337.3 | 1.00× | 17.21 | 13 | 0.04 h |
| `flex` | 319.1 ± 166.5 | 1.64× | 15.59 | 1696 | 0.02 h |
| `flex-nobias` | 212.2 ± 75.7 | 2.47× | 15.09 | 937 | 0.01 h |
| `sdpa` | 182.6 ± 64.5 | 2.87× | 15.46 | 4 | 0.01 h |

### graphqa / incidence/node_count — 24 real batches (B=4, train split)

L: mean 325, range 128–512, 7 distinct (L,N) shapes (ladder step 128); N max 172; padding 58%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 522.2 ± 337.0 | 1.00× | 17.21 | 13 | 0.04 h |
| `flex` | 319.4 ± 165.7 | 1.63× | 15.59 | 1700 | 0.02 h |
| `flex-nobias` | 213.8 ± 75.7 | 2.44× | 15.09 | 941 | 0.01 h |
| `sdpa` | 183.0 ± 64.8 | 2.85× | 15.46 | 4 | 0.01 h |

### graphqa / incidence/shortest_path — 24 real batches (B=4, train split)

L: mean 263, range 72–502, 24 distinct (L,N) shapes (ladder step 512); N max 172; padding 44%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 372.8 ± 248.5 | 1.00× | 16.85 | 9 | 0.03 h |
| `sdpa` | 158.1 ± 66.7 | 2.36× | 15.24 | 4 | 0.01 h |

### graphqa / incidence/node_count — 24 real batches (B=4, train split)

L: mean 255, range 64–494, 24 distinct (L,N) shapes (ladder step 512); N max 172; padding 46%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 362.0 ± 243.9 | 1.00× | 16.63 | 9 | 0.03 h |
| `sdpa` | 155.2 ± 68.1 | 2.33× | 15.06 | 4 | 0.01 h |

### graphqa / standard/node_count — 24 real batches (B=4, train split)

L: mean 33, range 25–36, 9 distinct (L,N) shapes (ladder step 512); N max 20; padding 12%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 40.9 ± 2.2 | — | 5.48 | 1 | 0.00 h |
| `sdpa` | 51.3 ± 58.3 | — | 5.48 | 1 | 0.00 h |

### graphqa / incidence/node_count — 24 real batches (B=4, train split)

L: mean 255, range 64–494, 24 distinct (L,N) shapes (ladder step 512); N max 172; padding 46%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 156.2 ± 67.9 | — | 15.09 | 4 | 0.01 h |
| `sdpa` | 155.7 ± 67.7 | — | 15.06 | 4 | 0.01 h |

### graphqa / standard/node_count — 24 real batches (B=4, train split)

L: mean 33, range 25–36, 9 distinct (L,N) shapes (ladder step 512); N max 20; padding 12%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 40.9 ± 2.0 | — | 5.48 | 1 | 0.00 h |
| `sdpa` | 42.2 ± 2.1 | — | 5.48 | 1 | 0.00 h |

### graphqa / incidence/node_count — 24 real batches (B=4, train split)

L: mean 255, range 64–494, 24 distinct (L,N) shapes (ladder step 512); N max 172; padding 46%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 156.0 ± 68.1 | — | 15.09 | 4 | 0.01 h |
| `sdpa` | 155.2 ± 67.9 | — | 15.06 | 4 | 0.01 h |

### graphqa / standard/node_count — 24 real batches (B=4, train split)

L: mean 33, range 25–36, 9 distinct (L,N) shapes (ladder step 512); N max 20; padding 12%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 40.4 ± 0.5 | — | 5.47 | 1 | 0.00 h |
| `sdpa` | 41.5 ± 0.6 | — | 5.48 | 1 | 0.00 h |

### graphqa / incidence/node_count — 24 real batches (B=4, train split)

L: mean 255, range 64–494, 24 distinct (L,N) shapes (ladder step 512); N max 172; padding 46%; dtype=fp32; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 155.6 ± 68.2 | — | 15.09 | 4 | 0.01 h |
| `sdpa` | 155.0 ± 67.9 | — | 15.06 | 4 | 0.01 h |

### graphqa / standard/node_count — 24 real batches (B=4, train split)

L: mean 128, range 128–128, 1 distinct (L,N) shapes (ladder step 128); N max 20; padding 77%; dtype=bf16; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 114.5 ± 1.6 | 1.00× | 4.71 | 3 | 0.01 h |
| `sdpa` | 48.9 ± 0.8 | 2.34× | 4.56 | 1 | 0.00 h |

### graphqa / incidence/node_count — 24 real batches (B=4, train split)

L: mean 325, range 128–512, 7 distinct (L,N) shapes (ladder step 128); N max 172; padding 58%; dtype=bf16; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 372.1 ± 240.6 | 1.00× | 14.04 | 10 | 0.03 h |
| `sdpa` | 60.2 ± 11.7 | 6.18× | 11.16 | 2 | 0.00 h |

### graphqa / standard/node_count — 24 real batches (B=1, train split)

L: mean 128, range 128–128, 1 distinct (L,N) shapes (ladder step 128); N max 19; padding 78%; dtype=bf16; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 112.3 ± 2.1 | 1.00× | 2.99 | 3 | 0.03 h |
| `flex` | 129.1 ± 2.8 | 0.87× | 2.94 | 42 | 0.04 h |
| `flex-nobias` | 69.5 ± 1.4 | 1.62× | 2.94 | 31 | 0.02 h |
| `sdpa-graphmask` | 50.9 ± 1.1 | 2.21× | 2.95 | 1 | 0.01 h |
| `sdpa` | 52.5 ± 0.6 | 2.14× | 2.95 | 1 | 0.01 h |

### graphqa / standard/shortest_path — 24 real batches (B=1, train split)

L: mean 128, range 128–128, 1 distinct (L,N) shapes (ladder step 128); N max 19; padding 71%; dtype=bf16; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 113.0 ± 2.2 | 1.00× | 2.99 | 3 | 0.03 h |
| `flex` | 124.0 ± 2.0 | 0.91× | 2.94 | 10 | 0.03 h |
| `flex-nobias` | 66.2 ± 0.7 | 1.71× | 2.94 | 2 | 0.02 h |
| `sdpa-graphmask` | 51.1 ± 1.8 | 2.21× | 2.95 | 2 | 0.01 h |
| `sdpa` | 50.7 ± 1.0 | 2.23× | 2.95 | 1 | 0.01 h |

### graphqa / incidence/node_count — 24 real batches (B=1, train split)

L: mean 208, range 128–384, 5 distinct (L,N) shapes (ladder step 128); N max 113; padding 34%; dtype=bf16; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 115.1 ± 2.8 | 1.00× | 4.46 | 3 | 0.03 h |
| `flex` | 135.7 ± 3.0 | 0.85× | 4.02 | 172 | 0.04 h |
| `flex-nobias` | 73.6 ± 2.8 | 1.56× | 4.01 | 63 | 0.02 h |
| `sdpa-graphmask` | 53.1 ± 1.9 | 2.17× | 4.04 | 2 | 0.01 h |
| `sdpa` | 54.5 ± 1.7 | 2.11× | 4.04 | 1 | 0.02 h |

### graphqa / incidence/shortest_path — 24 real batches (B=1, train split)

L: mean 208, range 128–384, 5 distinct (L,N) shapes (ladder step 128); N max 113; padding 29%; dtype=bf16; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 112.9 ± 2.8 | 1.00× | 4.46 | 3 | 0.03 h |
| `flex` | 121.8 ± 22.3 | 0.93× | 4.02 | 12 | 0.03 h |
| `flex-nobias` | 67.1 ± 1.5 | 1.68× | 4.01 | 3 | 0.02 h |
| `sdpa-graphmask` | 49.1 ± 1.3 | 2.30× | 4.04 | 2 | 0.01 h |
| `sdpa` | 49.3 ± 1.5 | 2.29× | 4.04 | 1 | 0.01 h |

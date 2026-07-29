# tag — real inputs, eager vs flex vs plain-LLM sdpa

GPU: NVIDIA H100 80GB HBM3

### tag / cora — 24 real batches (B=1, train split)

L: mean 917, range 512–1536, 19 distinct (L,N) shapes (ladder step 512); N max 60; padding 27%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 824.2 ± 465.3 | 1.00× | 6.33 | 21 | 0.37 h |
| `flex` | 202.0 ± 36.5 | 4.08× | 4.85 | 176 | 0.09 h |
| `sdpa` | 103.0 ± 26.8 | 8.00× | 4.85 | 4 | 0.05 h |

### tag / cora — 24 real batches (B=1, train split)

L: mean 896, range 512–1536, 19 distinct (L,N) shapes (ladder step 512); N max 60; padding 27%; dtype=bf16; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 764.7 ± 485.2 | 1.00× | 16.37 | 19 | 0.34 h |
| `flex` | 127.8 ± 9.6 | 5.98× | 8.86 | 12 | 0.06 h |
| `sdpa` | 54.4 ± 7.4 | 14.06× | 9.00 | 2 | 0.02 h |

### tag / reddit — 24 real batches (B=1, train split)

L: mean 1045, range 1024–1536, 3 distinct (L,N) shapes (ladder step 512); N max 30; padding 7%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 366.6 ± 254.6 | 1.00× | 6.17 | 9 | 0.34 h |
| `flex` | 211.8 ± 44.7 | 1.73× | 4.77 | 90 | 0.20 h |
| `flex-nobias` | 133.1 ± 39.9 | 2.76× | 4.77 | 63 | 0.12 h |
| `sdpa` | 105.1 ± 26.3 | 3.49× | 4.77 | 3 | 0.10 h |

### tag / reddit — 24 real batches (B=1, train split)

L: mean 1024, range 1024–1024, 1 distinct (L,N) shapes (ladder step 512); N max 30; padding 5%; dtype=bf16; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 251.4 ± 10.9 | 1.00× | 9.69 | 6 | 0.23 h |
| `flex` | 117.6 ± 3.1 | 2.14× | 6.66 | 10 | 0.11 h |
| `flex-nobias` | 67.3 ± 2.0 | 3.74× | 6.65 | 2 | 0.06 h |
| `sdpa` | 50.8 ± 4.3 | 4.95× | 6.75 | 1 | 0.05 h |

### tag / cora — 24 real batches (B=1, train split)

L: mean 896, range 512–1536, 19 distinct (L,N) shapes (ladder step 512); N max 60; padding 27%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 818.3 ± 496.8 | 1.00× | 6.33 | 21 | 0.37 h |
| `flex` | 205.7 ± 38.3 | 3.98× | 4.85 | 174 | 0.09 h |
| `flex-nobias` | 130.7 ± 30.7 | 6.26× | 4.85 | 95 | 0.06 h |
| `sdpa` | 105.9 ± 31.6 | 7.73× | 4.85 | 3 | 0.05 h |

### tag / pubmed — 24 real batches (B=1, train split)

L: mean 1216, range 512–1536, 11 distinct (L,N) shapes (ladder step 512); N max 30; padding 22%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 1310.6 ± 569.5 | 1.00× | 6.16 | 31 | 4.31 h |
| `flex` | 207.3 ± 45.6 | 6.32× | 4.77 | 131 | 0.68 h |
| `flex-nobias` | 129.2 ± 28.4 | 10.14× | 4.77 | 92 | 0.42 h |
| `sdpa` | 105.3 ± 30.0 | 12.45× | 4.77 | 3 | 0.35 h |

### tag / ogbn-arxiv — 24 real batches (B=1, train split)

L: mean 1387, range 1024–1536, 9 distinct (L,N) shapes (ladder step 512); N max 60; padding 23%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 2008.9 ± 340.8 | 1.00× | 6.33 | 55 | 50.75 h |
| `flex` | 216.1 ± 47.9 | 9.30× | 4.86 | 134 | 5.46 h |
| `flex-nobias` | 133.5 ± 41.0 | 15.05× | 4.86 | 63 | 3.37 h |
| `sdpa` | 104.5 ± 26.9 | 19.22× | 4.86 | 3 | 2.64 h |

### tag / pubmed — 24 real batches (B=1, train split)

L: mean 1195, range 512–1536, 3 distinct (L,N) shapes (ladder step 512); N max 30; padding 20%; dtype=bf16; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 1150.1 ± 531.6 | 1.00× | 16.26 | 28 | 3.78 h |
| `flex` | 130.4 ± 16.5 | 8.82× | 8.75 | 11 | 0.43 h |
| `flex-nobias` | 71.1 ± 8.1 | 16.17× | 8.75 | 3 | 0.23 h |
| `sdpa` | 56.6 ± 8.7 | 20.32× | 8.90 | 2 | 0.19 h |

### tag / cora — 24 real batches (B=1, train split)

L: mean 896, range 512–1536, 4 distinct (L,N) shapes (ladder step 512); N max 60; padding 27%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 102.2 ± 26.2 | — | 4.89 | 3 | 0.05 h |
| `sdpa` | 114.7 ± 86.6 | — | 4.85 | 3 | 0.05 h |

### tag / pubmed — 24 real batches (B=1, train split)

L: mean 1195, range 512–1536, 3 distinct (L,N) shapes (ladder step 512); N max 30; padding 20%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 106.2 ± 26.8 | — | 4.84 | 2 | 0.35 h |
| `sdpa` | 121.1 ± 210.5 | — | 4.77 | 3 | 0.40 h |

### tag / ogbn-arxiv — 24 real batches (B=1, train split)

L: mean 1387, range 1024–1536, 3 distinct (L,N) shapes (ladder step 512); N max 60; padding 23%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 105.4 ± 25.5 | — | 4.94 | 2 | 2.66 h |
| `sdpa` | 198.1 ± 770.6 | — | 4.85 | 3 | 5.00 h |

### tag / reddit — 24 real batches (B=1, train split)

L: mean 1045, range 1024–1536, 2 distinct (L,N) shapes (ladder step 512); N max 30; padding 6%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 105.5 ± 31.0 | — | 4.82 | 2 | 0.10 h |
| `sdpa` | 122.7 ± 205.7 | — | 4.77 | 4 | 0.11 h |

### tag / cora — 24 real batches (B=1, train split)

L: mean 917, range 512–1536, 4 distinct (L,N) shapes (ladder step 512); N max 60; padding 28%; dtype=bf16; k_hop=0; grad-ckpt=False

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 53.9 ± 9.4 | — | 9.04 | 1 | 0.02 h |
| `sdpa` | 55.3 ± 8.9 | — | 9.00 | 2 | 0.02 h |

### tag / cora — 24 real batches (B=1, train split)

L: mean 896, range 512–1536, 4 distinct (L,N) shapes (ladder step 512); N max 60; padding 27%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 106.3 ± 29.1 | — | 4.89 | 2 | 0.05 h |
| `sdpa` | 98.5 ± 16.0 | — | 4.85 | 3 | 0.04 h |

### tag / pubmed — 24 real batches (B=1, train split)

L: mean 1216, range 512–1536, 3 distinct (L,N) shapes (ladder step 512); N max 30; padding 22%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 101.6 ± 25.6 | — | 4.84 | 2 | 0.33 h |
| `sdpa` | 122.6 ± 210.3 | — | 4.77 | 4 | 0.40 h |

### tag / ogbn-arxiv — 24 real batches (B=1, train split)

L: mean 1387, range 1024–1536, 3 distinct (L,N) shapes (ladder step 512); N max 60; padding 23%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 105.6 ± 25.7 | — | 4.94 | 2 | 2.67 h |
| `sdpa` | 194.4 ± 784.3 | — | 4.85 | 3 | 4.91 h |

### tag / reddit — 24 real batches (B=1, train split)

L: mean 1045, range 1024–1536, 2 distinct (L,N) shapes (ladder step 512); N max 30; padding 7%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 100.0 ± 24.3 | — | 4.82 | 2 | 0.09 h |
| `sdpa` | 124.1 ± 208.4 | — | 4.77 | 3 | 0.12 h |

### tag / cora — 24 real batches (B=1, train split)

L: mean 896, range 512–1536, 4 distinct (L,N) shapes (ladder step 512); N max 60; padding 27%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 816.7 ± 501.0 | 1.00× | 6.33 | 21 | 0.37 h |
| `flex` | 217.2 ± 42.3 | 3.76× | 4.85 | 181 | 0.10 h |
| `flex-nobias` | **InductorError** | — | — | — | — |
| `sdpa-graphmask` | **RuntimeError** | — | — | — | — |
| `sdpa` | **RuntimeError** | — | — | — | — |

### tag / cora — 24 real batches (B=1, train split)

L: mean 896, range 512–1536, 4 distinct (L,N) shapes (ladder step 512); N max 60; padding 27%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 109.8 ± 31.6 | — | 4.89 | 2 | 0.05 h |
| `sdpa` | 106.3 ± 85.1 | — | 4.85 | 3 | 0.05 h |

### tag / pubmed — 24 real batches (B=1, train split)

L: mean 1216, range 512–1536, 3 distinct (L,N) shapes (ladder step 512); N max 30; padding 21%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 99.1 ± 24.8 | — | 4.84 | 2 | 0.33 h |
| `sdpa` | 124.5 ± 211.1 | — | 4.77 | 4 | 0.41 h |

### tag / ogbn-arxiv — 24 real batches (B=1, train split)

L: mean 1387, range 1024–1536, 3 distinct (L,N) shapes (ladder step 512); N max 60; padding 23%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 107.1 ± 27.9 | — | 4.94 | 2 | 2.70 h |
| `sdpa` | 193.8 ± 783.9 | — | 4.85 | 3 | 4.89 h |

### tag / reddit — 24 real batches (B=1, train split)

L: mean 1067, range 1024–1536, 2 distinct (L,N) shapes (ladder step 512); N max 30; padding 9%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `sdpa-graphmask` | 102.2 ± 22.9 | — | 4.82 | 2 | 0.09 h |
| `sdpa` | 128.3 ± 219.0 | — | 4.77 | 3 | 0.12 h |

### tag / cora — 24 real batches (B=1, train split)

L: mean 917, range 512–1536, 4 distinct (L,N) shapes (ladder step 512); N max 60; padding 28%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 836.9 ± 469.0 | 1.00× | 6.32 | 21 | 0.38 h |
| `flex` | 226.8 ± 56.9 | 3.69× | 4.85 | 181 | 0.10 h |
| `flex-nobias` | 143.0 ± 36.2 | 5.85× | 4.85 | 96 | 0.06 h |
| `sdpa-graphmask` | 107.0 ± 31.6 | 7.82× | 4.90 | 3 | 0.05 h |
| `sdpa` | 118.6 ± 37.5 | 7.06× | 4.85 | 3 | 0.05 h |

### tag / pubmed — 24 real batches (B=1, train split)

L: mean 1195, range 512–1536, 3 distinct (L,N) shapes (ladder step 512); N max 30; padding 21%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 1255.3 ± 557.8 | 1.00× | 6.17 | 30 | 4.12 h |
| `flex` | 215.9 ± 48.9 | 5.81× | 4.77 | 137 | 0.71 h |
| `flex-nobias` | 137.2 ± 33.5 | 9.15× | 4.77 | 95 | 0.45 h |
| `sdpa-graphmask` | 113.3 ± 35.5 | 11.08× | 4.84 | 3 | 0.37 h |
| `sdpa` | 112.1 ± 33.7 | 11.20× | 4.77 | 3 | 0.37 h |

### tag / ogbn-arxiv — 24 real batches (B=1, train split)

L: mean 1387, range 1024–1536, 3 distinct (L,N) shapes (ladder step 512); N max 60; padding 23%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 2006.2 ± 353.7 | 1.00× | 6.33 | 55 | 50.68 h |
| `flex` | 208.3 ± 40.1 | 9.63× | 4.86 | 12 | 5.26 h |
| `flex-nobias` | 126.8 ± 29.9 | 15.82× | 4.86 | 4 | 3.20 h |
| `sdpa-graphmask` | 107.5 ± 25.2 | 18.66× | 4.95 | 3 | 2.72 h |
| `sdpa` | 103.7 ± 25.6 | 19.35× | 4.86 | 2 | 2.62 h |

### tag / reddit — 24 real batches (B=1, train split)

L: mean 1067, range 1024–1536, 2 distinct (L,N) shapes (ladder step 512); N max 30; padding 8%; dtype=bf16; k_hop=0; grad-ckpt=True

| method | step ms | vs eager | peak GB | cold pass s | epoch est. |
|---|---|---|---|---|---|
| `eager` | 406.3 ± 311.6 | 1.00× | 6.17 | 10 | 0.38 h |
| `flex` | 201.3 ± 39.8 | 2.02× | 4.77 | 12 | 0.19 h |
| `flex-nobias` | 125.5 ± 22.4 | 3.24× | 4.77 | 5 | 0.12 h |
| `sdpa-graphmask` | 103.6 ± 20.1 | 3.92× | 4.82 | 3 | 0.10 h |
| `sdpa` | 103.6 ± 24.5 | 3.92× | 4.77 | 2 | 0.10 h |

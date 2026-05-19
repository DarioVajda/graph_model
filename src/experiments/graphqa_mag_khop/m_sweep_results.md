# M-sweep Results

Effect of truncating the magnetic Laplacian eigenvectors to M on GraphQA
shortest-path exact-match accuracy. M=all means no truncation.
Graph sizes range from 6 to 20 nodes (mean 13), so M=16 ≈ M=all for most graphs.

| M | K=0 | K=1 | K=2 | K=3 |
| --- | --- | --- | --- | --- |
| all | 0.827 ± 0.123 (n=3) | 0.881 ± 0.011 (n=3) | 0.783 ± 0.097 (n=3) | 0.784 ± 0.000 (n=2) |
| 8 | 0.711 ± 0.008 (n=3) | 0.685 ± 0.024 (n=3) | — | — |
| 4 | 0.643 ± 0.041 (n=3) | 0.660 ± 0.016 (n=3) | — | — |

*Values are mean ± std over multiple seeds. Single runs shown as bare accuracy.*

# Real-input benchmark — consolidated

Step latency in ms (mean over timed passes), **sorted by true sequence
length**. `L real` is the mean unpadded token count per graph; `L padded`
is the tensor width actually run, after bucketing. `flex vs eager` > 1
means flex is faster.

**Compare along a row, not down a column.** Rows differ in pad mode and
length ladder, and `pad=batch` rows run the dense arms at their natural
per-batch L, where flex cannot run at all.

**Absolute times are comparable across rows only at equal `dtype` and
`B`.** GraphQA's paper recipe is fp32/B=4 and TAG's is bf16/B=1, so a
graphqa row can show a larger absolute cost than a TAG row with longer
`L real` — it is pushing `tok/step` tokens through fp32 arithmetic. Use
the bf16/B=1 graphqa rows for any cross-experiment reading; the ratio
columns are within-row and always valid.

The arms, in order of how much graph machinery they carry:

| arm | mask | bias | kernel |
|---|---|---|---|
| `sdpa` | plain causal | — | fused flash — the theoretical floor |
| `sdpa+mask` | GTLM's | — | dense SDPA |
| `flex-nobias` | GTLM's | — | flex block-sparse |
| `eager` | GTLM's | full | dense |
| `flex` | GTLM's | full | flex block-sparse |

`sdpa` is deliberately the most favourable baseline: plain causal is
flash-eligible, and block skipping is precisely what GTLM gives up by
construction. `sdpa+mask` prices the mask shape alone — on Cora GTLM's
mask admits 0.70 of the L×L matrix against plain causal's 0.50.

⚠stall — some arm's mean exceeds its trimmed mean by >10%, the signature
of a stray allocator stall. ⚠stale — written before that bug was fixed,
so it cannot be checked; treat as indicative only.

| exp | arm | L real | L padded | tok/step | dtype | B | pad | gc | eager | flex | flex-nobias | sdpa+mask | sdpa | flex vs eager | flex vs sdpa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graphqa | standard/node_count | 28 | 128 | 128 | bf16 | 1 | bucket | off | 112.3 | 129.1 | 69.5 | 50.9 | 52.5 | 0.87× | 2.46× |
| graphqa | standard/node_count ⚠stale | 29 | 33 | 132 | fp32 | 4 | batch | off | 101.5 | — | — | 40.4 | 41.5 | — | — |
| graphqa | standard/node_count ⚠stale | 29 | 512 | 2048 | fp32 | 4 | bucket | off | 925.1 | 355.0 | — | — | 271.3 | 2.61× | 1.31× |
| graphqa | standard/node_count | 29 | 128 | 512 | bf16 | 4 | bucket | off | 114.5 | — | — | — | 48.9 | — | — |
| graphqa | standard/node_count ⚠stale | 29 | 128 | 512 | fp32 | 4 | bucket | off | 128.5 | 114.0 | 93.3 | — | 79.0 | 1.13× | 1.44× |
| graphqa | standard/shortest_path | 37 | 128 | 128 | bf16 | 1 | bucket | off | 113.0 | 124.0 | 66.2 | 51.1 | 50.7 | 0.91× | 2.45× |
| graphqa | standard/shortest_path ⚠stale | 38 | 44 | 174 | fp32 | 4 | batch | off | 101.8 | — | — | — | 44.0 | — | — |
| graphqa | standard/shortest_path ⚠stale | 38 | 128 | 512 | fp32 | 4 | bucket | off | 129.4 | 113.4 | 93.4 | — | 79.4 | 1.14× | 1.43× |
| graphqa | incidence/node_count ⚠stale | 137 | 255 | 1020 | fp32 | 4 | batch | off | 362.0 | — | — | 155.6 | 155.0 | — | — |
| graphqa | incidence/node_count | 137 | 325 | 1301 | bf16 | 4 | bucket | off | 372.1 | — | — | — | 60.2 | — | — |
| graphqa | incidence/node_count ⚠stale | 137 | 325 | 1301 | fp32 | 4 | bucket | off | 522.2 | 319.4 | 213.8 | — | 183.0 | 1.63× | 1.75× |
| graphqa | incidence/node_count | 138 | 208 | 208 | bf16 | 1 | bucket | off | 115.1 | 135.7 | 73.6 | 53.1 | 54.5 | 0.85× | 2.49× |
| graphqa | incidence/shortest_path ⚠stale | 147 | 263 | 1052 | fp32 | 4 | batch | off | 372.8 | — | — | — | 158.1 | — | — |
| graphqa | incidence/shortest_path ⚠stale | 147 | 325 | 1301 | fp32 | 4 | bucket | off | 523.8 | 319.1 | 212.2 | — | 182.6 | 1.64× | 1.75× |
| graphqa | incidence/shortest_path | 147 | 208 | 208 | bf16 | 1 | bucket | off | 112.9 | 121.8 | 67.1 | 49.1 | 49.3 | 0.93× | 2.47× |
| tag | cora ⚠stale | 660 | 917 | 917 | bf16 | 1 | bucket | off | 764.7 | 127.8 | — | 53.9 | 55.3 | 5.98× | 2.31× |
| tag | cora | 663 | 917 | 917 | bf16 | 1 | bucket | on | 836.9 | 226.8 | 143.0 | 107.0 | 118.6 | 3.69× | 1.91× |
| tag | pubmed | 950 | 1195 | 1195 | bf16 | 1 | bucket | on | 1255.3 | 215.9 | 137.2 | 113.3 | 112.1 | 5.81× | 1.93× |
| tag | pubmed ⚠stale | 955 | 1195 | 1195 | bf16 | 1 | bucket | off | 1150.1 | 130.4 | 71.1 | — | 56.6 | 8.82× | 2.30× |
| tag | reddit ⚠stale | 975 | 1024 | 1024 | bf16 | 1 | bucket | off | 251.4 | 117.6 | 67.3 | — | 50.8 | 2.14× | 2.31× |
| tag | reddit | 976 | 1067 | 1067 | bf16 | 1 | bucket | on | 406.3 | 201.3 | 125.5 | 103.6 | 103.6 | 2.02× | 1.94× |
| tag | ogbn-arxiv | 1065 | 1387 | 1387 | bf16 | 1 | bucket | on | 2006.2 | 208.3 | 126.8 | 107.5 | 103.7 | 9.63× | 2.01× |

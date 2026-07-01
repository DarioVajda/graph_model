# KGQA experiment (WIP)

Feeding GNN-RAG's retrieved KG subgraphs directly into a single GTLM model, replacing
GNN-RAG's GNN-reasoner + LLM-reader pipeline. Starting on **SR-WebQSP**.

> **NOTE:** This README is a temporary stub holding the answer-coverage measurements
> below. A full rewrite (task description, data-prep pipeline, usage) is pending.

## Answer-coverage ceilings (SR-WebQSP)

These bound how well *any* model can do given SR retrieval — the input either contains
the answer or it doesn't. WebQSP reports **macro** metrics (per-question, averaged over
questions), so the macro rows are the operative ceilings; micro is diagnostic only.

Measured from `data/data/sr-webqsp/{train,dev,test}.json` (answer present ⟺ its `kb_id`
is a node in `subgraph.tuples`). "Perfect precision" = model emits only correct, present
golds; `N_max=20` = generation capped at 20 answers.

| Ceiling | **test** (n=1628) | train (n=2826) | dev (n=246) | Bounds |
|---|---|---|---|---|
| ≥1 gold present per question | **91.1%** | 92.6% | 89.8% | **Hits@1** |
| Recall — macro (avg per-q present/total) | 89.2% | 90.5% | 86.7% | per-q recall |
| Recall — micro (Σpresent/Σtotal) | 63.3% | 56.9% | 34.0% | answer-instance recall *(diagnostic)* |
| **F1 — macro**, perfect precision, uncapped | **89.6%** | 91.0% | 87.1% | **macro-F1 (WebQSP metric)** |
| Recall — macro, cap N_max=20 | 86.4% | 87.9% | 85.2% | — |
| F1 — macro, cap N_max=20 | **87.4%** | 88.9% | 86.1% | macro-F1 under our cap |

**Reading it:**
- Operative test ceilings: **Hits@1 ≤ 91.1%**, **macro-F1 ≤ 89.6%** (→ **87.4%** with the N_max=20 cap).
- The `max_nodes=512` graph-size cap (paths-guided, round-robin truncation) lowers the test
  Hits@1 ceiling by only **1.4 pt → 89.7%** (train 92.6→91.2%); ~10.6% of train questions have
  no graph-present answer post-cap and are dropped from training (kept for eval).
- micro (63.3%) ≪ macro (89.2%): entirely the enumeration tail (6.8% of questions have >20 golds,
  up to 3688). Micro weights every (q, answer) pair equally so those questions dominate; it is
  **not** a benchmark ceiling — don't optimize for it.
- The N_max=20 cap costs only ~2 macro-F1 points → cheap.
- All rows assume perfect precision, so real achievable numbers are strictly below.
- GNN-RAG's SR Hits@1 (~78.9) sits ~12 pts under the 91.1% ceiling — that gap is the
  graph-reasoning headroom GTLM targets (genuine reasoning, not retrieval failure).

Answer-set sizes: median 1, mean 11.2; 52% single-answer, 31% 2–5, 10.5% 6–20, 6.8% >20.

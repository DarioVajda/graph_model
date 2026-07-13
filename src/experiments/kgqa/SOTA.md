# KGQA SOTA — method summaries

Companion to the [Published SOTA landscape](README.md#published-sota-landscape-as-of-2026-07)
table in the README (which holds the numbers and metric caveats). This file explains *how each
method works*, grouped by family — the grouping matters more than the ranking, because the
families differ in what they hold fixed (retrieval, KG access, entity linking) and therefore in
how comparable they are to our fixed-SR-subgraph setting.

Last reviewed: 2026-07.

---

## Family 1 — Fixed-subgraph retrieval + reader (our setting)

Retrieve once, then reason over the retrieved evidence in a single pass. Bounded by their
retriever's answer coverage, exactly like us.

### RoG — Reasoning on Graphs (2023, [arXiv:2310.01061](https://arxiv.org/abs/2310.01061))

*Planning–retrieval–reasoning* with one fine-tuned Llama-2-7B. The LLM first generates
**relation paths** (e.g. `child_of → spouse`) as a faithful plan, those paths are grounded in
the KG to retrieve concrete reasoning paths from the topic entities, and the same LLM then
reasons over the verbalized paths to answer. Training distills valid KG paths into the model
via two instruction-tuning objectives (plan generation + answer generation). The
grandparent of most of this table.

### GNN-RAG (2024, [arXiv:2405.20139](https://arxiv.org/abs/2405.20139)) — our direct predecessor

Two-stage pipeline: a **GNN reasoner** (ReaRev-style) runs over a dense retrieved subgraph and
scores every node as a candidate answer; the shortest paths from topic entities to the
high-scoring candidates are extracted, **verbalized**, and handed to a frozen/tuned LLM reader
(Llama-2-7B) that produces the final answer. The GNN does the multi-hop structure work, the
LLM does the language work. **+RA** (retrieval augmentation) unions the GNN-retrieved paths
with RoG's LLM-generated paths. The **SR variant** consumes exactly our inputs (Zhang et al.'s
subgraph retriever) — paper Table 15 row (d), our retrieval-matched baseline on both
benchmarks: WebQSP **Hit 83.4 / Hits@1 78.9 / F1 69.8**, CWQ **Hit 60.6 / Hits@1 55.6 /
F1 53.3**. (SR *hurts* GNN-RAG on CWQ vs its dense retriever — 61.3 → 55.6 Hits@1; the paper
blames disconnected sparse subgraphs breaking its shortest-path extraction. We consume the
same sparse subgraphs directly, so nothing shields us from the same coverage loss — but we
have no path-extraction stage to break.)
GTLM's claim is that one graph-native model can replace this entire two-model pipeline.

### SubgraphRAG (ICLR 2025, [arXiv:2410.20724](https://arxiv.org/abs/2410.20724))

"Simple is effective": a **lightweight MLP retriever** (no GNN) scores individual triples
using question embeddings plus **Directional Distance Encoding** (structural features relative
to topic entities), takes the top-k triples as a flexible-size subgraph, and feeds them as
plain text to a **frozen, un-fine-tuned LLM** with CoT prompting. All the learning is in the
sub-second retriever; reader quality then scales freely with the plugged-in LLM (8B → GPT-4o).
Also the source of the Hit-vs-Hits@1 mislabeling audit we cite. Its retriever is public and
subgraph-shaped — the natural candidate for our step-4 retrieval-portability experiment.

### RPO-RAG (2026, [arXiv:2601.19225](https://arxiv.org/abs/2601.19225)) — our same-size anchor

Targets **small LLMs** (1B–8B). A semantic-matching retriever selects KG paths, organized into
answer-centered prompts; the reader is then trained with **relation-aware preference
optimization**: weakly-supervised preference pairs where relations from path clusters
semantically consistent with the question are "preferred" and dissimilar ones "non-preferred",
via a margin objective weighted by proximity to cluster centroids. Supervises the
*intermediate relation choices*, not just the final answer. Its Llama-3.2-1B row (69.8 F1
WebQSP) is the closest published comparison to our setup: same base model, single-pass
reading, ~3 F1 above us with a different retriever.

## Family 2 — Path generation / constrained decoding

No fixed subgraph: a model generates relation paths (or KG-constrained token sequences) that
are grounded against the full KG, then a larger LLM reasons over the grounded evidence.
Retrieval and reasoning are interleaved, so our coverage ceiling does not apply.

### GCR — Graph-Constrained Reasoning (2024, [arXiv:2410.13080](https://arxiv.org/abs/2410.13080))

Builds a **KG-Trie** of formatted paths starting at the question's topic entities, then a
KG-specialized fine-tuned Llama-3.1-8B decodes reasoning paths **under trie-constrained
decoding** — every generated path is a real KG path by construction (zero structural
hallucination). Multiple decoded paths + hypothesis answers are then passed to a general LLM
(GPT-4o-mini) for inductive aggregation. Strong Hit, mediocre F1: good at finding *an*
answer, weaker at enumerating the full answer set.

### PathISE (2026, [arXiv:2605.10791](https://arxiv.org/abs/2605.10791))

Attacks the supervision problem RoG/GCR share: which of the many paths reaching an answer are
actually *informative*? A transformer **Multiple-Instance-Learning estimator** treats all
answer-reaching candidate paths as a positive bag and scores each path's utility; top-scoring
paths become pseudo-labels that train a Llama-3.1-8B **path generator** by distillation. At
inference the generator beam-searches relation paths, grounds them in the KG, and hands the
compact grounded evidence to GPT-4o/4.1. Notable for us: reports genuine **Hits@1** (86.8
WebQSP) and re-scores GCR/SubgraphRAG under it — the cleanest metric hygiene in the table.

## Family 3 — Agentic / interactive KG access

No retrieval stage at all: an LLM agent queries the live KG over multiple turns (ToG lineage).
Coverage is bounded only by the KG itself, which is why this family tops the leaderboard.
Methodologically farthest from our single-pass setting.

### ToG — Think-on-Graph (2023, [arXiv:2307.07697](https://arxiv.org/abs/2307.07697)) *(context; not in our table)*

The lineage-founder: an LLM performs iterative **beam search on the KG**, at each step asked
to score and prune candidate relations/entities to expand. Purely prompted (GPT-4).

### ReKnoS (ICLR 2025, [arXiv:2503.22166](https://arxiv.org/abs/2503.22166))

Diagnoses ToG-style agents' failure mode: greedy forward-only expansion causes high
*non-retrieval* (agent never reaches the answer). Introduces **super-relations** — summaries
that bundle many concrete relation paths — letting the LLM reason **forward and backward**
over groups of paths at once, expanding the effective search space per LLM call. Prompted
(no training); numbers reported over Qwen backbones up to 235B.

### KG-R1 (2025, [arXiv:2509.26383](https://arxiv.org/abs/2509.26383))

A single **Qwen2.5-3B agent trained end-to-end with RL** (GRPO-style). The KG sits behind a
lightweight retrieval server with a schema-agnostic interface of four 1-hop operations
(head/tail relations, head/tail entities); rewards combine turn-level shaping (format, query
validity) with episode-level answer correctness and retrieval coverage. Schema-agnostic
actions make the trained agent **plug-and-play across KGs** without retraining. The proof
that small-model RL beats big-model prompting in this family.

### KnowCoder-A1 (2025, [arXiv:2510.25101](https://arxiv.org/abs/2510.25101))

Frames KBQA as **agentic code generation**: a Qwen2.5-Coder-7B emits thoughts plus tool calls
(`SearchTypes`, `SearchGraphPatterns`, `ExecuteSPARQL`) in a ReAct loop against the live KB.
Trained with **outcome-only supervision**: cold-start SFT on rejection-sampled successful
trajectories, then curriculum GRPO whose reward shifts from precision-focused F₀.₅ to F₁ —
no process/path annotations anywhere.

### TRACE (2026, [arXiv:2604.11193](https://arxiv.org/abs/2604.11193))

Fully **prompted** (GPT-4.1, no training). Three mechanisms on top of iterative KG traversal:
dynamic context generation (the evolving path is re-narrated in natural language to guide the
next relation choice), exploration generalization (past trajectories are abstracted into
reusable "experiential priors"), and dual-feedback re-ranking of candidate relations using
both. Its "Hits@1" is likely Hit (ToG-lineage convention) — treat as an upper bound.

### GraphWalker (2026, [arXiv:2603.28533](https://arxiv.org/abs/2603.28533)) — current F1 leader

Agent with two primitive KG tools — `get_relations(e)` and `get_triples(e, R')` — trained on a
**synthetic trajectory curriculum**: 15k structurally diverse constrained-random-walk paths
(GraphSynth-15k) for broad exploration, then 6k expert trajectories demonstrating reflection
and error recovery (GraphRoll-6k), as two SFT stages on Qwen2.5-7B, followed by GRPO RL with
sparse exact-match reward. The recipe (synthetic-structure pretraining → behavior SFT → RL)
is the interesting part: it manufactures its graph supervision instead of mining it.

## Family 4 — Semantic parsing with oracle entities

Generate an executable logical form (S-expression → SPARQL), execute it on Freebase. Scored
**with gold topic entities given** — a materially easier setting (entity linking is the hard
part they skip), which is why they top CWQ by ~8 F1. Not comparable to end-to-end systems;
listed for leaderboard completeness.

### ChatKBQA (ACL 2024 Findings, [arXiv:2310.08975](https://arxiv.org/abs/2310.08975))

**Generate-then-retrieve**: a LoRA-fine-tuned Llama-2-13B generates the logical-form skeleton
directly from the question; entities and relations in the skeleton are then bound by
unsupervised dense retrieval over KB labels, and the result is converted to SPARQL and
executed. Inverts the classic retrieve-then-generate order — generation is easy for a tuned
LLM, binding is easy given a skeleton.

### PGDA-KGQA (2025, [arXiv:2506.09414](https://arxiv.org/abs/2506.09414))

ChatKBQA's framework plus **prompt-guided LLM data augmentation**: generates pseudo
(question, logical-form) pairs — semantically equivalent rephrasings, answer-preserving
perturbations, multi-hop compositions — to densify logical-form supervision before the same
fine-tune → bind → execute pipeline. Pure training-data play; the architecture is ChatKBQA's.

### KBQA-o1 (ICML 2025, [arXiv:2501.18922](https://arxiv.org/abs/2501.18922)) *(excluded from our table)*

Agentic logical-form construction with **MCTS**: a ReAct-style agent builds the logical form
stepwise while exploring the KB, MCTS with a reward model steering the search. Excluded from
the README table because it evaluates WebQSP only in a 100-shot low-resource setting (59.8 F1
@ Llama-3.1-8B) and skips CWQ — impressive sample efficiency, not a full-supervision SOTA
point.

---

## What this means for us (condensed)

- The leaderboard's top (Families 2–3) is **not our game**: gains come from interleaved or
  interactive KG access and big/tuned readers. Our fixed-SR setting is bounded at 87.3 F1
  regardless — see the README's Goal section for why we hold retrieval fixed anyway.
- Within **Family 1** — our actual peer group — the published frontier at our model size is
  RPO-RAG's Llama-3.2-1B (69.8 F1 WebQSP vs our 66.9), and at 8B it's ~81 F1. Both use their
  own retrievers, so the retrieval-matched target remains SR-GNN-RAG (Table 15(d):
  WebQSP 78.9 Hits@1 / 69.8 F1, CWQ 55.6 Hits@1 / 53.3 F1).
- Recurring ingredients across families that are compatible with our architecture:
  **supervising intermediate structure, not just answers** (RPO-RAG's relation preferences,
  PathISE's path scoring), **synthetic graph-structure curricula** (GraphWalker), and
  **RL on top of a converged SFT model** (KG-R1, GraphWalker, KnowCoder-A1).

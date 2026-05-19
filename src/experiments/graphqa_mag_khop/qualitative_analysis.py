"""
Qualitative analysis for graphqa_mag_khop.

Tasks
-----
1. Masking statistics: for each K in {0,1,2,3}, compute the fraction of
   prefix→prefix token pairs that are blocked by the k-hop gate, averaged
   over the full test set.  K=0 is always 0 % (no gate).

2. Attention plots: find the first test-set sample that every K model
   predicts correctly, then save per-layer attention heatmaps (mean across
   heads + individual heads) for each model.

Run from repo root:
    python -m src.experiments.graphqa_mag_khop.qualitative_analysis
"""

import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate.utils import send_to_device

from ...utils import GraphCollator
from ...utils.plot_attention import plot_graph_attention
from ...utils.plot_text_graph import visualize_text_graph
from ...train.eval import _load_model_from_checkpoint
from ..graphqa.load_dataset import load_graphqa_datasets

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR   = os.path.join(EXPERIMENT_DIR, "qualitative_analysis")
DATASETS_DIR   = "src/experiments/graphqa/processed_datasets"
MODEL_NAME     = "meta-llama/Llama-3.2-1B"

# Median checkpoint per K (2 runs → worse; 3 runs → middle)
# K=0: 0.886 (worse of 2: 0.910, 0.886)
# K=1: 0.880 (median of 3: 0.870, 0.880, 0.892)
# K=2: 0.736 (median of 3: 0.718, 0.736, 0.894)
# K=3: 0.784 (both runs tied at 0.784)
CHECKPOINTS = {
    0: "checkpoints/graphqa_mag_khop/GraphQA_shortest_path_K=0_M=0_v3/checkpoint-480",
    1: "checkpoints/graphqa_mag_khop/GraphQA_shortest_path_K=1_M=0/checkpoint-480",
    2: "checkpoints/graphqa_mag_khop/GraphQA_shortest_path_K=2_M=0_v3/checkpoint-520",
    3: "checkpoints/graphqa_mag_khop/GraphQA_shortest_path_K=3_M=0/checkpoint-520",
}


# ── Task 1: masking statistics ────────────────────────────────────────────────

def compute_masking_stats(test_dataset) -> dict:
    """
    For each K, return the fraction of prefix-node token pairs that the
    k-hop gate would block.  Computed on node-level boolean masks (no
    model forward pass required).
    """
    stats = {0: 0.0}

    for k in [1, 2, 3]:
        collator = GraphCollator(k_hop=k, magnetic_m=0)
        total = masked = 0

        for item in tqdm(test_dataset, desc=f"masking stats  K={k}"):
            N = item['num_nodes']
            p = item['prompt_node']
            if torch.is_tensor(p):
                p = p.item()

            node_lens = [len(ids) for ids in item['input_ids']]
            prefix    = [u for u in range(N) if u != p]
            mask      = collator._single_k_hop_mask(item['edges'], N, p)  # (N,N) bool

            for u in prefix:
                for v in prefix:
                    n_pairs  = node_lens[u] * node_lens[v]
                    total   += n_pairs
                    # Diagonal is always True; only cross-node pairs can be blocked
                    if u != v and not mask[u, v].item():
                        masked += n_pairs

        stats[k] = masked / total if total else 0.0
        print(f"  K={k}: {stats[k] * 100:.1f}% of prefix token pairs masked")

    return stats


# ── Task 2: attention plots ────────────────────────────────────────────────────

def _is_correct(model, item, collator, device) -> bool:
    """
    Return True iff the model's greedy predictions match all non-masked
    label tokens for this sample.

    Uses the causal-LM convention: logit at position (p-1) predicts the
    token at position p.  The prompt node's tokens occupy the last
    `prompt_len` positions of the flat sequence.
    """
    prompt_idx = item['prompt_node']
    if torch.is_tensor(prompt_idx):
        prompt_idx = prompt_idx.item()

    labels = item.get('labels')
    if labels is None:
        return False
    valid = labels != -100
    if not valid.any():
        return False

    batch = collator([item])
    batch = send_to_device(batch, device)
    batch.pop('labels', None)

    with torch.no_grad():
        outputs = model(input_ids=None, input_graph_batch=batch)

    logits     = outputs.logits[0]                      # (seq_len, vocab_size)
    seq_len    = logits.shape[0]
    prompt_len = len(item['input_ids'][prompt_idx])

    for j in range(prompt_len):
        if not valid[j]:
            continue
        logit_pos = seq_len - prompt_len + j - 1       # position predicting token j
        if logit_pos < 0:
            continue
        if logits[logit_pos].argmax().item() != labels[j].item():
            return False
    return True


def find_common_correct_index(test_dataset, device) -> int:
    """
    Return the smallest test-set index that all four K-models predict
    correctly.  Models are loaded and freed one at a time.
    """
    per_k_correct: dict[int, set] = {}

    for k, ckpt in CHECKPOINTS.items():
        print(f"\nScanning test set — K={k} ...")
        model, _ = _load_model_from_checkpoint(ckpt, device)
        model.eval()
        collator = GraphCollator(k_hop=k, magnetic_m=0)

        correct = {
            idx for idx in tqdm(range(len(test_dataset)))
            if _is_correct(model, test_dataset[idx], collator, device)
        }
        print(f"  K={k}: {len(correct)}/{len(test_dataset)} correct")
        per_k_correct[k] = correct

        del model
        torch.cuda.empty_cache()

    common = set.intersection(*per_k_correct.values())
    if not common:
        raise RuntimeError("No test sample found that all K-models predict correctly.")

    idx = sorted(common)[0]
    print(f"\nChosen sample index: {idx}")
    return idx


def plot_graph_structure(test_dataset, sample_idx: int) -> None:
    """Save a visualisation of the graph structure for sample_idx."""
    out_path = os.path.join(ANALYSIS_DIR, "attention_plots", "graph_structure.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    visualize_text_graph(
        graph_data=test_dataset[sample_idx],
        output_path=out_path,
        max_line_length=30,
        use_spectral_layout=False,
        figsize=(8, 8),
    )
    print(f"Saved → {out_path}")


def plot_all_k(test_dataset, sample_idx: int, tokenizer, device) -> None:
    """Load each K-model in turn and save attention heatmaps for sample_idx."""
    plot_graph_structure(test_dataset, sample_idx)

    for k, ckpt in CHECKPOINTS.items():
        print(f"\nPlotting attention — K={k} ...")
        model, _ = _load_model_from_checkpoint(ckpt, device)
        model.eval()
        collator = GraphCollator(k_hop=k, magnetic_m=0)

        out_dir = os.path.join(ANALYSIS_DIR, "attention_plots", f"K={k}")
        os.makedirs(out_dir, exist_ok=True)

        plot_graph_attention(
            model=model,
            graph_example=test_dataset[sample_idx],
            tokenizer=tokenizer,
            collator=collator,
            output_path=out_dir,
            plot_means=True,
            plot_heads=True,
            device=device,
        )

        del model
        torch.cuda.empty_cache()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("Loading test dataset...")
    _, test_dataset = load_graphqa_datasets(
        DATASETS_DIR, ["shortest_path"], ["shortest_path"], graph_type="standard",
    )

    # ── Task 1 ────────────────────────────────────────────────────────────────
    print("\n=== Task 1: Masking Statistics ===")
    stats = compute_masking_stats(test_dataset)
    stats_path = os.path.join(ANALYSIS_DIR, "masking_stats.json")
    with open(stats_path, "w") as f:
        json.dump(
            {f"K={k}": f"{v * 100:.2f}%" for k, v in sorted(stats.items())},
            f, indent=2,
        )
    print(f"Saved → {stats_path}")

    # ── Task 2 ────────────────────────────────────────────────────────────────
    print("\n=== Task 2: Attention Plots ===")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    sample_idx = find_common_correct_index(test_dataset, device)
    plot_all_k(test_dataset, sample_idx, tokenizer, device)
    print(f"\nDone. All outputs in {ANALYSIS_DIR}/")

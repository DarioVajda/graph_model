"""
Standard-batch collator for the v2 GTLM-Llama model.

Unlike :class:`GraphCollator` (v0/v1), which returns a ragged
``input_graph_batch`` dict consumed by a heavily customised ``forward``, this
collator does all sequence *packing* up-front and emits a flat, HuggingFace-
idiomatic batch:

    input_ids       (B, L)      long   - packed token ids, right padded
    position_ids    (B, L)      long   - per-node position ids (reset each node)
    node_ids        (B, L)      long   - which graph node each token belongs to
    attention_mask  (B, L)      long   - 1 for real tokens, 0 for padding
    labels          (B, L)      long   - -100 everywhere except the prompt span
    prompt_node     (B,)        long
    num_nodes       (B,)        long

plus the per-graph structural features needed by the attention biases:

    shortest_path_dists   (B, N, N)        long
    laplacian_coordinates (B, N, S)        float
    rwse                  (B, N, R)        float
    rrwp                  (B, N, N, T)      float
    magnetic_V            (B, N, M, 2)      float   (None when absent)
    magnetic_lambdas      (B, M)            float   (None when absent)
    k_hop_mask            (B, N, N)         bool    (only when k_hop > 0)

Every entry is a plain tensor (magnetic is split into two tensors rather than a
tuple) so that the HF ``Trainer`` keeps them as columns and the model's
``forward`` can name them directly in its signature — no ``remove_unused_columns``
hack required.

Packing semantics (identical to v0's ``_prepare_inputs`` with ``padding_side
="right"``):
  * non-prompt nodes are concatenated first, in node-index order, then the
    prompt node is appended last;
  * ``position_ids`` restart from 0 at every node boundary;
  * ``labels`` for graph i are placed at ``[prefix_len : prefix_len + prompt_len]``
    so that HF's shift-by-one causal loss predicts the prompt tokens.
"""

import torch

from .text_graph_dataset import TextGraph


# ── K-hop reachability helpers (kept self-contained for the v2 stack) ──────────

def _k_hop_reachability(A: torch.Tensor, K: int) -> torch.Tensor:
    """
    Boolean (N, N) reachability in 1 … K directed steps along adjacency ``A``.

    The diagonal is NOT set here; callers add it explicitly.  O(K · N^2).
    """
    A_float   = A.float()
    reachable = A.clone()
    power     = A_float.clone()
    for _ in range(K - 1):
        power     = torch.mm(power, A_float)
        reachable = reachable | (power > 0)
    return reachable


def _single_k_hop_mask(edges, N: int, prompt_node: int, k_hop: int, directed: bool = False) -> torch.Tensor:
    """
    (N, N) boolean K-hop mask for one graph.

    Entry ``mask[i, j]`` is True iff query token-node ``i`` may attend to key
    token-node ``j``.

    Prefix↔prefix reachability is computed on the graph with the prompt node's
    edges removed (so the prompt cannot act as a structural bridge); the prompt
    node's own row/column use the full graph.  The diagonal is always True.

    When ``directed`` is False the graph is symmetrised, so reachability is
    distance ≤ K in either direction.  When ``directed`` is True edges are
    traversed only in their own direction — ``mask[i, j]`` is True iff there is a
    directed path ``i -> j`` of length ≤ K (out-edge traversal from the query).
    """
    p = prompt_node

    A_full = torch.zeros(N, N, dtype=torch.bool)
    for u, v in edges:
        if 0 <= u < N and 0 <= v < N:
            A_full[u, v] = True
            if not directed:
                A_full[v, u] = True

    A_prefix = A_full.clone()
    A_prefix[p, :] = False
    A_prefix[:, p] = False

    R_prefix = _k_hop_reachability(A_prefix, k_hop)
    R_full   = _k_hop_reachability(A_full,   k_hop)

    mask       = R_prefix.clone()
    mask[p, :] = R_full[p, :]
    mask[:, p] = R_full[:, p]
    mask.fill_diagonal_(True)
    return mask


# ── Collator ───────────────────────────────────────────────────────────────────

class GraphCollatorV2:
    """Collate a list of :class:`TextGraph` items into a standard packed batch."""

    def __init__(self, tokenizer=None, pad_token_id: int = None, k_hop: int = 0,
                 k_hop_directed: bool = False, magnetic_m: int = 0):
        """
        Args:
            tokenizer:   Optional tokenizer; used only to source ``pad_token_id``.
            pad_token_id: Explicit pad id; overrides the tokenizer's. Defaults to
                         the tokenizer's pad (or eos) id, else 0. Pad positions are
                         masked out of both attention and loss, so the exact value
                         is immaterial to the result.
            k_hop:       When > 0 a boolean (N, N) K-hop mask is emitted per batch.
            k_hop_directed: When True the K-hop mask follows edge direction (query
                         i attends to key j iff a directed path i -> j of length
                         ≤ K exists); when False the graph is symmetrised. To keep
                         a checkpoint reproducible, construct this from the model's
                         config: ``GraphCollatorV2(k_hop=cfg.k_hop,
                         k_hop_directed=cfg.k_hop_directed)``.
            magnetic_m:  When > 0, magnetic eigenvectors are truncated to the first
                         min(stored_m, magnetic_m) columns before batching.
        """
        self.tokenizer      = tokenizer
        self.k_hop          = k_hop
        self.k_hop_directed = k_hop_directed
        self.magnetic_m     = magnetic_m

        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        elif tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
            self.pad_token_id = tokenizer.pad_token_id
        elif tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
            self.pad_token_id = tokenizer.eos_token_id
        else:
            self.pad_token_id = 0

    def __call__(self, batch: list[TextGraph]) -> dict:
        B = len(batch)
        sizes        = torch.tensor([item['num_nodes'] for item in batch], dtype=torch.long)
        prompt_nodes = torch.tensor([item['prompt_node'] for item in batch], dtype=torch.long)
        max_num_nodes = int(sizes.max().item())

        has_labels = "labels" in batch[0] and batch[0]["labels"] is not None

        # ── Pack the per-node token sequences (prompt node last) ───────────────
        packed = [self._pack_one(item) for item in batch]
        max_len = max(p['length'] for p in packed)

        input_ids      = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        position_ids   = torch.zeros((B, max_len), dtype=torch.long)
        node_ids       = prompt_nodes.view(B, 1).expand(B, max_len).clone()  # pad → prompt node
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels         = torch.full((B, max_len), -100, dtype=torch.long) if has_labels else None

        for i, p in enumerate(packed):
            L = p['length']
            input_ids[i, :L]      = p['tokens']
            position_ids[i, :L]   = p['positions']
            node_ids[i, :L]       = p['nodes']
            attention_mask[i, :L] = 1
            if has_labels:
                lab = batch[i]['labels']
                s, e = p['prefix_len'], p['prefix_len'] + p['prompt_len']
                if lab.shape[0] != p['prompt_len']:
                    raise ValueError(
                        f"labels for graph {i} have length {lab.shape[0]} but the prompt "
                        f"node has {p['prompt_len']} tokens."
                    )
                labels[i, s:e] = lab

        out = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'node_ids': node_ids,
            'attention_mask': attention_mask,
            'prompt_node': prompt_nodes,
            'num_nodes': sizes,
        }
        if has_labels:
            out['labels'] = labels

        # ── Structural feature tensors (padded to max_num_nodes) ───────────────
        out.update(self._collate_features(batch, B, max_num_nodes))

        if self.k_hop > 0:
            masks = torch.zeros(B, max_num_nodes, max_num_nodes, dtype=torch.bool)
            for i, item in enumerate(batch):
                N = item['num_nodes']
                masks[i, :N, :N] = _single_k_hop_mask(
                    item['edges'], N, item['prompt_node'], self.k_hop,
                    directed=self.k_hop_directed,
                )
            out['k_hop_mask'] = masks

        return out

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _pack_one(self, item: TextGraph) -> dict:
        """Build packed token/position/node tensors for one graph (prompt last)."""
        graph_input_ids = [torch.as_tensor(ids, dtype=torch.long) for ids in item['input_ids']]
        prompt_idx = int(item['prompt_node'])
        order = [j for j in range(len(graph_input_ids)) if j != prompt_idx] + [prompt_idx]

        tokens    = torch.cat([graph_input_ids[j] for j in order])
        positions = torch.cat([torch.arange(graph_input_ids[j].shape[0]) for j in order])
        nodes     = torch.cat([
            torch.full((graph_input_ids[j].shape[0],), j, dtype=torch.long) for j in order
        ])

        prompt_len = graph_input_ids[prompt_idx].shape[0]
        prefix_len = tokens.shape[0] - prompt_len
        return {
            'tokens': tokens,
            'positions': positions,
            'nodes': nodes,
            'length': tokens.shape[0],
            'prefix_len': prefix_len,
            'prompt_len': prompt_len,
        }

    def _collate_features(self, batch: list[TextGraph], B: int, max_num_nodes: int) -> dict:
        """Pad the per-graph structural features into dense batch tensors."""
        spectral_dim = batch[0]['laplacian_coordinates'].shape[1] if "laplacian_coordinates" in batch[0] else 0
        rwse_dim     = batch[0]['rwse'].shape[1] if "rwse" in batch[0] else 0
        max_rw_steps = batch[0]['rrwp'].shape[2] if "rrwp" in batch[0] else 0

        laplacian_coordinates = torch.zeros(B, max_num_nodes, spectral_dim, dtype=torch.float)
        shortest_path_dists   = torch.full((B, max_num_nodes, max_num_nodes), max_num_nodes, dtype=torch.long)
        rwse                  = torch.zeros(B, max_num_nodes, rwse_dim, dtype=torch.float)
        rrwp                  = torch.zeros(B, max_num_nodes, max_num_nodes, max_rw_steps, dtype=torch.float)

        max_m = max(
            (item['magnetic_V'].shape[1] for item in batch if 'magnetic_V' in item),
            default=max_num_nodes,
        )
        if self.magnetic_m > 0:
            max_m = min(max_m, self.magnetic_m)
        magnetic_V       = torch.zeros(B, max_num_nodes, max_m, 2, dtype=torch.float)
        magnetic_lambdas = torch.zeros(B, max_m, dtype=torch.float)
        any_magnetic = False

        for i, item in enumerate(batch):
            n = item['num_nodes']
            if "laplacian_coordinates" in item:
                laplacian_coordinates[i, :n, :] = item['laplacian_coordinates'].detach().clone()
            if "shortest_path_dists" in item:
                shortest_path_dists[i, :n, :n] = item['shortest_path_dists'].detach().clone()
            if "rwse" in item:
                rwse[i, :n, :] = item['rwse'].detach().clone()
            if "rrwp" in item:
                rrwp[i, :n, :n, :] = item['rrwp'].detach().clone()
            if "magnetic_V" in item and "magnetic_lambdas" in item:
                m_eff = item['magnetic_V'].shape[1]
                if self.magnetic_m > 0:
                    m_eff = min(m_eff, self.magnetic_m)
                magnetic_V[i, :n, :m_eff, :] = item['magnetic_V'][:, :m_eff, :].detach().clone()
                magnetic_lambdas[i, :m_eff]  = item['magnetic_lambdas'][:m_eff].detach().clone()
                any_magnetic = True

        return {
            'laplacian_coordinates': laplacian_coordinates,
            'shortest_path_dists': shortest_path_dists,
            'rwse': rwse,
            'rrwp': rrwp,
            'magnetic_V': magnetic_V if any_magnetic else None,
            'magnetic_lambdas': magnetic_lambdas if any_magnetic else None,
        }

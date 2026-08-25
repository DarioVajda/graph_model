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
from ..models.flex_kernel import bucketize, default_len_buckets, default_node_buckets


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
                 k_hop_directed: bool = False, magnetic_m: int = 0,
                 pad_to_block: bool = False, block_size: int = 128,
                 len_buckets=None, node_buckets=None,
                 node_position_mode: str = "reset", max_spd: int = 64,
                 landmark_d_max: int = 8, landmark_required: bool = False):
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
            pad_to_block: When True (required for ``graph_attn_impl='flex'``), both
                         the packed sequence length L *and* the node count N are
                         padded up to coarse buckets (see ``len_buckets`` /
                         ``node_buckets``) instead of the raw batch max. This keeps
                         the flex kernel off its ~14x unaligned-length cliff AND
                         bounds the number of distinct (L, N) shapes the compiled
                         flex kernel guards on — both L and N trigger recompiles,
                         so without N bucketing every distinct max-node-count would
                         force a fresh autotune. Harmless for the dense backends
                         (padded tokens/nodes are masked / unused).
            block_size:  Kernel block-mask alignment for L (64 | 128). Every L
                         bucket must be a multiple of it; 128 is safe for both flex
                         block sizes (a 128-aligned L is also 64-divisible, #6).
            len_buckets: Sequence-length bucketing spec — ``None`` (use the default
                         512-multiple+midpoint ladder), a callable ``f(L)->L``, or
                         a sorted list of allowed lengths. Each resulting bucket
                         must be a multiple of ``block_size``.
            node_buckets: Node-count bucketing spec — ``None`` (use the default
                         power-of-two-floored-at-32 ladder), a callable
                         ``f(N)->N``, or a sorted list of allowed node counts. No
                         alignment constraint (N is not a kernel tile dimension).
            node_position_mode: How each node's tokens' ``position_ids`` start
                         (E3, TODO.md). ``"reset"`` (default) — every node
                         restarts at local position 0, the historical behavior;
                         RoPE then carries zero inter-node relative-position
                         signal. ``"spd_depth"`` — a node's tokens start at
                         ``STRIDE * depth(node)``, where ``depth`` is its
                         shortest-path distance from the prompt node (clamped to
                         ``max_spd``, requires ``shortest_path_dists`` on every
                         item) and the prompt node's own depth is one past the
                         deepest prefix node. With a single-node (prompt-only)
                         graph both modes are identical to plain ``arange(len)``
                         (backward compatible); depth is a pure function of
                         graph structure, invariant to input node-listing order
                         (permutation equivariant). See ``_node_base_offsets``.
            max_spd:     Depth clamp for ``"spd_depth"`` (ignored otherwise) —
                         the same far/unreachable bucket cap ``SPDBias`` itself
                         clamps into; construct from the model's config
                         (``GraphCollatorV2(..., max_spd=cfg.max_spd)``).
        """
        self.tokenizer      = tokenizer
        self.k_hop          = k_hop
        self.k_hop_directed = k_hop_directed
        self.magnetic_m     = magnetic_m
        self.pad_to_block   = pad_to_block
        self.block_size     = block_size
        self.len_buckets    = len_buckets  if len_buckets  is not None else default_len_buckets
        self.node_buckets   = node_buckets if node_buckets is not None else default_node_buckets
        self.node_position_mode = node_position_mode
        self.max_spd        = max_spd
        # PAD symbol for the landmark block: the alphabet is {0..d_max}, UNREACH
        # (d_max+1), PAD (d_max+2). Must match the value the dataset column was
        # built with — `landmark_d_max` is part of that column, not a free knob.
        self.landmark_d_max = landmark_d_max
        # A cache built before the landmark column existed yields items with no
        # 'landmark' key, LandmarkBias then returns None, and the run trains
        # cleanly with NO bias at all — the exact silent-negative this repo has
        # been bitten by (see config.py's gate warnings). Refuse instead.
        self.landmark_required = landmark_required

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

        # Flex bucketing: pad both L (block-aligned, off the ~14x cliff) and N
        # (the (B,H,N,N) bias dim the compiled kernel guards on) up to coarse
        # buckets so torch.compile sees few distinct (L, N) shapes. The pad
        # defaults below (pad_token_id / 0 / prompt node / mask 0 / label -100)
        # are already correct for the extra positions, and the extra node slots
        # are simply never gathered — so only the allocation sizes change.
        target_len = bucketize(max_len, self.len_buckets) if self.pad_to_block else max_len
        target_n   = bucketize(max_num_nodes, self.node_buckets) if self.pad_to_block else max_num_nodes
        if self.pad_to_block and target_len % self.block_size != 0:
            raise ValueError(
                f"len bucket {target_len} (from L={max_len}) is not a multiple of "
                f"block_size={self.block_size}; the flex kernel requires "
                f"block-aligned lengths. Fix the len_buckets spec.")

        input_ids      = torch.full((B, target_len), self.pad_token_id, dtype=torch.long)
        position_ids   = torch.zeros((B, target_len), dtype=torch.long)
        node_ids       = prompt_nodes.view(B, 1).expand(B, target_len).clone()  # pad → prompt node
        attention_mask = torch.zeros((B, target_len), dtype=torch.long)
        labels         = torch.full((B, target_len), -100, dtype=torch.long) if has_labels else None

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

        # ── Structural feature tensors (padded to the node bucket target_n) ────
        out.update(self._collate_features(batch, B, target_n))

        if self.k_hop > 0:
            masks = torch.zeros(B, target_n, target_n, dtype=torch.bool)
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

        base_offset = self._node_base_offsets(item, graph_input_ids, prompt_idx)

        tokens    = torch.cat([graph_input_ids[j] for j in order])
        positions = torch.cat([base_offset[j] + torch.arange(graph_input_ids[j].shape[0])
                               for j in order])
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

    def _node_base_offsets(self, item: TextGraph, graph_input_ids: list, prompt_idx: int) -> dict:
        """Per-node starting position for this graph's tokens (E3, TODO.md) —
        ``{node_idx: int}``, added to each node's local ``arange(len)``.

        ``"reset"``: every node starts at 0 — byte-identical to the
        pre-E3 code (pinned by ``test_reset_mode_is_a_noop`` in
        ``tests/test_node_position_encoding.py``).

        ``"spd_depth"``: node ``j``'s offset is ``STRIDE * depth(j)``, where
        ``depth(j)`` is ``shortest_path_dists[prompt_idx, j]`` clamped to
        ``self.max_spd`` (the same far/unreachable bucket ``SPDBias`` itself
        clamps into — see ``src/models/bias.py``'s ``SPDBias.forward``), and
        ``depth(prompt_idx)`` is defined as one past the deepest prefix node
        so the prompt's tokens (the generation query) sit at strictly larger
        RoPE positions than any prefix node — generalizing "prompt packed
        last" into position space. ``STRIDE`` is computed per-graph as the
        longest node's token count, so no node's local ``arange`` can ever
        spill into the next depth band.

        With no prefix nodes (single-node, prompt-only graph) ``depth`` is
        vacuously 0 for the prompt and every offset is 0 regardless of mode —
        this is *why* single-node graphs collapse to plain ``arange(len)``
        (backward compatible). ``depth`` is a pure function of graph
        structure (shortest-path distance), not of input node-listing order,
        so it is unchanged by how the caller orders/labels nodes on input
        (permutation equivariant).
        """
        n = len(graph_input_ids)
        if self.node_position_mode == "reset":
            return {j: 0 for j in range(n)}

        spd = item.get('shortest_path_dists')
        if spd is None:
            raise ValueError(
                "node_position_mode='spd_depth' requires 'shortest_path_dists' on "
                "every item (build the dataset with spd=True) — got an item without it."
            )
        spd_row = spd[prompt_idx]
        prefix = [j for j in range(n) if j != prompt_idx]
        depth = {j: min(int(spd_row[j].item()), self.max_spd) for j in prefix}
        depth[prompt_idx] = (max(depth.values()) + 1) if prefix else 0

        stride = max(int(t.shape[0]) for t in graph_input_ids)
        return {j: stride * depth[j] for j in range(n)}

    def _collate_features(self, batch: list[TextGraph], B: int, n_pad: int) -> dict:
        """Pad the per-graph structural features into dense (B, n_pad, …) tensors.

        ``n_pad`` is the target node-count (the bucketed ``target_n`` under
        ``pad_to_block``, otherwise the raw batch max); each graph fills its
        first ``num_nodes`` slots and the rest stay at the pad value."""
        spectral_dim = batch[0]['laplacian_coordinates'].shape[1] if "laplacian_coordinates" in batch[0] else 0
        rwse_dim     = batch[0]['rwse'].shape[1] if "rwse" in batch[0] else 0
        max_rw_steps = batch[0]['rrwp'].shape[2] if "rrwp" in batch[0] else 0

        laplacian_coordinates = torch.zeros(B, n_pad, spectral_dim, dtype=torch.float)
        shortest_path_dists   = torch.full((B, n_pad, n_pad), n_pad, dtype=torch.long)
        rwse                  = torch.zeros(B, n_pad, rwse_dim, dtype=torch.float)
        rrwp                  = torch.zeros(B, n_pad, n_pad, max_rw_steps, dtype=torch.float)

        max_m = max(
            (item['magnetic_V'].shape[1] for item in batch if 'magnetic_V' in item),
            default=n_pad,
        )
        if self.magnetic_m > 0:
            max_m = min(max_m, self.magnetic_m)
        magnetic_V       = torch.zeros(B, n_pad, max_m, 2, dtype=torch.float)
        magnetic_lambdas = torch.zeros(B, max_m, dtype=torch.float)
        any_magnetic = False

        # Landmark anchor coordinates (B, n_pad, 3, k), integer symbols. Padded
        # NODES and padded ANCHOR slots both take the PAD symbol, which LandmarkBias
        # maps to a zero table row — so padding contributes exactly 0 rather than a
        # spurious distance. PAD is d_max+2 and lives in the stored column already;
        # here the only pad written is for node slots beyond num_nodes.
        lm_k = max((item['landmark'].shape[-1]
                    for item in batch if item.get('landmark') is not None),
                   default=0)
        if self.landmark_required and lm_k == 0:
            raise ValueError(
                "landmark bias is enabled but no item carries a 'landmark' column. "
                "Add it to the cache in place:\n  python -m "
                "src.experiments.bias_experiments.landmark.add_landmark_column "
                "--roots <processed_datasets>/<cache>\n"
                "Refusing to train an unbiased run that would read as a clean null.")
        landmark, any_landmark = None, False
        if lm_k:
            pad_sym = self.landmark_d_max + 2
            landmark = torch.full((B, n_pad, 3, lm_k), pad_sym, dtype=torch.long)

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
            if landmark is not None and item.get("landmark") is not None:
                lm = item['landmark']
                landmark[i, :n, :, :lm.shape[-1]] = lm.detach().clone()
                any_landmark = True

        return {
            'laplacian_coordinates': laplacian_coordinates,
            'shortest_path_dists': shortest_path_dists,
            'rwse': rwse,
            'rrwp': rrwp,
            'magnetic_V': magnetic_V if any_magnetic else None,
            'magnetic_lambdas': magnetic_lambdas if any_magnetic else None,
            'landmark': landmark if any_landmark else None,
        }

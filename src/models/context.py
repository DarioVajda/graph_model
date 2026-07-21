"""
Typed per-batch graph context.

The causal-LM forward builds one :class:`GraphContext` per batch and installs it
on every attention module (as ``module._graph_ctx``). The registered ``gtlm_*``
attention functions read it back to apply the structural mask + soft bias. This
replaces the old stringly-keyed ``ctx`` dict with an explicit, type-checked
contract between the forward orchestration and the attention functions.

The three independent channels the model threads (see REFACTOR.md §8) appear here
as distinct fields: ``node_ids`` (token→node, drives bias gather + dense mask),
``node_ids_flex`` (int32 copy for the flex kernel gather), and the prebuilt
``structural_mask`` / ``block_mask`` (causal order + structure).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class GraphContext:
    node_ids:        torch.Tensor                 # (B, kv_len) token → node index
    num_nodes:       Optional[torch.Tensor]       # (B,) node count per batch item
    cache:           Dict[str, Any]               # per-generation bias cache (shared dict)
    features:        Dict[str, Any]               # spd / laplacian / rwse / rrwp / magnetic
    structural_mask: Optional[torch.Tensor] = None  # (B, 1, q, kv) dense path
    block_mask:      Optional[Any] = None            # flex BlockMask
    node_ids_flex:   Optional[torch.Tensor] = None   # (B, kv_len) int32, flex path
    shared_node_bias: Optional[torch.Tensor] = None  # (B, H, N, N)
    node_start_indices: Optional[torch.Tensor] = None  # (B, N) first-token pos per node (magnetic_content)

    def install_on(self, attn_modules) -> None:
        """Attach this context to each attention module so the registered
        ``gtlm_*`` function can read it. Not cleared afterwards: it must survive
        gradient-checkpointing recompute in backward; the next forward overwrites
        it."""
        for module in attn_modules:
            module._graph_ctx = self

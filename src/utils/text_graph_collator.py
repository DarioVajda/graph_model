import torch
from torch.nn.utils.rnn import pad_sequence

from .text_graph_dataset import TextGraph


def _k_hop_reachability(A: torch.Tensor, K: int) -> torch.Tensor:
    """
    Compute K-hop reachability from a boolean adjacency matrix.

    Returns a (N, N) bool tensor where entry (i, j) is True iff node j is
    reachable from node i in exactly 1 … K directed steps along A.
    The diagonal is NOT set here; callers add it explicitly as needed.

    Uses repeated matrix multiplication, so O(K · N^2) — fine for small K and N.
    """
    A_float   = A.float()
    reachable = A.clone()     # 1-hop
    power     = A_float.clone()
    for _ in range(K - 1):
        power     = torch.mm(power, A_float)   # A^(k+1)
        reachable = reachable | (power > 0)
    return reachable


class GraphCollator:
    def __init__(self, tokenizer=None, k_hop: int = 0):
        """
        Args:
            tokenizer: Optional tokenizer (unused internally, kept for callers).
            k_hop:     When > 0, a boolean (N×N) K-hop reachability mask is
                       added to every batch under the key 'k_hop_mask'.
                       When 0 (default) the collator behaves exactly as before.
        """
        self.tokenizer = tokenizer
        self.k_hop     = k_hop

    def __call__(self, batch: list[TextGraph]):
        """
        Collates a list of TextGraph dictionaries into a batch.
        """
        
        sizes = torch.tensor([item['num_nodes'] for item in batch], dtype=torch.long)
        prompt_nodes = torch.tensor([item['prompt_node'] for item in batch], dtype=torch.long)
        # edges = [torch.tensor(item['edges'], dtype=torch.long) for item in batch]
        
        input_ids = [
            [torch.tensor(ids, dtype=torch.long) for ids in item['input_ids']]
            for item in batch
        ]

        labels = [ item['labels'] for item in batch ] if "labels" in batch[0] else None

        batch_size = len(batch)
        max_num_nodes = max(sizes).item()

        # initialise laplacian_coordinates
        spectral_dim = batch[0]['laplacian_coordinates'].shape[1] if "laplacian_coordinates" in batch[0] else 0
        laplacian_coordinates = torch.zeros(batch_size, max_num_nodes, spectral_dim, dtype=torch.float)

        # initialise shortest_path_dists
        shortest_path_dists = torch.full((batch_size, max_num_nodes, max_num_nodes), max_num_nodes, dtype=torch.long)

        # initialise rwse
        rwse_dim = batch[0]['rwse'].shape[1] if "rwse" in batch[0] else 0
        rwse = torch.zeros(batch_size, max_num_nodes, rwse_dim, dtype=torch.float)

        # initialise rrwp
        max_rw_steps = batch[0]['rrwp'].shape[2] if "rrwp" in batch[0] else 0
        rrwp = torch.zeros(batch_size, max_num_nodes, max_num_nodes, max_rw_steps, dtype=torch.float)

        # initialise magnetic laplacian
        magnetic_V = torch.zeros(batch_size, max_num_nodes, max_num_nodes, 2, dtype=torch.float) # 2 for real and imaginary parts
        magnetic_lambdas = torch.zeros(batch_size, max_num_nodes, dtype=torch.float)

        for i, item in enumerate(batch):
            num_nodes = item['num_nodes']
            if "laplacian_coordinates" in item:
                laplacian_coordinates[i, :num_nodes, :] = item['laplacian_coordinates'].detach().clone()
            if "shortest_path_dists" in item:
                shortest_path_dists[i, :num_nodes, :num_nodes] = item['shortest_path_dists'].detach().clone()
            if "rwse" in item:
                rwse[i, :num_nodes, :] = item['rwse'].detach().clone()
            if "rrwp" in item:
                rrwp[i, :num_nodes, :num_nodes, :] = item['rrwp'].detach().clone()
            if "magnetic_V" in item and "magnetic_lambdas" in item:
                magnetic_V[i, :num_nodes, :num_nodes, :] = item['magnetic_V'].detach().clone()
                magnetic_lambdas[i, :num_nodes] = item['magnetic_lambdas'].detach().clone()

        batch_dict = {
            'num_nodes': sizes,
            'prompt_node': prompt_nodes,
            # 'edges': edges,
            'input_ids': input_ids,
            'labels': labels,
            'laplacian_coordinates': laplacian_coordinates,
            'shortest_path_dists': shortest_path_dists,
            'rwse': rwse,
            'rrwp': rrwp,
            'magnetic': (magnetic_V, magnetic_lambdas) if torch.any(magnetic_V) or torch.any(magnetic_lambdas) else None,
        }

        if self.k_hop > 0:
            batch_dict['k_hop_mask'] = self._build_k_hop_masks(batch, max_num_nodes)

        return batch_dict

    # ── K-hop mask helpers ────────────────────────────────────────────────────

    def _build_k_hop_masks(self, batch: list[TextGraph], max_num_nodes: int) -> torch.Tensor:
        """Compute and pad K-hop masks for every graph in the batch."""
        batch_size = len(batch)
        masks = torch.zeros(batch_size, max_num_nodes, max_num_nodes, dtype=torch.bool)
        for i, item in enumerate(batch):
            N = item['num_nodes']
            mask = self._single_k_hop_mask(item['edges'], N, item['prompt_node'])
            masks[i, :N, :N] = mask
        return masks

    def _single_k_hop_mask(self, edges, N: int, prompt_node: int) -> torch.Tensor:
        """
        Compute the (N, N) K-hop boolean mask for one graph.

        Two separate reachability computations are performed:
        - Prefix subgraph (prompt node's edges removed): used for all
          prefix-node ↔ prefix-node pairs so that the prompt cannot act
          as a structural bridge between graph nodes.
        - Full graph: used for the prompt node's own row and column so that
          it can attend only to nodes within K hops of itself via its actual
          edges.

        The diagonal is always True (every node can attend to itself).
        """
        p = prompt_node

        # Build full symmetric (undirected) adjacency
        A_full = torch.zeros(N, N, dtype=torch.bool)
        for u, v in edges:
            if 0 <= u < N and 0 <= v < N:
                A_full[u, v] = True
                A_full[v, u] = True

        # Prefix subgraph: cut out the prompt node as a potential relay
        A_prefix = A_full.clone()
        A_prefix[p, :] = False
        A_prefix[:, p] = False

        R_prefix = _k_hop_reachability(A_prefix, self.k_hop)
        R_full   = _k_hop_reachability(A_full,   self.k_hop)

        # Combine: prefix↔prefix uses R_prefix; prompt row/col uses R_full
        mask      = R_prefix.clone()
        mask[p, :] = R_full[p, :]
        mask[:, p] = R_full[:, p]

        mask.fill_diagonal_(True)   # self-attention always allowed
        return mask
import torch
from torch.nn.utils.rnn import pad_sequence

from .text_graph_dataset import TextGraph

class GraphCollator:
    def __init__(self, tokenizer=None, padding_type="right"):
        self.tokenizer = tokenizer
        self.padding_type = padding_type

    def __call__(self, batch: list[TextGraph]):
        """
        Collates a list of TextGraph dictionaries into a batch.
        """

        sizes = torch.tensor([item['num_nodes'] for item in batch], dtype=torch.long)
        texts = [item['text'] for item in batch] # texts will be kept as they are
        prompt_nodes = torch.tensor([item['prompt_node'] for item in batch], dtype=torch.long)
        edges = [torch.tensor(item['edges'], dtype=torch.long) for item in batch]
        
        if "input_ids" in batch[0]:
            input_ids = [
                [torch.tensor(ids, dtype=torch.long) for ids in item['input_ids']]
                for item in batch
            ]
        else:
            input_ids = None

        labels = [ item['labels'] for item in batch ] if "labels" in batch[0] else None

        # spectral_coords = [ item['spectral_coords'] for item in batch ] if "spectral_coords" in batch[0] else None
        # shortest_path_dists = [ item['shortest_path_dists'] for item in batch ] if "shortest_path_dists" in batch[0] else None
        # rwse = [ item['rwse'] for item in batch ] if "rwse" in batch[0] else None
        # rrwp = [ item['rrwp'] for item in batch ] if "rrwp" in batch[0] else None

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

        return {
            'num_nodes': sizes,
            'text': texts,
            'prompt_node': prompt_nodes,
            'edges': edges,
            'input_ids': input_ids,
            'labels': labels,
            'laplacian_coordinates': laplacian_coordinates,
            'shortest_path_dists': shortest_path_dists,
            'rwse': rwse,
            'rrwp': rrwp,
            'magnetic': (magnetic_V, magnetic_lambdas) if torch.any(magnetic_V) or torch.any(magnetic_lambdas) else None,
        }
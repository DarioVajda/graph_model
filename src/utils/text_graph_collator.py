import torch
from torch.nn.utils.rnn import pad_sequence

from .text_graph_dataset import TextGraph

class GraphCollator:
    def __init__(self, tokenizer=None, ):
        self.tokenizer = tokenizer

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

        spectral_coords = [ item['spectral_coords'] for item in batch ] if "spectral_coords" in batch[0] else None
        shortest_path_dists = [ item['shortest_path_dists'] for item in batch ] if "shortest_path_dists" in batch[0] else None
        rwse = [ item['rwse'] for item in batch ] if "rwse" in batch[0] else None
        rrwp = [ item['rrwp'] for item in batch ] if "rrwp" in batch[0] else None

        return {
            'num_nodes': sizes,
            'text': texts,
            'prompt_node': prompt_nodes,
            'edges': edges,
            'input_ids': input_ids,
            'labels': labels,
            'spectral_coords': spectral_coords,
            'shortest_path_dists': shortest_path_dists,
            'rwse': rwse,
            'rrwp': rrwp
        }
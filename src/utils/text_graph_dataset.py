import os
import shutil
import pickle
import networkx as nx
import torch
from datasets import Dataset as HFDataset, load_from_disk
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any, Union
from tqdm import tqdm

from .spectral_coordinates import get_spectral_coordinates
from .rwse import compute_rwse
from .rrwp import compute_rrwp

# define the item type for type hints (must have 'text', 'num_nodes', 'prompt_node' and 'edges' at minimum and may have 'input_ids', 'spectral_coords', 'shortest_path_dists')
TextGraph = Dict[str, Any]

# -----------------------------------------------------------------------------
# Main Dataset Class
# -----------------------------------------------------------------------------
class TextGraphDataset(Dataset):
    """
    A Dataset class for Text-Attributed Graphs that efficiently stores:
    - Raw Text (Ragged)
    - Token IDs (Ragged)
    - Laplacian Spectral Coordinates (Dense, per graph)
    - Shortest Path Distances (Dense/Flattened, per graph)
    - Random Walk Structural Encoding (Dense, per graph)
    Backend: Hugging Face Datasets (Arrow) + NetworkX (Topology).
    """

    def __init__(self, graphs: List[nx.Graph], _hf_dataset: Optional[HFDataset] = None):
        """
        Args:
            graphs: List of NetworkX graphs. Nodes must have 'text' attribute.
            _hf_dataset: (Internal use only) Used by .load() to bypass re-initialization.
        """
        self.graphs = graphs
        
        if _hf_dataset is not None:
            self._hf_dataset = _hf_dataset
        else:
            self._build_initial_dataset()

    def _build_initial_dataset(self):
        """Converts NetworkX node text attributes to a HF Dataset."""
        print("Initializing dataset storage from graphs...")
        data_dict = {
            "text": [],
            "num_nodes": [],
            "prompt_node": [], # Indicates which node is the 'prompt' (if -1 then all nodes can be a valid prompt node)
        }

        for g in tqdm(self.graphs, desc="Reading Graphs"):
            # Ensure 0..N-1 indexing for safety
            g = nx.convert_node_labels_to_integers(g, ordering="sorted")
            
            # Extract text
            texts = [g.nodes[i].get('text', "") for i in range(g.number_of_nodes())]
            
            data_dict["text"].append(texts)
            data_dict["num_nodes"].append(g.number_of_nodes())
            data_dict["prompt_node"].append(g.graph.get('prompt_node', -1)) # Default to -1 if not specified

        # Create the Arrow Dataset
        self._hf_dataset = HFDataset.from_dict(data_dict)

    def __len__(self):
        return len(self._hf_dataset)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing ONLY the features that currently exist.
        This method does NOT compute any new features. It simply retrieves the existing data for the given index.
         - 'text'                   ---> List of strings (raw text for each node)
         - 'num_nodes'              ---> Integer (number of nodes in the graph)
         - 'prompt_node'            ---> Integer (index of the prompt node, or -1 if not specified)
         - 'edges'                  ---> List of tuples (edges between nodes, retrieved from RAM)
         - 'spectral_coords'        ---> Tensor of shape (num_nodes, spectral_dim)
         - 'shortest_path_dists'    ---> Tensor of shape (num_nodes, num_nodes)
         - 'rwse'                   ---> Tensor of shape (num_nodes, max_rwse_steps)
         - 'rrwp'                   ---> Dictionary of relative random walk probabilities for each node pair (num_nodes, num_nodes, max_rrwp_steps)
         - 'input_ids'              ---> List of num_nodes lists of token ids (tokenized input for each node)
         - 'labels'                 ---> Tensor of labels for the prompt node
        """
        # 1. Get the base item from Arrow (fast load)
        item = self._hf_dataset[idx]
        
        # 2. Add Topology (Edges) from RAM
        g = self.graphs[idx]
        item['edges'] = list(g.edges())
        item['num_nodes'] = g.number_of_nodes()
        item['prompt_node'] = g.graph.get('prompt_node', -1)

        # 3. Post-process specific fields if they exist in the dataset
        
        # Handle Spectral Coordinates (Convert list back to torch tensor)
        if 'spectral_coords' in item and item['spectral_coords'] is not None:
            # Check if it's empty
            if len(item['spectral_coords']) == 0:
                item['spectral_coords'] = None
            else:
                item['spectral_coords'] = torch.tensor(item['spectral_coords'], dtype=torch.float32)
            
        # Handle SPD (Reshape flattened array back to Matrix)
        if 'shortest_path_dists' in item and item['shortest_path_dists'] is not None:
            n = item['num_nodes']
            if len(item['shortest_path_dists']) == 0:
                item['shortest_path_dists'] = None
            else:
                flat_spd = torch.tensor(item['shortest_path_dists'], dtype=torch.int32)
                # Safety check for shape
                if flat_spd.numel() == n * n:
                    item['shortest_path_dists'] = flat_spd.reshape((n, n))
                else:
                    item['shortest_path_dists'] = torch.zeros((n,n))

        # Handle RWSE (Convert list back to torch tensor)
        if 'rwse' in item and item['rwse'] is not None:
            item['rwse'] = torch.tensor(item['rwse'], dtype=torch.float32)

        # Handle RRWP (Convert list back to torch tensor)
        if 'rrwp' in item and item['rrwp'] is not None:
            # RRWP is stored as a list of lists (num_nodes, num_nodes, max_rrwp_steps)
            item['rrwp'] = torch.tensor(item['rrwp'], dtype=torch.float32).reshape((item['num_nodes'], item['num_nodes'], -1))

        # Handle Labels
        if 'labels' in item and item['labels'] is not None:
            item['labels'] = torch.tensor(item['labels'], dtype=torch.long)

        return item

    # ------------------------------------------------------------------------
    # Feature Computation Methods
    # ------------------------------------------------------------------------
    
    def tokenize(self, tokenizer, max_length=512):
        """Tokenizes text and adds the 'input_ids' column."""
        if 'input_ids' in self._hf_dataset.column_names:
            print("Dataset already tokenized. Skipping.")
            return

        print("Tokenizing dataset...")
        
        def _tokenize_batch(examples):
            all_input_ids = []
            
            for graph_texts in examples['text']:
                enc = tokenizer(
                    graph_texts, 
                    padding=False, 
                    truncation=True, 
                    max_length=max_length,
                    add_special_tokens=False,
                )
                all_input_ids.append(enc['input_ids'])
               
            return {"input_ids": all_input_ids}

        self._hf_dataset = self._hf_dataset.map(
            _tokenize_batch, 
            batched=True, 
            batch_size=10, 
            desc="Tokenizing"
        )

    def compute_spectral_coordinates(self, embedding_dim=16):
        """Computes Laplacian Eigenmaps and adds 'spectral_coords' column."""
        print(f"Computing Spectral Coordinates (dim={embedding_dim})...")
        
        coords_list = []
        for g in tqdm(self.graphs, desc="Eigen-decomposition"):
            # Ensure standardized node labels 0..N-1
            g_int = nx.convert_node_labels_to_integers(g, ordering="sorted")
            n = g_int.number_of_nodes()
            
            # Call user function
            coords = get_spectral_coordinates(g_int, embedding_dim)
            
            # --- FIX: ROBUST CONVERSION LOGIC ---
            # Handle Dictionary {0: [...], 1: [...]} from user function
            if isinstance(coords, dict):
                # Convert dict to sorted list [coords[0], coords[1], ...]
                # Assumes keys match 0..N-1 because we converted g_int
                try:
                    coords = [coords[i] for i in range(n)]
                except KeyError:
                    print(f"Warning: Spectral dict keys mismatch for graph with {n} nodes. Filling zeros.")
                    coords = [[0.0]*embedding_dim for _ in range(n)]
            
            # Handle PyTorch Tensor
            if isinstance(coords, torch.Tensor):
                coords = coords.tolist()
            
            # Handle Numpy Array
            elif hasattr(coords, "tolist"):
                coords = coords.tolist()
            
            # Handle List of Arrays/Tensors (if previous steps produced list of objects)
            if isinstance(coords, list) and n > 0:
                if isinstance(coords[0], torch.Tensor) or hasattr(coords[0], 'tolist'):
                    coords = [c.tolist() if hasattr(c, 'tolist') else c for c in coords]

            coords_list.append(coords)

        # Remove old column if exists
        if "spectral_coords" in self._hf_dataset.column_names:
            self._hf_dataset = self._hf_dataset.remove_columns("spectral_coords")
            
        self._hf_dataset = self._hf_dataset.add_column("spectral_coords", coords_list)

    def compute_shortest_path_distances(self, cutoff=None):
        """Computes APSP and adds 'shortest_path_dists' column (flattened)."""
        print("Computing Shortest Path Distances...")
        
        spd_list = []
        for g in tqdm(self.graphs, desc="Floyd-Warshall / BFS"):
            n = g.number_of_nodes()
            
            # Using int16 to save space (max distance 32,767)
            # 32_767 represents unreachable
            dist_matrix = torch.full((n, n), 32_767, dtype=torch.int16)
            dist_matrix.fill_diagonal_(0)
            
            path_gen = nx.shortest_path_length(g, source=None, target=None)
            for src, targets in path_gen:
                for tgt, dist in targets.items():
                    if cutoff is None or dist <= cutoff:
                        dist_matrix[src, tgt] = dist

            # FLATTEN for storage compatibility with Arrow
            spd_list.append(dist_matrix.flatten().tolist())

        if "shortest_path_dists" in self._hf_dataset.column_names:
            self._hf_dataset = self._hf_dataset.remove_columns("shortest_path_dists")
            
        self._hf_dataset = self._hf_dataset.add_column("shortest_path_dists", spd_list)

    def compute_rwse(self, max_rwse_steps=8):
        """Computes Random Walk Structural Encoding and adds 'rwse' column."""
        print("Computing Random Walk Structural Encoding (RWSE)...")
        rwse_dataset_list = []
        for g in tqdm(self.graphs, desc=f"Computing RWSE (max steps: {max_rwse_steps})"):
            rwse_dict = compute_rwse(g, max_rwse_steps)
            rwse_list = [ rwse_dict[i] for i in range(g.number_of_nodes()) ]
            rwse_dataset_list.append(rwse_list)
        if "rwse" in self._hf_dataset.column_names:
            self._hf_dataset = self._hf_dataset.remove_columns("rwse")
        self._hf_dataset = self._hf_dataset.add_column("rwse", rwse_dataset_list)

    def compute_rrwp(self, max_rrwp_steps=8):
        """Computes Relative Random Walk Probabilities (RRWP) and adds 'rrwp' column."""
        print("Computing Relative Random Walk Probabilities (RRWP)...")
        rrwp_dataset_list = []
        for g in tqdm(self.graphs, desc=f"Computing RRWP (max steps: {max_rrwp_steps})"):
            rrwp_dict = compute_rrwp(g, max_distance=max_rrwp_steps)
            rrwp_list = [ rrwp_dict[(i,j)] for i in range(g.number_of_nodes()) for j in range(g.number_of_nodes()) ]
            rrwp_dataset_list.append(rrwp_list)
        if "rrwp" in self._hf_dataset.column_names:
            self._hf_dataset = self._hf_dataset.remove_columns("rrwp")
        self._hf_dataset = self._hf_dataset.add_column("rrwp", rrwp_dataset_list)

    def compute_labels(self, get_graph_labels):
        """Computes labels for each graph using the provided function and adds 'labels' column."""
        print("Computing Labels...")
        
        labels_list = []
        for i in tqdm(range(len(self)), desc="Generating Labels"):
            item = self[i]
            
            if 'input_ids' not in item:
                raise ValueError("Dataset must be tokenized before computing labels.")
            
            prompt_node = item['prompt_node']
            if prompt_node == -1:
                raise ValueError(f"Graph at index {i} does not have a valid prompt node. Cannot compute labels.")
                
            # Execute the user-provided mapping function
            label = get_graph_labels(item)
            
            # Validate shape matches the prompt node's token_ids
            expected_length = len(item['input_ids'][prompt_node])
            if len(label) != expected_length:
                raise ValueError(f"Label shape mismatch at index {i}. Expected length {expected_length}, got {len(label)}.")
                
            # Convert to list for efficient Hugging Face Arrow storage
            if isinstance(label, torch.Tensor):
                label = label.tolist()
            elif hasattr(label, "tolist"):
                label = label.tolist()
                
            labels_list.append(label)

        if "labels" in self._hf_dataset.column_names:
            self._hf_dataset = self._hf_dataset.remove_columns("labels")
            
        self._hf_dataset = self._hf_dataset.add_column("labels", labels_list)

    # ------------------------------------------------------------------------
    # Saving and Loading
    # ------------------------------------------------------------------------

    @classmethod
    def gtds_path(cls, base_path: str) -> str:
        """Ensures the path ends with .gtds for consistency. (gtds = Graph Text Dataset)"""
        if base_path.endswith(".gtds") or base_path.endswith(".gtds/"):
            return base_path
        else:
            return base_path + ".gtds"

    def save(self, path: str):
        """
        Saves the dataset to disk. The file structure will be:
        path/
            features/       <-- Hugging Face Dataset (Arrow format)
            graphs.pkl      <-- List of NetworkX graphs (raw topology in RAM)
        """
        path = self.gtds_path(path)        

        if os.path.exists(path):
            print(f"Warning: Directory {path} exists. Overwriting...")
            shutil.rmtree(path)
        
        os.makedirs(path)
        print(f"Saving features to {path}/features...")
        self._hf_dataset.save_to_disk(os.path.join(path, "features"))
        
        print(f"Saving graphs to {path}/graphs.pkl...")
        with open(os.path.join(path, "graphs.pkl"), "wb") as f:
            pickle.dump(self.graphs, f)
        print("Save complete.")

    @classmethod
    def load(cls, path: str) -> 'TextGraphDataset':
        """Loads from disk."""
        path = cls.gtds_path(path)

        features_path = os.path.join(path, "features")
        graphs_path = os.path.join(path, "graphs.pkl")
        
        if not os.path.exists(features_path) or not os.path.exists(graphs_path):
            raise FileNotFoundError(f"Could not find valid dataset at {path}")
            
        print(f"Loading graphs from {graphs_path}...")
        with open(graphs_path, "rb") as f:
            graphs = pickle.load(f)
            
        print(f"Loading features from {features_path}...")
        hf_dataset = load_from_disk(features_path)
        return cls(graphs=graphs, _hf_dataset=hf_dataset)

def generate_text_graph_example(dataset_size=3, base_num_nodes=5, calc_attributes=False, tokenizer=None, spec_emb_dim=4, max_rwse_steps=4, max_rrwp_steps=6) -> TextGraphDataset:
    graphs = []
    for i in range(dataset_size):
        # Uses barabasi_albert_graph (safe for all nx versions)
        g = nx.barabasi_albert_graph(n=base_num_nodes + i, m=1, seed=42)
        nx.set_node_attributes(g, {n: f"Node {n} in graph {i}{' !'*n}" for n in g.nodes()}, "text")
        g.graph['prompt_node'] = i  # Example graph-level attribute
        graphs.append(g)

    ds = TextGraphDataset(graphs)
    if calc_attributes:
        ds.compute_spectral_coordinates(embedding_dim=spec_emb_dim)
        ds.compute_shortest_path_distances()
        ds.compute_rwse(max_rwse_steps=max_rwse_steps)
        ds.compute_rrwp(max_rrwp_steps=max_rrwp_steps)
        if tokenizer is not None:
            ds.tokenize(tokenizer)
    return ds

def prepare_example_labels(graph_dataset):
    labels = []
    for i in range(len(graph_dataset)):
        item = graph_dataset[i]
        prompt_node = item['prompt_node']
        if prompt_node == -1:
            raise ValueError(f"Graph at index {i} does not have a valid prompt node. Cannot prepare labels.")
        else:
            prompt_tokens = torch.tensor(item['input_ids'][prompt_node], dtype=torch.long)
            prompt_tokens[:len(prompt_tokens) // 2] = -100  # Mask out the first half for loss
            labels.append(prompt_tokens)
    return labels
            
# -----------------------------------------------------------------------------
# Demonstration
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("--- 1. Creating Mock Data ---")
    ds = generate_text_graph_example(dataset_size=3, base_num_nodes=5)
    
    print("\n--- 2. Processing (Calculating Features) ---")
    ds.compute_spectral_coordinates(embedding_dim=4)
    ds.compute_shortest_path_distances()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds.tokenize(tokenizer)
    
    # Check item 0
    item = ds[0]
    print(f"Keys present: {item.keys()}")
    print(f"Prompt Node: {item['prompt_node']}")
    print(f"Text: {item['text']}")
    print(f"Edges: {item['edges']}")
    print(f"Spectral Shape: {item['spectral_coords'].shape}")
    print(f"SPD Shape: {item['shortest_path_dists'].shape}")
    print(f"Input IDs: {item['input_ids']}")
    print(f"Type of Input IDs: {type(item['input_ids'])}; Type of input_ids[0]: {type(item['input_ids'][0])}; type of input_ids[0][0]: {type(item['input_ids'][0][0])}")

    # Test the new compute_labels method
    print("\n--- Testing compute_labels ---")
    def mock_get_graph_labels(example):
        prompt_tokens = torch.tensor(example['input_ids'][example['prompt_node']], dtype=torch.long)
        prompt_tokens[:len(prompt_tokens) // 2] = -100
        return prompt_tokens
        
    ds.compute_labels(mock_get_graph_labels)
    print(f"Labels Shape for item 0: {ds[0]['labels'].shape}")

    # stop program execution to avoid saving files in the environment, remove this to test saving/loading
    exit()

    print("\n--- 3. Saving to Disk ---")
    save_path = "./my_text_graph_dataset"
    ds.save(save_path)

    print("\n--- 4. Loading from Disk ---")
    ds_loaded = TextGraphDataset.load(save_path)

    print("\n--- Success ---")
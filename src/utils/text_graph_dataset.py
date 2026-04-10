import os
import shutil
import pickle
import networkx as nx
import torch
from datasets import Dataset as HFDataset, load_from_disk, concatenate_datasets
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any, Union
from tqdm import tqdm
import random
import json
import numpy as np

from .laplacian import get_laplacian_coordinates
from .rwse import compute_rwse
from .rrwp import compute_rrwp
from .magnetic_lap import get_magnetic_laplacian_coords

# define the item type for type hints (must have 'text', 'num_nodes', 'prompt_node' and 'edges' at minimum and may have 'input_ids', 'laplacian_coordinates', 'shortest_path_dists')
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

    def __init__(self, graphs: List[nx.Graph], _hf_dataset: Optional[HFDataset] = None, dataset_label: Optional[str] = None, per_graph_versions=1):
        """
        Args:
            graphs: List of NetworkX graphs. Nodes must have 'text' attribute. Should have N x per_graph_versions graphs structured like this: [g1_v1,... g1_vk, g2_v1,... g2_vk, ...] where k=per_graph_versions. The prompt node index should be stored in graph.graph['prompt_node']
            _hf_dataset: (Internal use only) Used by .load() to bypass re-initialization.
            dataset_label: Optional label for the dataset.
            per_graph_versions: Number of versions of each graph, item i will return one of the versions of graph i (used for data augmentation, default=1 means no augmentation)
        """
        self.per_graph_versions = per_graph_versions

        if _hf_dataset is not None:
            self.graphs = graphs
            self._hf_dataset = _hf_dataset
            if dataset_label is not None:
                self.assign_label(dataset_label)
        else:
            self.graphs = []
            dataset_name = dataset_label if dataset_label is not None else f"ds_{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=6))}"

            for g in graphs:
                mapping = {old_label: new_int for new_int, old_label in enumerate(g.nodes())}
                g_int = nx.relabel_nodes(g, mapping, copy=True)
                nx.set_node_attributes(g_int, {new_int: old for old, new_int in mapping.items()}, "original_id")
                
                old_prompt = g_int.graph.get('prompt_node', -1)
                if old_prompt != -1 and old_prompt in mapping:
                    g_int.graph['prompt_node'] = mapping[old_prompt]
                    
                self.graphs.append(g_int)
            
            self._build_initial_dataset(dataset_name)

    def _build_initial_dataset(self, dataset_label):
        """
        Converts NetworkX node text attributes to a HF Dataset and saves a list of dataset labels to each item to preserve the item origin when merging datasets.
        """
        print("Initializing dataset storage from graphs...")
        data_dict = {
            "text": [],
            "num_nodes": [],
            "prompt_node": [], # Indicates which node is the 'prompt' (if -1 then all nodes can be a valid prompt node)
            "ds_label": [] # Label for the dataset to preserve item origin when merging datasets
        }

        for g in tqdm(self.graphs, desc="Reading Graphs"):
            # Extract text
            texts = [g.nodes[i].get('text', "") for i in range(g.number_of_nodes())]
            
            data_dict["text"].append(texts)
            data_dict["num_nodes"].append(g.number_of_nodes())
            data_dict["prompt_node"].append(g.graph.get('prompt_node', -1)) # Default to -1 if not specified
            data_dict["ds_label"].append(dataset_label) # Add the dataset label

        # Create the Arrow Dataset
        self._hf_dataset = HFDataset.from_dict(data_dict)

    def __len__(self):
        if len(self._hf_dataset) % self.per_graph_versions != 0:
            raise ValueError(f"Dataset length {len(self._hf_dataset)} is not divisible by per_graph_versions {self.per_graph_versions}. Check your dataset construction.")
        return len(self._hf_dataset) // self.per_graph_versions

    def __getitem__(self, idx):
        """
        Returns a dictionary containing ONLY the features that currently exist.
        This method does NOT compute any new features. It simply retrieves the existing data for the given index.
         - 'text'                       ---> List of strings (raw text for each node)
         - 'num_nodes'                  ---> Integer (number of nodes in the graph)
         - 'prompt_node'                ---> Integer (index of the prompt node, or -1 if not specified)
         - 'edges'                      ---> List of tuples (edges between nodes, retrieved from RAM)
         - 'laplacian_coordinates'      ---> Tensor of shape (num_nodes, spectral_dim)
         - 'shortest_path_dists'        ---> Tensor of shape (num_nodes, num_nodes)
         - 'rwse'                       ---> Tensor of shape (num_nodes, max_rwse_steps)
         - 'rrwp'                       ---> Tensor of shape (num_nodes, num_nodes, max_rrwp_steps)
         - 'magnetic_V'                 ---> Tensor of shape (num_nodes, num_nodes, 2) containing real and imaginary parts of the magnetic eigenvectors
         - 'magnetic_lambdas'           ---> Tensor of shape (num_nodes) containing the magnetic eigenvalues (which are real-valued, as the magnetic Laplacian is Hermitian matrix)
         - 'input_ids'                  ---> List of num_nodes lists of token ids (tokenized input for each node)
         - 'labels'                     ---> Tensor of labels for the prompt node
        """
        # 0. Calculate the actual index in the underlying dataset considering per_graph_versions
        version = random.randint(0, self.per_graph_versions - 1)
        idx = idx * self.per_graph_versions + version # eg. if idx=52, per_graph_versions=4 => new idx can be 208, 209, 210, or 211 (randomly selected) which correspond to different versions of the same graph

        # 1. Get the base item from Arrow (fast load)
        item = self._hf_dataset[idx]
        
        # 2. Add Topology (Edges) from RAM
        g = self.graphs[idx]
        item['edges'] = list(g.edges())
        item['num_nodes'] = g.number_of_nodes()
        item['prompt_node'] = g.graph.get('prompt_node', -1)

        # 3. Retrieve the node mapping to original IDs (if needed for debugging or future use)
        original_id_mapping = {g.nodes[i]['original_id']: i for i in range(g.number_of_nodes())}
        item['original_ids'] = original_id_mapping

        # 4. Post-process specific fields if they exist in the dataset
        
        # Handle Laplacian Coordinates (Convert list back to torch tensor)
        if 'laplacian_coordinates' in item and item['laplacian_coordinates'] is not None:
            # Check if it's empty
            if len(item['laplacian_coordinates']) == 0:
                item['laplacian_coordinates'] = None
            else:
                item['laplacian_coordinates'] = torch.tensor(item['laplacian_coordinates'], dtype=torch.float32)

        # Handle Magnetic Spectral Coordinates (Convert list back to torch tensor)
        if 'magnetic_V' in item and item['magnetic_V'] is not None and 'magnetic_lambdas' in item and item['magnetic_lambdas'] is not None:
            item['magnetic_V'] = torch.tensor(item['magnetic_V'], dtype=torch.float32).reshape((item['num_nodes'], item['num_nodes'], 2)) # reshape back to (num_nodes, num_nodes, 2)
            item['magnetic_lambdas'] = torch.tensor(item['magnetic_lambdas'], dtype=torch.float32).reshape((item['num_nodes'],)) # reshape back to (num_nodes,)
            
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

    def __add__(self, other: 'TextGraphDataset') -> 'TextGraphDataset':
        """Allows merging two TextGraphDatasets using the '+' operator."""
        if not isinstance(other, TextGraphDataset):
            return NotImplemented

        if self.per_graph_versions != other.per_graph_versions:
            raise ValueError(f"Cannot merge datasets with different per_graph_versions. Left: {self.per_graph_versions}, Right: {other.per_graph_versions}")
            
        # Safety check: Ensure both datasets have computed the same features
        if set(self._hf_dataset.column_names) != set(other._hf_dataset.column_names):
            raise ValueError(
                f"Cannot merge datasets with different feature columns.\n"
                f"Left dataset columns: {self._hf_dataset.column_names}\n"
                f"Right dataset columns: {other._hf_dataset.column_names}"
            )

        # 1. Merge the raw NetworkX graphs (standard Python list concatenation)
        merged_graphs = self.graphs + other.graphs
        
        # 2. Merge the Hugging Face arrow tables
        merged_hf = concatenate_datasets([self._hf_dataset, other._hf_dataset])
        
        # 3. Return a new instance using your existing internal constructor
        return TextGraphDataset(graphs=merged_graphs, _hf_dataset=merged_hf, per_graph_versions=self.per_graph_versions)

    def assign_label(self, label: str):
        """Assigns a label to the entire dataset."""
        if "ds_label" in self._hf_dataset.column_names:
            self._hf_dataset = self._hf_dataset.remove_columns("ds_label")
        self._hf_dataset = self._hf_dataset.add_column("ds_label", [label] * len(self._hf_dataset))

    # ------------------------------------------------------------------------
    #region Feature Computation Methods
    # ------------------------------------------------------------------------
    
    def tokenize(self, tokenizer, max_length=512, add_eos=False):
        """
        Tokenizes text and adds the 'input_ids' column.
        If add_eos is True, append the tokenizer's EOS token to the prompt node's text after tokenization.
        """
        if 'input_ids' in self._hf_dataset.column_names:
            print("Warning: Dataset already tokenized. Retokenization will overwrite the existing 'input_ids' column.")
        
        # Safety check: ensure the tokenizer actually has an EOS token defined
        if add_eos and tokenizer.eos_token_id is None:
            raise ValueError("add_eos is True, but the provided tokenizer does not have an eos_token_id.")
        
        def _tokenize_single(example):
            # example['text'] is a list of strings corresponding to the nodes in ONE graph
            enc = tokenizer(
                example['text'], 
                padding=False, 
                truncation=True, 
                max_length=max_length,
                add_special_tokens=False,
            )
            
            # Extract the token IDs for this graph
            graph_input_ids = enc['input_ids']
            
            if add_eos:
                prompt_node = example.get('prompt_node', None)
                
                # Check if a valid prompt node exists for this specific graph
                if prompt_node is not None:
                    # Append the EOS token ID to that specific node's sequence
                    graph_input_ids[prompt_node].append(tokenizer.eos_token_id)
                    
            return {"input_ids": graph_input_ids}

        # Set batched=False to process exactly one graph at a time
        self._hf_dataset = self._hf_dataset.map(
            _tokenize_single, 
            batched=False, 
            desc="Tokenizing"
        )

    def compute_laplacian_coordinates(self, embedding_dim=16):
        """Computes Laplacian Eigenmaps and adds 'laplacian_coordinates' column."""
        coords_list = []
        for g in tqdm(self.graphs, desc="Eigen-decomposition"):
            # Ensure standardized node labels 0..N-1
            g_int = g
            n = g_int.number_of_nodes()
            
            # Call user function
            coords = get_laplacian_coordinates(g_int, embedding_dim)
            
            # --- FIX: ROBUST CONVERSION LOGIC ---
            # Handle Dictionary {0: [...], 1: [...]} from user function
            if isinstance(coords, dict):
                # Convert dict to sorted list [coords[0], coords[1], ...]
                # Assumes keys match 0..N-1 because we converted g_int
                try:
                    coords = [coords[i] for i in range(n)]
                except KeyError:
                    print(f"Warning: Laplacian dict keys mismatch for graph with {n} nodes. Filling zeros.")
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
        if "laplacian_coordinates" in self._hf_dataset.column_names:
            self._hf_dataset = self._hf_dataset.remove_columns("laplacian_coordinates")
            
        self._hf_dataset = self._hf_dataset.add_column("laplacian_coordinates", coords_list)

    def compute_shortest_path_distances_slow(self, cutoff=None):
        """Computes APSP and adds 'shortest_path_dists' column (flattened)."""       
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

    def compute_shortest_path_distances(self, cutoff=None):
        # 1. Setup Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Computing Shortest Paths on: {device}")

        # 2. Define the Mapping Function
        # We use this inside .map() to handle chunks of data efficiently
        def batch_apsp(batch, indices):
            batch_results = []
            
            for idx in indices:
                # Get the specific graph object for this row
                g = self.graphs[idx]
                n = g.number_of_nodes()
                
                # Convert graph to adjacency on GPU
                # We use float32 for the math then cast back to int16 to save memory
                adj = torch.from_numpy(nx.to_numpy_array(g)).to(device).to(torch.float32)
                
                # Initialize distance matrix
                # 32767 is our 'infinity' for int16
                dist = torch.full((n, n), 32767, device=device, dtype=torch.float32)
                dist[adj > 0] = 1.0
                dist.fill_diagonal_(0)

                # Vectorized Floyd-Warshall: O(N^3) but fully parallel in CUDA
                for k in range(n):
                    # dist = min(dist, dist[:, k] + dist[k, :])
                    # Note: .unsqueeze handles the broadcasting for the addition
                    dist = torch.min(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))

                # Apply cutoff and handle unreachable nodes
                if cutoff is not None:
                    dist[dist > cutoff] = 32767
                
                # Flatten and cast to int16 before moving to CPU
                # Arrow handles NumPy arrays much faster than Python lists
                flat_dist = dist.to(torch.int16).flatten().cpu().numpy()
                batch_results.append(flat_dist)

            return {"shortest_path_dists": batch_results}

        # 3. Clean up existing column if necessary
        if "shortest_path_dists" in self._hf_dataset.column_names:
            self._hf_dataset = self._hf_dataset.remove_columns("shortest_path_dists")

        # 4. Execute the mapping
        # batched=True processes multiple rows at once, reducing overhead
        # with_indices=True allows us to look up the correct graph in self.graphs
        self._hf_dataset = self._hf_dataset.map(
            batch_apsp,
            batched=True,
            batch_size=128, # Adjust based on your GPU VRAM
            with_indices=True,
            desc="Vectorized APSP (Torch)"
        )

    def compute_rwse(self, max_rwse_steps=8):
        """Computes Random Walk Structural Encoding and adds 'rwse' column."""
        rwse_dataset_list = []
        for g in tqdm(self.graphs, desc=f"Computing RWSE (max steps: {max_rwse_steps})"):
            rwse_dict = compute_rwse(g, max_rwse_steps)
            rwse_list = [ rwse_dict[i] for i in range(g.number_of_nodes()) ]
            rwse_dataset_list.append(rwse_list)
        if "rwse" in self._hf_dataset.column_names:
            self._hf_dataset = self._hf_dataset.remove_columns("rwse")
        self._hf_dataset = self._hf_dataset.add_column("rwse", rwse_dataset_list)

    def compute_rrwp(self, max_rrwp_steps=8):
        """
        Optimized RRWP using batch tensor operations.
        """
        # 1. Clean up old column
        if "rrwp" in self._hf_dataset.column_names:
            self._hf_dataset = self._hf_dataset.remove_columns("rrwp")

        # 2. Define the vectorized batching function
        def _compute_batch_vectorized(batch, indices):
            # Retrieve all graph objects for this batch at once
            graph_objects = [self.graphs[i] for i in indices]
            
            # Compute everything in one GPU pass
            # This returns a list of flattened numpy arrays
            rrwp_batch = compute_rrwp(graph_objects, max_distance=max_rrwp_steps)
            
            return {"rrwp": rrwp_batch}

        # 3. Stream data using larger batches to saturate the GPU
        self._hf_dataset = self._hf_dataset.map(
            _compute_batch_vectorized,
            with_indices=True,
            batched=True,
            batch_size=16, # Increased for better GPU utilization
            desc=f"Batched GPU RRWP (steps: {max_rrwp_steps})"
        )

    def compute_magnetic_lap(self, q=0.25):
        """
        High-speed Batched Magnetic Laplacian calculation.
        """
        # 1. Clean up old columns
        cols_to_remove = [c for c in ["magnetic_V", "magnetic_lambdas"] if c in self._hf_dataset.column_names]
        if cols_to_remove:
            self._hf_dataset = self._hf_dataset.remove_columns(cols_to_remove)

        # 2. Optimized Batched Function
        def _compute_batch_fast(batch, indices):
            # Gather all graph objects for the batch
            graph_objects = [self.graphs[idx] for idx in indices]
            
            # Batch process on GPU
            # V_list contains (n, n, 2) arrays, L_list contains (n,) arrays
            V_list, L_list = get_magnetic_laplacian_coords(graph_objects, q=q)
            
            # Flatten for Arrow storage compatibility
            # We use .reshape(-1) which is faster than .flatten() in many cases
            v_batch = [v.reshape(-1).tolist() for v in V_list]
            l_batch = [l.tolist() for l in L_list]
            
            return {"magnetic_V": v_batch, "magnetic_lambdas": l_batch}

        # 3. Stream to dataset
        # We use a smaller batch size for eigh because eigvecs consume O(N^2) memory
        self._hf_dataset = self._hf_dataset.map(
            _compute_batch_fast,
            with_indices=True,
            batched=True,
            batch_size=1, 
            desc=f"GPU Magnetic Spectral Decomposition (q={q})"
        )

    def compute_labels(self, get_graph_labels, num_proc=None):
        """
        Parallelized label computation using multi-processing.
        """
        if 'input_ids' not in self._hf_dataset.column_names:
            raise ValueError("Dataset must be tokenized before computing labels.")

        # 1. Pre-extract prompt node indices.
        # We do this to avoid pickling the entire 'self.graphs' list (NetworkX objects),
        # which would be very slow and memory-intensive when starting worker processes.
        prompt_nodes = [g.graph.get('prompt_node', -1) for g in self.graphs]

        # 2. Define the worker function
        def _map_fn(example, idx):
            # Inject metadata required by your GetGraphLabels class
            example['prompt_node'] = prompt_nodes[idx]
            
            if example['prompt_node'] == -1:
                raise ValueError(f"Graph at index {idx} has no valid prompt node.")
            
            # Execute your existing callable class
            label = get_graph_labels(example)
            
            # Standardize for Hugging Face Arrow storage
            if hasattr(label, "tolist"):
                label = label.tolist()
            return {"labels": label}

        # 3. Run parallelized mapping
        # On your Linux environment, this uses 'fork' which is extremely efficient.
        print(f"Parallelizing label generation across {num_proc or os.cpu_count()} cores...")
        self._hf_dataset = self._hf_dataset.map(
            _map_fn,
            with_indices=True,
            num_proc=num_proc if num_proc is not None else os.cpu_count(),
            desc="Generating Labels (Parallel CPU)"
        )
    #endregion
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    #region Saving and Loading
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
        temp_path = self.gtds_path(path + "_temp")
        path = self.gtds_path(path)        

        # 1. Save to a temporary directory first to avoid memory-mapping conflicts
        os.makedirs(temp_path, exist_ok=True)
        print(f"Saving features to temporary path {temp_path}/features...")
        self._hf_dataset.save_to_disk(
            os.path.join(temp_path, "features"),
            max_shard_size="10GB"
        )
        
        print(f"Saving graphs to {temp_path}/graphs.pkl...")
        with open(os.path.join(temp_path, "graphs.pkl"), "wb") as f:
            pickle.dump(self.graphs, f)

        metadata = { "per_graph_versions": self.per_graph_versions }
        with open(os.path.join(temp_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
            
        # 2. Safely swap the directories
        if os.path.exists(path):
            shutil.rmtree(path)
        os.rename(temp_path, path)
        
        # 3. REFRESH MEMORY MAP: Point the active dataset to the new files
        self._hf_dataset = load_from_disk(os.path.join(path, "features"))
        
        print(f"Save complete to {path}.")

    @classmethod
    def load(cls, path: str) -> 'TextGraphDataset':
        """Loads from disk and ensures legacy compatibility."""
        path = cls.gtds_path(path)

        features_path = os.path.join(path, "features")
        graphs_path = os.path.join(path, "graphs.pkl")
        metadata_path = os.path.join(path, "metadata.json")
        
        if not os.path.exists(features_path) or not os.path.exists(graphs_path):
            raise FileNotFoundError(f"Could not find valid dataset at {path}")
            
        print(f"Loading dataset from {graphs_path}...")

        with open(graphs_path, "rb") as f:
            graphs = pickle.load(f)
            
        hf_dataset = load_from_disk(features_path)

        per_graph_versions = 1
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                per_graph_versions = metadata.get("per_graph_versions", 1)
        else:
            print(f"Legacy dataset detected (no metadata.json). Defaulting per_graph_versions to 1.")
        
        # Support legacy datasets which did not have the 'ds_label' column by injecting a random label
        if "ds_label" not in hf_dataset.column_names:
            fallback_label = f"ds_{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=6))}"
            print(f"Legacy dataset detected at {path}. Injecting random 'ds_label': {fallback_label}")
            hf_dataset = hf_dataset.add_column("ds_label", [fallback_label] * len(hf_dataset))
            
        return cls(graphs=graphs, _hf_dataset=hf_dataset, per_graph_versions=per_graph_versions)
    #endregion
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    #region Utils for printing and representation
    # ------------------------------------------------------------------------
    def __repr__(self) -> str:
        """
        Defines the string representation of the dataset for clean printing.
        """
        if self._hf_dataset is None:
            return "<TextGraphDataset: Uninitialized>"
            
        # Safely extract the dataset label if it exists
        features = self._hf_dataset.column_names
        
        # Calculate dataset statistics
        total_items = len(self)
        total_graphs = len(self.graphs)
        
        # Format the features into a readable string wrapper
        formatted_features = ", ".join(features)
        
        return (
            f"TextGraphDataset(\n"
            f"  Total Items: {total_items}\n"
            f"  Versions per Item: {self.per_graph_versions}\n"
            f"  Total Graphs: {total_graphs}\n"
            f"  Computed Features: [{formatted_features}]\n"
            f")"
        )

    def __str__(self) -> str:
        return self.__repr__()
    #endregion
    # ------------------------------------------------------------------------



def generate_text_graph_example(dataset_size=3, base_num_nodes=5, calc_attributes=False, tokenizer=None, spec_emb_dim=4, max_rwse_steps=4, max_rrwp_steps=6, graph_type="undirected") -> TextGraphDataset:
    graphs = []
    for i in range(dataset_size):
        if graph_type == "undirected":
            # Uses barabasi_albert_graph (safe for all nx versions)
            g = nx.barabasi_albert_graph(n=base_num_nodes + i, m=1, seed=42)
        elif graph_type == "directed":
            g = nx.gnp_random_graph(n=base_num_nodes + i, p=0.3, directed=True, seed=42)
        else:
            raise ValueError(f"Unsupported graph_type: {graph_type}")
        nx.set_node_attributes(g, {n: f"Node {n} in graph {i}{' !'*n}" for n in g.nodes()}, "text")
        g.graph['prompt_node'] = i  # Example graph-level attribute
        graphs.append(g)

    ds = TextGraphDataset(graphs)
    if calc_attributes:
        ds.compute_laplacian_coordinates(embedding_dim=spec_emb_dim)
        ds.compute_magnetic_lap(q=0.25)
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
    ds = generate_text_graph_example(
        graph_type="directed",
        dataset_size=3, 
        base_num_nodes=5, 
        calc_attributes=True, 
        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"), 
        spec_emb_dim=4, 
        max_rwse_steps=4, 
        max_rrwp_steps=6
    )
    
    print("\n--- 2. Processing (Calculating Features) ---")
    
    # Check item 0
    item = ds[0]
    print(f"Keys present: {item.keys()}")
    print(f"Prompt Node: {item['prompt_node']}")
    print(f"Text: {item['text']}")
    print(f"Edges: {item['edges']}")
    print(f"Laplacian Shape: {item['laplacian_coordinates'].shape}")
    print(f"Magnetic V Shape: {item['magnetic_V'].shape}")
    print(f"Magnetic Lambdas Shape: {item['magnetic_lambdas'].shape}")
    for i, emb in enumerate(item['magnetic_V']):
        print(f"Node {i} Magnetic Spectral Coords (real, imag):\n{emb}")
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
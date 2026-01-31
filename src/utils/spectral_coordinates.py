import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

def get_spectral_coordinates(G, m):
    """
    Computes m-dimensional spectral coordinates for nodes in G.
    Handles small graphs (N <= m) via zero-padding and large graphs 
    via a sparse hybrid solver.
    
    Args:
        G (nx.Graph): The input networkx graph.
        m (int): The target number of spectral dimensions (radial dimensions).
        
    Returns:
        dict: Node labels mapped to a numpy array of size (m,).
    """
    N = G.number_of_nodes()
    node_list = list(G.nodes())
    
    # edge case when graph has only one node
    if N <= 1:
        return {node: np.zeros(m) for node in node_list}

    # We skip the first eigenvector (lambda=0), so we can get at most N-1 features.
    k_available = min(m, N - 1)
    
    # 1. normalized laplacian matrix
    L = nx.normalized_laplacian_matrix(G).astype(float)
    
    # 2. compute eigenvalues and eigenvectors
    if (m + 1) >= N:
        eigenvalues, eigenvectors = eigh(L.toarray())
        available_features = eigenvectors[:, 1:k_available + 1]
    else:
        eigenvalues, eigenvectors = eigsh(L, k=k_available + 1, which='SM')
        available_features = eigenvectors[:, 1:]

    # 3. padding logic
    if k_available < m:
        padding = np.zeros((N, m - k_available))
        final_features = np.hstack([available_features, padding])
    else:
        final_features = available_features

    return {node: final_features[i] for i, node in enumerate(node_list)}

# --- Test ---
if __name__ == "__main__":
    test_G = nx.Graph()
    test_G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('C', 'D')])
    
    m_dims = 5
    try:
        coords = get_spectral_coordinates(test_G, m_dims)
        print(f"Successfully computed coordinates for {test_G.number_of_nodes()} nodes.")
        for node, vector in coords.items():
            print(f"Node {node}: {vector}")
    except Exception as e:
        print(f"Error: {e}")
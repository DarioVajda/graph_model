import numpy as np
import networkx as nx
from scipy.linalg import eigh

def get_magnetic_laplacian_coords(G, m=16, q=0.25):
    """
    Computes the magnetic Laplacian spectral coordinates for a given graph G.
    
    Args:
        G (networkx.Graph): Input graph.
        m (int): Number of spectral coordinates to compute.
        q (float): Parameter for calculating the rotation angles by this formula: theta_ij = 2pi * q * (a_ij  a_ji).
    Returns:
        dict: A dictionary mapping node to its magnetic spectral coordinates (tensor of shape (m, 2) for real and imaginary parts).
    """
    A = nx.adjacency_matrix(G).astype(float)

    # compute undirected adjacency matrix
    As = 1/2 * (A + A.T)

    # compute the degree matrix
    Ds = np.diag(As.sum(axis=1))

    # compute the angles for the magnetic Laplacian rotation
    thetas = 2 * np.pi * q * (A - A.T).todense()

    # compute the normalised Magnetic Laplacian (Hermitian) matrix == I - (Ds^-1/2 * As * Ds^-1/2) .* exp(i * thetas)
    Ds_sqrt_inv = np.diag(1/np.sqrt(Ds.diagonal()))
    L_N = np.eye(Ds.shape[0]) - Ds_sqrt_inv @ As @ Ds_sqrt_inv * np.exp(1j * thetas)

    # compute Eigenvalues and Eigenvectors
    k = min(m, L_N.shape[0])
    evals, evecs = eigh(L_N, subset_by_index=[0, k-1])
    
    # format the output mapping with zero-padding
    coords = {}
    nodes = list(G.nodes())
    
    for i, node in enumerate(nodes):
        node_complex_coords = evecs[i, :]  # Shape: (k,)
        
        node_tensor = np.zeros((m, 2))
        
        node_tensor[:k, 0] = node_complex_coords.real
        node_tensor[:k, 1] = node_complex_coords.imag
        
        coords[node] = node_tensor
        
    return coords

if __name__ == "__main__":    
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 1), (2, 3), (3, 0)])
    magnetic_coords = get_magnetic_laplacian_coords(G, m=8)
    for node, coords in magnetic_coords.items():
        print(f"Node {node}:")
        print(coords)
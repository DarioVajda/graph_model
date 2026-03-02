import numpy as np
import networkx as nx
from scipy.linalg import eigh

def get_magnetic_laplacian_coords(G, q=0.25):
    """
    Computes the magnetic Laplacian spectral coordinates for a given directed graph G and parameter q.
    The following equation holds: L_N = V @ diag(lambdas) @ conj(V.T)
    
    Args:
        G (networkx.Graph): Input graph.
        q (float): Parameter in interval [0, 0.25] for calculating the rotation angles by this formula: theta_ij = 2pi * q * (a_ij  a_ji).
    Returns:
        tuple (V, lambdas):
            V (numpy.ndarray): (n, n, 2) Full matrix of the eigenvectors of the magnetic Laplacian with real and imaginary parts separated in the last dimension.
            lambdas (numpy.ndarray): (n,) Array of eigenvalues of the magnetic Laplacian.
    """
    A = nx.adjacency_matrix(G).astype(float)

    # compute undirected adjacency matrix
    As = 1/2 * (A + A.T)

    # compute the degree matrix
    Ds = np.diag(As.sum(axis=1))

    # compute the angles for the magnetic Laplacian rotation
    thetas = 2 * np.pi * q * (A - A.T).todense()

    # compute the normalised Magnetic Laplacian (Hermitian) matrix == I - (Ds^-1/2 * As * Ds^-1/2) .* exp(i * thetas)
    diag_Ds = Ds.diagonal()
    safe_inv_sqrt = np.divide(1.0, np.sqrt(diag_Ds), out=np.zeros_like(diag_Ds, dtype=float), where=diag_Ds!=0)
    Ds_sqrt_inv = np.diag(safe_inv_sqrt)
    L_N = np.eye(Ds.shape[0]) - Ds_sqrt_inv @ As @ Ds_sqrt_inv * np.exp(1j * thetas)

    # compute Eigenvalues and Eigenvectors of the magnetic Laplacian
    eigvals, eigvecs = eigh(L_N)
    
    # separate real and imaginary parts of the eigenvectors
    V = np.zeros((eigvecs.shape[0], eigvecs.shape[1], 2))
    V[:,:,0] = np.real(eigvecs)
    V[:,:,1] = np.imag(eigvecs)

    return V, eigvals


if __name__ == "__main__":    
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 1), (2, 3), (3, 0)])
    V, lambdas = get_magnetic_laplacian_coords(G, q=0.25)
    print("Magnetic Laplacian Eigenvalues:", lambdas)
    print("Magnetic Laplacian Eigenvectors:\n", V)
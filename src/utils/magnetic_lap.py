import numpy as np
import networkx as nx
import torch

def get_magnetic_laplacian_coords(graphs, q=0.25):
    """
    Optimized Magnetic Laplacian spectral coordinates using Batched PyTorch.
    Supports single nx.Graph or list of nx.Graph.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Handle Input Consistency
    is_single = isinstance(graphs, nx.Graph)
    graph_list = [graphs] if is_single else graphs
    num_graphs = len(graph_list)
    node_counts = [g.number_of_nodes() for g in graph_list]
    max_n = max(node_counts)

    # 2. Build Batched Adjacency Tensors
    # A: [Batch, N, N]
    A = torch.zeros((num_graphs, max_n, max_n), device=device, dtype=torch.float32)
    for i, g in enumerate(graph_list):
        adj = nx.to_numpy_array(g)
        A[i, :node_counts[i], :node_counts[i]] = torch.from_numpy(adj)

    # 3. Compute Hermitian Magnetic Laplacian components
    # As: Symmetric adjacency
    As = 0.5 * (A + A.transpose(1, 2))
    
    # Ds: Symmetric Degree Matrix
    ds_diag = As.sum(dim=2)
    # Safe inverse sqrt for normalization
    ds_inv_sqrt = torch.where(ds_diag > 0, 1.0 / torch.sqrt(ds_diag), 0.0)
    
    # Thetas: Rotation angles
    # theta_ij = 2pi * q * (a_ij - a_ji)
    thetas = 2 * np.pi * q * (A - A.transpose(1, 2))
    
    # 4. Construct Normalized Magnetic Laplacian (L_N)
    # L_N = I - (Ds^-1/2 * As * Ds^-1/2) * exp(i * thetas)
    # We use complex tensors for the rotation
    rotation = torch.exp(1j * thetas.to(torch.complex64))
    
    # Apply normalization: Ds_inv_sqrt @ As @ Ds_inv_sqrt
    # This is equivalent to row and column scaling
    normalized_As = ds_inv_sqrt.unsqueeze(2) * As * ds_inv_sqrt.unsqueeze(1)
    
    # L_N Construction
    eye = torch.eye(max_n, device=device).unsqueeze(0)
    L_N = eye.to(torch.complex64) - (normalized_As.to(torch.complex64) * rotation)

    # 5. Batched Eigen-decomposition (The GPU Heavy Lifter)
    # torch.linalg.eigh is optimized for Hermitian matrices
    eigvals, eigvecs = torch.linalg.eigh(L_N)

    # 6. Formatting Outputs
    # Separate real and imaginary for eigenvectors: [B, N, N, 2]
    V_final = torch.stack([eigvecs.real, eigvecs.imag], dim=-1)
    
    # Move to CPU for final formatting
    V_cpu = V_final.cpu().numpy()
    lambdas_cpu = eigvals.real.cpu().numpy() # Eigenvalues of Hermitian are real

    if is_single:
        # Return exact original format: (n, n, 2), (n,)
        n = node_counts[0]
        return V_cpu[0, :n, :n, :], lambdas_cpu[0, :n]
    else:
        # Return lists for batched processing
        v_list = []
        l_list = []
        for b in range(num_graphs):
            n = node_counts[b]
            v_list.append(V_cpu[b, :n, :n, :])
            l_list.append(lambdas_cpu[b, :n])
        return v_list, l_list

if __name__ == "__main__":
    # Test compatibility
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 1), (2, 3), (3, 0)])
    V, lambdas = get_magnetic_laplacian_coords(G, q=0.25)
    print("Eigenvalues shape:", lambdas.shape) # Should be (4,)
    print("Eigenvectors shape:", V.shape)     # Should be (4, 4, 2)

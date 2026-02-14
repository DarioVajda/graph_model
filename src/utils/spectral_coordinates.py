import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

def get_spectral_coordinates_old(G, m, random_sign_flips=False):
    """
    Computes m-dimensional spectral coordinates for nodes in G.
    Handles small graphs (N <= m) via zero-padding and large graphs 
    via a sparse hybrid solver.
    
    Args:
        G (nx.Graph): The input networkx graph.
        m (int): The target number of spectral dimensions (radial dimensions).
        random_sign_flips (bool): If True, randomly flip signs of eigenvectors to address sign ambiguity.
        
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

    # --- SIGN FLIP LOGIC ---
    if random_sign_flips:
        # 1. Generate one sign per dimension (column), not per node (row)
        # signs shape: (1, k_available)
        signs = np.random.choice([-1, 1], size=(1, k_available))
        
        # 2. Multiply the entire matrix by the signs across columns
        # This flips the entire eigenvector for all nodes simultaneously
        available_features = available_features * signs
    # ---------------------------------

    # 3. padding logic
    if k_available < m:
        padding = np.zeros((N, m - k_available))
        final_features = np.hstack([available_features, padding])
    else:
        final_features = available_features

    return {node: final_features[i] for i, node in enumerate(node_list)}

def get_spectral_coordinates(G, m, random_sign_flips=False):
    N = G.number_of_nodes()
    node_list = list(G.nodes())
    
    if N <= 1:
        return {node: np.zeros(m) for node in node_list}

    # Compute up to m+1 eigenvectors to discard the constant 0th.
    num_eigenvectors_to_compute = min(m + 1, N)
    
    # 1. Use Combinatorial Laplacian (Unnormalized) for purer geometry
    L = nx.laplacian_matrix(G).astype(float) 
    
    # 2. Compute Eigenvalues
    if N < 200:
        eigenvalues, eigenvectors = eigh(L.toarray())
    else:
        # 'SM' = Smallest Magnitude. 
        eigenvalues, eigenvectors = eigsh(L, k=num_eigenvectors_to_compute, which='SM', sigma=1e-5)
    
    # 3. SORTING
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 4. Filter features
    available_features = eigenvectors[:, 1 : num_eigenvectors_to_compute]
    
    # 5. Sign Flips
    if random_sign_flips:
        signs = np.random.choice([-1, 1], size=(1, available_features.shape[1]))
        available_features = available_features * signs

    # 6. Padding
    current_dim = available_features.shape[1]
    if current_dim < m:
        padding = np.zeros((N, m - current_dim))
        final_features = np.hstack([available_features, padding])
    else:
        final_features = available_features

    # 7. Scale Up by sqrt(N)
    final_features = final_features * np.sqrt(N)

    return {node: final_features[i] for i, node in enumerate(node_list)}

def fit_transformations_single_ls(x1, x2):
    """
    Given two sets of points x1 and x2 (numpy arrays of shape [num_nodes, m]), fit a linear transformation A*x2+b = x1.

    Args:
        x1 (np.ndarray): set of target points of shape [num_nodes, m]
        x2 (np.ndarray): set of original points of shape [num_nodes, m]
    Returns:
        tuple: (A, b) where A is the linear transformation matrix of shape m x
                m, and b is the translation vector of shape m x 1.
    """
    assert x1.shape == x2.shape, "Input point sets must have the same shape."
    num_nodes, m = x1.shape

    # 1. Augment x2 with a column of ones to handle the translation (b)
    # x2_augmented shape: [num_nodes, m + 1]
    ones = np.ones((num_nodes, 1))
    x2_augmented = np.hstack([x2, ones])

    # 2. Solve the linear least squares problem: x2_augmented @ W = x1
    # W will have shape [m + 1, m]
    # W[:m, :] will be A.T (transpose of transformation matrix)
    # W[m, :]  will be b (translation vector)
    W, residuals, rank, s = np.linalg.lstsq(x2_augmented, x1, rcond=None)

    # 3. Extract A and b
    # Since we solved x2 @ A.T + b = x1, we transpose back for the return
    A = W[:m, :].T
    b = W[m, :]

    return A, b

def fit_transformations_single(x1, x2):
    """
    Fits A*x2 + b = x1 using a robust Procrustes-style approach.
    Prevents dimensional collapse by centering and using SVD.
    """
    assert x1.shape == x2.shape, "Input point sets must have the same shape."
    num_nodes, m = x1.shape

    # 1. Center the points to handle translation (b) separately
    mu1 = x1.mean(axis=0)
    mu2 = x2.mean(axis=0)
    
    x1_centered = x1 - mu1
    x2_centered = x2 - mu2

    # 2. Use SVD to find the best rotation/scaling matrix A
    # This solves the 'Orthogonal Procrustes Problem'
    # It finds the A that minimizes ||A*x2_centered - x1_centered||
    H = x2_centered.T @ x1_centered
    U, S, Vt = np.linalg.svd(H)
    
    # A is the rotation/reflection matrix
    A = (U @ Vt).T

    # 3. Handle Scaling (Optional but recommended for your m=2 vs N case)
    # If the variance is very different, we scale A
    # var1 = np.sum(np.square(x1_centered))
    # var2 = np.sum(np.square(x2_centered))
    # if var2 > 1e-9:
    #     scale = np.sqrt(var1 / var2)
    #     A = A * scale

    # 4. Calculate translation b
    # b = x1_mean - A * x2_mean
    b = mu1 - (A @ mu2)

    return A, b

def fit_transformations(spectral_coords_list):
    """
    Given a list of spectral coordinates lists (one per graph), fit a linear transformation (scaling + translation) to align them such that the previous nodes map to similar coordinates.

    Args:
        spectral_coords_list (list): List of lists of spectral coordinates of shapes [ [1], [2],... [num_graphs] ]

    Returns:
        list: List of num_graphs-1 transformation matrices A_{i->i+1} (i=0,...num_graphs-1) of shape m x m, where A_{i->i+1} maps spectral coordinates from nodes {0,...,i-1} from graphs G_i to G_{i+1}
    """
    transformations = []
    num_graphs = len(spectral_coords_list)

    for i in range(num_graphs - 1):
        print("Fitting transformation from graph", i+1, "to graph", i, end=' ')
        spec_coords_i = np.array(list(spectral_coords_list[i].values()))
        spec_coords_next = np.array(list(spectral_coords_list[i+1].values())[:-1]) # exclude the last node, as it doesn't exist in graph i

        transformation = fit_transformations_single(spec_coords_i, spec_coords_next)
        transformations.append(transformation)

        # test the transformation by applying it to graph i+1's spectral coordinates and computing mse
        A, b = transformation
        transformed_coords = (A @ spec_coords_next.T).T + b
        mse = np.mean((transformed_coords - spec_coords_i)**2)
        print("--> Mean Squared Error after transformation:", mse)
    
    return transformations

# --- Test ---
def plot_spectral_coordinates_progression(m=4):
    # plot the progression graphs and their respective laplacian spectral coordinates as we add nodes one-by-one and save to file "spectral_coords_progression.png"
    import matplotlib.pyplot as plt

    # graphs = [
    #     nx.path_graph(1),
    #     nx.path_graph(2),
    #     nx.complete_graph(3),
    #     nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)]),
    #     nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]),
    #     nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (3, 5), (4, 5)]),
    #     nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)]),
    # ]
    graphs = [
        nx.Graph([(0, 1)]),
        nx.Graph([(0, 1), (1, 2)]),
        nx.Graph([(0, 1), (1, 2), (2, 3)]),
        nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4)]),  # Horizontal line length 5
        # Start adding a vertical arm at the center (Node 2)
        nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)]),
        nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]),
        nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7)]),
        # Vertical arm is now longer than horizontal. Expect flip.
        nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7), (7, 8)]),
    ]
    # graphs = [
    #     nx.Graph([(0, 1)]),
    #     nx.Graph([(0, 1), (1, 2)]),
    #     nx.Graph([(0, 1), (1, 2), (2, 3)]),
    #     nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4)]),
    #     nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
    #     # Node 6 connects to 5 AND 0, closing the loop
    #     nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)]),
    # ]

    # compute spectral coordinates for each graph <-- Perform this twice to see how these features can differ based on sign flips!
    spectral_coords_list = [get_spectral_coordinates(G, m) for G in graphs]


    transformations = fit_transformations(spectral_coords_list)

    # print the transformations
    # print()
    # print("------------------------------------------------------------------------")
    # print("----------------------- ESTIMATED TRANSFORMATIONS -----------------------")
    # print("------------------------------------------------------------------------")
    # for i, (A, b) in enumerate(transformations):
    #     print(f"Transformation from Graph {i+1} to Graph {i+2}:")
    #     print("A:\n", A)
    #     print("b:\n", b)
    #     print()

    # apply transformations sequentially to align all spectral coordinates to the first graph's coordinate system
    spectral_coords_list_transformed = []
    for i in range(len(graphs)):
        print("Applying transformations to graph", i+1)
        if i == 0:
            spectral_coords_list_transformed.append(spectral_coords_list[0])
        else:
            # apply all transformations from graph i down to graph 0
            spec_coords = np.array(list(spectral_coords_list[i].values()))
            for j in range(i-1, -1, -1):
                A, b = transformations[j]
                spec_coords = (A @ spec_coords.T).T + b
                print(f" Applied transformation from graph {i+1} to graph {j+1}, transformed spectral features:\n", spec_coords)
            # convert back to dict
            spectral_coords_list_transformed.append({node: spec_coords[idx] for idx, node in enumerate(graphs[i].nodes())})
        

    # plotting
    fig, axes = plt.subplots(3, len(graphs), figsize=(20, 12))
    for i, G in enumerate(graphs):
        ax_graph = axes[0, i]
        ax_spec1 = axes[1, i]
        ax_spec2 = axes[2, i]

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, ax=ax_graph)
        ax_graph.set_title(f"Graph {i+1}")
        spec_coords1 = spectral_coords_list[i]
        x_coords1 = [spec_coords1[node][0] for node in G.nodes()]
        y_coords1 = [spec_coords1[node][1] for node in G.nodes()]
        ax_spec1.scatter(x_coords1, y_coords1)
        for node in G.nodes():
            ax_spec1.text(spec_coords1[node][0], spec_coords1[node][1], str(node))
        spec_coords2 = spectral_coords_list_transformed[i]
        x_coords2 = [spec_coords2[node][0] for node in G.nodes()]
        y_coords2 = [spec_coords2[node][1] for node in G.nodes()]
        ax_spec2.scatter(x_coords2, y_coords2)
        for node in G.nodes():
            ax_spec2.text(spec_coords2[node][0], spec_coords2[node][1], str(node))
        ax_spec1.set_title(f"Spectral Coords 1 - Graph {i+1}")
        ax_spec2.set_title(f"Spectral Coords 2 - Graph {i+1}")
    # plt.tight_layout()

    # set title for the entire figure
    plt.suptitle("Spectral Coordinates Progression as New Nodes Are Added", fontsize=16)

    # print the spectral coordinates for each graph
    print()
    print("------------------------------------------------------------------------")
    print("-------------------- ORIGINAL SPECTRAL COORDINATES ---------------------")
    print("------------------------------------------------------------------------")
    for i, spec_coords in enumerate(spectral_coords_list):
        print(f"Spectral coordinates for Graph {i+1} (Set 1):")
        for node, coords in spec_coords.items():
            print(f" Node {node}: {coords}")
        print()

        if i < len(transformations):
            print("Transformation from Graph", i+2, "to Graph", i+1, "is:")
            A, b = transformations[i]
            print("A:\n", A)
            print("b:\n", b)

        print()
        print()

    print()
    print("------------------------------------------------------------------------")
    print("----------------- TRANSFORMED SPECTRAL COORDINATES ---------------------")
    print("------------------------------------------------------------------------")
    for i, spec_coords in enumerate(spectral_coords_list_transformed):
        print(f"Spectral coordinates for Graph {i+1} (Set 2 - Transformed):")
        for node, coords in spec_coords.items():
            print(f" Node {node}: {coords}")
        print()


    # compute distances between each pair of nodes in each graph for both sets of spectral coordinates (print the distance bi-matrices)
    print()
    print("------------------------------------------------------------------------")
    print("---------------- SPECTRAL COORDINATE DISTANCE MATRICES -----------------")
    print("------------------------------------------------------------------------")
    for i in range(len(graphs)):
        spec_coords1 = spectral_coords_list[i]
        spec_coords2 = spectral_coords_list_transformed[i]
        nodes = list(graphs[i].nodes())
        num_nodes = len(nodes)

        dist_matrix1 = np.zeros((num_nodes, num_nodes))
        dist_matrix2 = np.zeros((num_nodes, num_nodes))

        for j in range(num_nodes):
            for k in range(num_nodes):
                dist_matrix1[j, k] = np.linalg.norm(spec_coords1[nodes[j]] - spec_coords1[nodes[k]])
                dist_matrix2[j, k] = np.linalg.norm(spec_coords2[nodes[j]] - spec_coords2[nodes[k]])

        print(f"Distance matrix for Graph {i+1} (Set 1):")
        print(dist_matrix1)
        print()
        # print(f"Distance matrix for Graph {i+1} (Set 2 - Transformed):")
        # print(dist_matrix2)
        # print()
        print("------------------------------------------------------------------------")

    # check if "plots" directory exists, if not create it
    import os
    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig("plots/spectral_coords_progression.png")
    print("Saved spectral coordinates progression to 'plots/spectral_coords_progression.png'")

def test_transformation_estimation():
    # test the transformation from graph 3 to graph 4
    spec_coords_graph3 = np.array([
        [0.81649658, 0., 0., 0.],
        [-0.40824829, -0.70710678, 0., 0.],
        [-0.40824829, 0.70710678, 0., 0.]
    ])
    spec_coords_graph4 = np.array([
        [0.43620995, 0.70710678, 0.24437856, 0.],
        [0.43620995, -0.70710678, 0.24437856, 0.],
        [-0.28986734, -7.43764502e-16, -0.735511335, 0.],
        [-0.73172309, 3.04752249e-16, 0.582736062, 0.]
    ])

    A, b = fit_transformations_single(spec_coords_graph3, spec_coords_graph4[:-1,:])  # exclude last node of graph 4
    print("Estimated transformation from Graph 3 to Graph 4:")
    print("A:\n", A)
    print("b:\n", b)

    # check the transformation by applying it to graph 4's spectral coordinates
    transformed_coords = (A @ spec_coords_graph4.T).T + b

    print("Transformed Spectral Coordinates of Graph 4:")
    print(transformed_coords)
    print("Original Spectral Coordinates of Graph 3 (excluding last node):")
    print(spec_coords_graph3)

    # calculate mse
    mse = np.mean((transformed_coords[:-1,:] - spec_coords_graph3)**2)
    print("Mean Squared Error between transformed Graph 4 coords and Graph 3 coords:", mse)

def test_disconnected_graph():
    # create a disconnected graph with 5 nodes: two components (0-1-2) and (3-4)
    G = nx.Graph()
    G.add_edges_from([
        # first component with 8 nodes
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7),
        # second component with 7 nodes
        (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (10, 11), (10, 12), (10, 13), (10, 14), (11, 12), (11, 13), (11, 14), (12, 13), (12, 14), (13, 14),
    ])

    spectral_coords = get_spectral_coordinates(G, m=8)
    print("Spectral coordinates for disconnected graph:")
    for node, coords in spectral_coords.items():
        print(f" Node {node}: {list(np.round(coords, 3))}")

    # calculate dot products between all pairs of nodes
    dot_products = np.zeros((len(G.nodes()), len(G.nodes())))
    for i in range(len(G.nodes())):
        for j in range(len(G.nodes())):
            dot_products[i, j] = np.dot(spectral_coords[i], spectral_coords[j])
    print("Dot products between all pairs of nodes:")
    for row in np.round(dot_products, 3):
        for val in row:
            # account for the potential minus sign
            print(f"{str(val):7s}", end=" ")
        print()

    # plot the graph and its spectral coordinates
    import matplotlib.pyplot as plt
    fig, (ax_graph, ax_spec) = plt.subplots(1, 2, figsize=(10, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, ax=ax_graph)
    ax_graph.set_title("Disconnected Graph")
    x_coords = [spectral_coords[node][0] for node in G.nodes()]
    y_coords = [spectral_coords[node][1] for node in G.nodes()]
    ax_spec.scatter(x_coords, y_coords)
    for node in G.nodes():
        ax_spec.text(spectral_coords[node][0], spectral_coords[node][1], str(node))
    ax_spec.set_title("Spectral Coordinates")
    plt.tight_layout()
    # check if "plots" directory exists, if not create it
    import os
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig("plots/disconnected_graph_spectral_coords.png")
    print("Saved disconnected graph spectral coordinates to 'plots/disconnected_graph_spectral_coords.png'")

if __name__ == "__main__":
    # plot_spectral_coordinates_progression(m=4)
    # test_transformation_estimation()
    test_disconnected_graph()
    pass
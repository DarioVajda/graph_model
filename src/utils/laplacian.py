import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import warnings

def get_laplacian_coordinates(G, m, random_sign_flips=False):
    N = G.number_of_nodes()
    node_list = list(G.nodes())
    
    if N <= 1:
        return {node: np.zeros(m) for node in node_list}

    # Compute up to m+1 eigenvectors to discard the constant 0th.
    num_eigenvectors_to_compute = min(m + 1, N)
    
    # 1. Use Combinatorial Laplacian (Unnormalized) for purer geometry
    L = nx.laplacian_matrix(G).astype(float)
    
    # Check if the matrix is symmetric (it should be for an undirected graph)
    if not np.allclose(L.toarray(), L.toarray().T, atol=1e-8):
        warnings.warn("Laplacian matrix is not symmetric (because the graph is directed). This will produce unreliable eigenvalues/eigenvectors.")
    
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
    
    # 4. Filter and Scale features (THIS IS THE FIX)
    available_features = eigenvectors[:, 1 : num_eigenvectors_to_compute]
    available_eigenvalues = eigenvalues[1 : num_eigenvectors_to_compute]
    
    # Scale by inverse square root of eigenvalues to capture topology
    # np.maximum prevents division by zero in case of disconnected graph components
    scaling_factors = 1.0 / np.sqrt(np.maximum(available_eigenvalues, 1e-2))
    available_features = available_features * scaling_factors
    
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

#region --- Test ---
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
    spectral_coords_list = [get_laplacian_coordinates(G, m) for G in graphs]


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

def test_disconnected_graph():
    # create a disconnected graph with 5 nodes: two components (0-1-2) and (3-4)
    G = nx.Graph()
    G.add_edges_from([
        # first component with 8 nodes
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7),
        # second component with 7 nodes
        (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (10, 11), (10, 12), (10, 13), (10, 14), (11, 12), (11, 13), (11, 14), (12, 13), (12, 14), (13, 14),
    ])

    spectral_coords = get_laplacian_coordinates(G, m=8)
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
#endregion

if __name__ == "__main__":
    # plot_spectral_coordinates_progression(m=4)
    test_disconnected_graph()
    pass
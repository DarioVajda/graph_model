import os
import textwrap
import torch
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any

# Type hint based on your definition
TextGraph = Dict[str, Any]

def visualize_text_graph(
    graph_data: TextGraph, 
    output_path: str, 
    max_line_length: int = 30,
    use_spectral_layout: bool = False,
    figsize: tuple = (12, 12)
):
    """
    Visualizes a TextGraph with text wrapped inside node shapes.
    
    Args:
        graph_data: The dictionary containing 'text', 'num_nodes', 'edges', etc.
        output_path: Where to save the resulting image (e.g., './graph.png')
        max_line_length: The maximum number of characters per line in a node before wrapping.
        use_spectral_layout: If True and 'spectral_coords' exists, uses the first 2 dims as X/Y coordinates.
        figsize: The size of the matplotlib figure.
    """
    num_nodes = graph_data.get('num_nodes', 0)
    texts = graph_data.get('text', [])
    edges = graph_data.get('edges', [])
    prompt_node = graph_data.get('prompt_node', -1)
    
    # 1. Build the NetworkX graph for layout purposes
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # 2. Determine node positions
    pos = None
    if use_spectral_layout and 'spectral_coords' in graph_data and graph_data['spectral_coords'] is not None:
        coords = graph_data['spectral_coords']
        if isinstance(coords, torch.Tensor):
            coords = coords.numpy()
        
        # Check if we have at least 2 dimensions to plot
        if coords.shape[1] >= 2:
            # Typically, the 0th eigenvector is constant, so we use 1st and 2nd for plotting
            pos = {i: (coords[i, 1], coords[i, 2] if coords.shape[1] > 2 else coords[i, 0]) for i in range(num_nodes)}
            print("Using Spectral Coordinates for graph layout.")
            
    if pos is None:
        # Fallback to a spring layout (force-directed)
        # k controls the distance between nodes; scale it up slightly for text nodes
        pos = nx.spring_layout(G, seed=42, k=2.5 / (num_nodes ** 0.5))

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # 4. Draw Edges
    # We draw edges first so they sit *behind* the node text boxes
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.0, alpha=0.5, edge_color="gray")

    # 5. Draw Nodes (using matplotlib text boxes for beautiful wrapping)
    for i in range(num_nodes):
        x, y = pos[i]
        
        # Wrap the text
        raw_text = texts[i] if i < len(texts) else "Missing Text"
        wrapped_text = textwrap.fill(raw_text, width=max_line_length)
        
        # Format the label
        label = f"Node {i}\n" + "-"*7 + f"\n{wrapped_text}"
        
        # Styling: Highlight the prompt node differently
        if i == prompt_node:
            bg_color = '#d4edda'      # Light green
            border_color = '#28a745'  # Darker green
            border_width = 3.0
            z_order = 10              # Bring prompt node to the very front
        else:
            bg_color = '#f8f9fa'      # Light gray/white
            border_color = '#6c757d'  # Gray
            border_width = 1.5
            z_order = 5

        # Bounding box properties (rounded rectangle)
        bbox_props = dict(
            boxstyle="round,pad=0.6",
            fc=bg_color,     # Face color
            ec=border_color, # Edge color
            lw=border_width, # Line width
            alpha=0.95       # Slight transparency
        )
        
        # Render the text box at the given coordinate
        ax.text(
            x, y, label, 
            ha="center", va="center", 
            fontsize=10, 
            bbox=bbox_props, 
            zorder=z_order,
            family='monospace' # Monospace keeps the separator line tidy
        )

    # 6. Final touches
    ax.axis('off') # Hide the grid and axes
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph visualization saved to {output_path}")


# ==============================================================================
# TEST EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Simulate a TextGraph dictionary (just like the dataset __getitem__ returns)
    # I've included a long text to test the wrapping mechanism
    mock_graph_data = {
        'num_nodes': 5,
        'prompt_node': 4,
        'text': [
            "This is a short premise.",
            "This node contains significantly more text, which will test how well the text wrapping mechanism handles long sentences inside the node bounding box.",
            "Another standard node.",
            "Short text.",
            "This is the prompt node. Based on the previous nodes, what should the answer be?"
        ],
        'edges': [(0, 1), (1, 2), (2, 3), (1, 4), (3, 4)],
        # Mocking spectral coordinates: shape (num_nodes, spectral_dim)
        'spectral_coords': torch.tensor([
            [0.1,  0.5,  0.2, 0.0],
            [0.1,  0.0,  0.0, 0.0],
            [0.1, -0.5,  0.2, 0.0],
            [0.1, -0.2, -0.5, 0.0],
            [0.1,  0.2, -0.5, 0.0],
        ])
    }

    # 1. Test standard Force-Directed Layout
    print("Testing Force-Directed Layout...")
    visualize_text_graph(
        graph_data=mock_graph_data,
        output_path="./plots/graph_vis_standard.png",
        max_line_length=35,
        use_spectral_layout=False,
    )

    # 2. Test Spectral Coordinate Layout
    print("Testing Spectral Coordinate Layout...")
    visualize_text_graph(
        graph_data=mock_graph_data,
        output_path="./plots/graph_vis_spectral.png",
        max_line_length=35,
        use_spectral_layout=True,
    )
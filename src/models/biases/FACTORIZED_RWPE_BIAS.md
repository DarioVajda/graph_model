### Directed Random Walk Positional Encodings (RWPE) for Factorized Attention

**1. Graph Preliminaries**
Given a directed graph $G = (V, E)$ with $N$ nodes, let $A \in \mathbb{R}^{N \times N}$ be the adjacency matrix where $A_{ij} = 1$ if there is an edge $i \to j$, and $0$ otherwise. 
Define the out-degree diagonal matrix $D_{out}$ and in-degree diagonal matrix $D_{in}$ such that $(D_{out})_{ii} = \sum_j A_{ij}$ and $(D_{in})_{ii} = \sum_i A_{ij}$.

**2. Transition Matrices**
To capture directed structural flows, we define the forward and backward random walk transition matrices:
$$ M_{fwd} = D_{out}^{-1} A $$
$$ M_{bwd} = D_{in}^{-1} A^T $$

**3. Expressive Node-Level RWPE Extraction**
For a walk of length $t \in \{1, \dots, k\}$, the diagonal entries of $M^t$ represent the probability of a random walk returning to the starting node after $t$ steps (capturing local cyclic and motif structures).
For each node $i$, the forward and backward return probabilities up to $k$ (default to $k=24$) steps are:
$$ r_{fwd, i} = \left[ (M_{fwd}^1)_{ii}, (M_{fwd}^2)_{ii}, \dots, (M_{fwd}^k)_{ii} \right]^T \in \mathbb{R}^k $$
$$ r_{bwd, i} = \left[ (M_{bwd}^1)_{ii}, (M_{bwd}^2)_{ii}, \dots, (M_{bwd}^k)_{ii} \right]^T \in \mathbb{R}^k $$
The raw structural feature vector for node $i$ is the concatenation of these return probabilities:
$$ p_i = \left[ r_{fwd, i} \parallel r_{bwd, i} \right] \in \mathbb{R}^{2k} $$

**4. Non-Linear Feature Mapping**
To ensure expressiveness, $p_i$ is mapped through a shallow multi-layer perceptron (MLP) to form a dense positional encoding:
$$ z_i = \text{MLP}_{PE}(p_i) \in \mathbb{R}^{d_{pos}}$$

Example of an **MLP shape**: Linear($48 \rightarrow 128$) $\rightarrow$ SiLU $\rightarrow$ Linear($128 \rightarrow d_{pos}=32$)

**5. Factorized Q and K Projections**
Let $h_i \in \mathbb{R}^{d_{model}}$ be the content feature of node $i$. The Key ($K$) and Query ($Q$) representations are computed by projecting the content and structural features separately and concatenating them:
$$ Q_i = \left[ W_Q h_i \parallel W_{Q, pos} z_i \right] \in \mathbb{R}^{d_Q + d_{pos}} $$
$$ K_i = \left[ W_K h_i \parallel W_{K, pos} z_i \right] \in \mathbb{R}^{d_K + d_{pos}} $$
*(where $W_Q, W_K \in \mathbb{R}^{d_{model} \times d_Q}$ and $W_{Q, pos}, W_{K, pos} \in \mathbb{R}^{d_{pos} \times d_{pos}}$)*

**6. Flash Attention Compatibility**
Because the structural bias is strictly factorized into the $Q$ and $K$ node embeddings, the attention mechanism is a standard dot product without edge-level bias additions:
$$ \text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_Q + d_{pos}}}\right) V $$
This dot product naturally decomposes into content-content and structure-structure correlations:
$$ Q_i K_j^T = (W_Q h_i)(W_K h_j)^T + (W_{Q, pos} z_i)(W_{K, pos} z_j)^T $$
Since the attention matrix is computed purely via dense matrix multiplication of $Q$ and $K$, it natively supports hardware-accelerated exact attention algorithms like FlashAttention.
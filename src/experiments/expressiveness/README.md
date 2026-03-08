## Expressiveness of Graph-aware Positional Encodings

In this experiment, I will try to train both the GraphLlamaModel and the LlamaModel on a simple task to show that the modified graph-aware positional encodings empower the LLM to see graphs, without having explicit information about edges in the token sequence.

### Problem setup
- The model is given a set of $N$ node labels $l_1, l_2,... l_N$ (possibly letters or numbers).
- These nodes will form a graph, with two distinct connected components $\{l_{i_1},...l_{i_{M}}\}$ and $\{l_{i_{M+1}},...l_{i_N}\}$, where $M<N$.
- The model is asked the following question: *"Are nodes $l_x$ and $l_y$ connected?"* $\rightarrow$ *Yes/No*
- We will use the Laplacian coordinates and Shortest Path Distance matrix to see if those are expressive enough to solve the task.
- **HARD version** - this version of the problem contains a variable number of components, where they are not fully connected and the graph contains directed edges 

### Hypothesis
1. The default LlamaModel will not have any information on the graph structure, as it is only presented the set of nodes without edge information. Therefore, the model cannot perform better than a random binary classifier (~$50\%$ accuracy)
2. The GraphLlamaModel will be able to learn the simple pattern and answer with high accuracy, as a result of the graph-aware positional embeddings.

### Findings
Success ✅

### **Conclusion**
The modified Llama model DOES SEE GRAPHS.
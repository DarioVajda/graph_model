# Towards Native Unification of Graph and Text Modalities

### Dario Vajda, University of Ljubljana
This is a research project I am working on as student and a Machine Learning researcher at Faculty of Computer and Information Science in Ljubljana, with the goal of empowering LLMs to natively process graph data. 

I believe this original model architecture will help solve LLM hallucinations, and also bridge the gap between foundation models for sequential text, Geometric Deep Learning and Relational Deep Learning.

## This Codebase

This is a repository with all of the code used for my research – it includes low-level architecture implementations, data processing, experiments, and evaluation scripts. Moreover, it captures the entire raw research process I am going through and is ready for straight-forward reproduction.

## Research Overview
The main motivation for this research project is finding a way to overcome a core limitation that traditional Large Language Models (LLMs) have - they operate purely on sequential data. However, as modern LLMs can be taught of as a special case of a Graph Transformer (GT), it provokes the idea of unifying the functionalities of LLMs and general GTs. 

To unify the graph and text modality, all data would be represented with an underlying graph structure, where all nodes (and possibly edges too) are pieces of text. This global graph would be processed by a GT-like model which would take into account both the global graph structure and also the relative positions of text tokens inside each node. This architecture enables LLMs to see graphs in a very parameter-efficient way, training less than 0.01% of the total number of LLM parameters.



## Base Architecture Innovation: Graph-Aware Relative Positional Encodings
The main innovation of this approach is the combination of relative positional embeddings, traditionally used by LLMs such as RoPE, and positional encodings used for graphs. To maintain backward compatibility and leverage the pretrained knowledge of the LLM, the relative sequence positional encoding preserves the original approach used by the given LLM, with the only modification being that the token indices are reset for every node individually. 

The model injects structural graph information directly into the self-attention computation via additive biases. The raw attention score matrix for the $h$-th attention head within the $l$-th layer is calculated as:

$$A_{i,j}^{(l,h)} = \frac{Q_{i}^{(l,h)} \cdot (K_{j}^{(l,h)})^T}{\sqrt{d_{head}}} + b_{graph}^{(l,h)}(i,j)$$

Where $b_{graph}$ represents the sum of selected graph-aware attention biases. This architecture produces the exact same attention scores between two tokens within the same node, under the simple assumption that the bias term is equal to 0 when the distance is 0.

### Available Attention Biases
The custom attention layer supports five distinct types of graph-aware biases:

* **Shortest Path Distance (SPD) Bias:** A learned lookup table mapping the shortest path distance (SPD) of the corresponding nodes to an attention bias. 

$$
b_{ij} = \text{spd\_weights}[d_{ij}] \quad \text{if } d_{ij} > 0 \text{ else } 0
$$

* **Laplacian Bias:** This bias term is proportional to the $L_2$-distance of the Laplacian embeddings...

$$
b_{ij} = w_k \cdot D_{L_2}(s_i, s_j)
$$

* **Random Walk Structural Encoding (RWSE) Bias:** Maps the $L_2$ distance between the random walk structural features of two nodes into a scalar bias value.
  $$b_{ij} = w_k \cdot D_{L_2}(r_i, r_j)$$

* **Relative Random Walk Probability (RRWP) Bias:** Based on the multi-hop probability of a random walk starting at node $i$ and landing on node $j$. A 2-layer MLP maps the vector of transition probabilities directly to the attention heads.
  $$b_{ij} = \text{MLP}(\text{RRWP}_{ij}) \quad \text{if } i \neq j \text{ else } 0$$

## Intended Use Cases
By fundamentally changing how the LLM perceives input context, this model unlocks several capabilities:

* **Native Knowledge Graph Ingestion (GraphRAG):** Instead of flattening and serializing complex knowledge graphs for the LLM prompt, this model can natively input knowledge graphs using a GraphRAG-like algorithm. Allowing the LLM to traverse explicit graph structures directly in its attention layers can significantly reduce hallucinations common in flattened context windows.
* **Mixed Modality Few-Shot Reasoning:** The model will be able to perform many different tasks with text and/or graphs with few-shot prompts.
* **Text-Heavy Node Classification:** Excels at tasks where nodes contain rich textual information that dictates their category within a wider network, such as the OGBN-Arxiv benchmark.
* **Graph-Level Classification:** Applicable to domains requiring holistic understanding of a graph's structure, such as chemical molecule data tested through the MoleculeNet benchmark suite.
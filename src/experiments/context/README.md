# Experimental Plan: GTLM Context Exhaustion Stress Test (Needle in a Graph)

## 1. Objective
To empirically determine the exact graph size and token length at which the Graph Transformer Language Model (GTLM) degrades due to context exhaustion (attention dilution), directly addressing the reviewer's concern regarding the uncompressed text processing.

## 2. Experimental Design
* **Task:** Synthetic Key-Value (KV) Retrieval on a Star Graph.
* **Architecture:** A single GTLM model trained on a continuous mixture of graph sizes and token lengths.
* **Independent Variables:** 
  * Number of nodes (N): 8, 16, 32, 64, 128
  * Tokens per node (T): 32, 64, 128, 256, 512
  * **Note**: Limit the total sequence length to some empirically chosen value as an optimization to speed up training (e.g. 16.384) – TODO choose max size
* **Dependent Variable:** Exact Match Accuracy of the retrieved value.

---

## 3. Execution Steps

### Phase 1: Synthetic Dataset Generation
Generate a training dataset where the model must extract a specific value from a single "gold" node surrounded by distractor nodes.

1. **Topology:** Create Star Graphs. One node is the "question/query" node which is in the graph prefix, connected to the other nodes, while the final "target" node where the answer is predicted is the node with the causal mask where the model learns.
2. **Distractor Content:** Fill N-2 neighbor nodes with random text chunks (e.g., from Wikipedia) truncated to T tokens. Insert a fake/random KV pair into each (e.g., "The access code for this node X is Y").
3. **Gold Content:** In exactly one random neighbor node, insert the true KV pair required to answer the query, padded to T tokens with random text.
4. **Query Node:** Set the text of Node 0 to: "What is the access code for [Gold Node ID]?"
5. **Size Mixture:** For the training set, uniformly sample N between all available sizes

### Phase 2: Model Training
Train a single model to prevent configuration-specific overfitting.

1. Initialize the GTLM model with Llama 1B.
2. Train the model on the mixed-size dataset generated in Phase 1.
3. Monitor validation loss to ensure the model learns the basic KV extraction mechanism regardless of the prompt size.

### Phase 3: Grid Evaluation
Evaluate the single trained model systematically.

1. Generate 25 distinct Test Sets, one for each (N, T) combination in our grid.
2. Generate 200 graphs for the largest setting (128 nodes and 512 tokens) and get all other test graphs by subsampling this biggest setting (node subset for smaller N and truncated texts for smaller T)
3. Run inference using the trained model on all 25 Test Sets.
4. Record the Exact Match Accuracy for each grid cell. Track the total token sequence length (N * T) for each cell.

### Phase 4: Visualization and Table
Plot the results to include in the paper.

1. Use Python (seaborn/matplotlib) to generate a 2D heatmap.
2. X-axis: Tokens per Node (T).
3. Y-axis: Number of Neighbors (N).
4. Cell Values: Exact Match Accuracy (annotated with the total sequence length).
5. Generate a textual markdown table with the same information
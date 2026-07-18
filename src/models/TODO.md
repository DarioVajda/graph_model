# GTLM Architecture: Solving "RoPE Shock" and the 1.4 F1 Deficit

## Context
During fine-tuning on WebQSP/CWQ, moving the query into a global bidirectional prefix node (using the `isolated` connection strategy) successfully cured GTLM's "query-blindness." Every seed beat the old graph construction approach, closing a significant portion of the performance gap.

However, GTLM still trails a flat-text LLM baseline by ~1.4 F1 points. This remaining deficit is diagnosed as **"RoPE Shock"** (or the RoPE Tax). 
By continuously resetting position IDs to 0 at the start of every node to strictly preserve permutation equivariance, we are feeding the pre-trained LLaMA base model an out-of-distribution positional signal. The model struggles to parse the internal linguistic syntax of the node texts because its pre-trained Rotary Position Embeddings (RoPE) are suffering from massive phase collisions.

The goal is to eliminate this RoPE shock while strictly preserving GTLM's core values: **permutation equivariance** and **graph-native structural routing**.

---

## Idea 1: Fully Unfreeze W_q and W_k (The Optimization Fix)

**The Concept:** Currently, the model uses LoRA adapters. RoPE rotations are applied directly to the Query and Key vectors after they are projected by the `W_q` and `W_k` matrices. Because low-rank adapters lack the parameter capacity to fundamentally redefine vector subspaces, the model cannot "unlearn" its pre-trained RoPE collision expectations.

**The Fix:** Fully unfreeze the `W_q` and `W_k` projection matrices across all attention layers. 

**Why it preserves core values:** Requires absolutely zero changes to the architecture, position IDs, or graph topology. It simply gives the attention mechanism the full-rank freedom to project token embeddings into a new subspace where RoPE phase collisions are ignored, forcing the model to rely entirely on the injected SPD biases for spatial routing.

**Implementation:**
- Remove `q_proj` and `k_proj` from the LoRA `target_modules` config.
- Manually set `requires_grad = True` for these specific layers.
- Expected outcome: Slower initial convergence, but a higher expressivity ceiling as the model learns to decouple syntax from global position.

---

## Idea 2: Topological Phase Shifting (The Mathematical Fix)

**The Concept:** Instead of resetting every node's position ID strictly to 0, offset the starting position ID of each node based on its shortest path distance (hops) from the isolated query node. 

**The Math:**
- Let `C` be a safe constant buffer covering max node token length (e.g., C = 64).
- Let `d(u)` be the topological distance from the query to node `u`.
- Position IDs for node `u` become: `[C * d(u), (C * d(u)) + 1, (C * d(u)) + 2, ...]`

**Why it preserves core values:**
Permutation equivariance is strictly maintained. The offset `d(u)` is a structural property of the graph, entirely immune to 1D serialization order. While nodes in the same "distance shell" (e.g., all 1-hop neighbors) will still suffer index collisions with each other, this is mathematically correct for permutation equivariance. The crucial win is that parent/child nodes on a traversal path no longer suffer syntax collisions, allowing RoPE to handle intra-node syntax while the SPD biases handle inter-node routing.

**Implementation:**
- No architectural or weight changes required.
- Update the data collator to compute `d(u)` and generate the custom 1D `position_ids` array.
- Pass this array directly into the standard LLaMA forward pass.

---

## Reference Baseline: "Proposal B" (Flat Serialization + Node Spans)

**The Concept:** Serialize the entire graph into a continuous 1D string (Token 0 to Token N). Then, overlay GTLM's structural SPD biases specifically onto the token spans that correspond to the respective nodes.

**Why it is a reference, not the solution:**
This completely eliminates RoPE shock because it gives LLaMA the exact 1D positional flow it was trained on. However, it requires imposing a synthetic linear order on the graph, breaking true permutation equivariance. 
*Action:* We do not adopt this as the core architecture. However, it may be worth running one experiment with this setup purely to prove that restoring 1D RoPE mathematically accounts for the exact 1.4 F1 point gap.
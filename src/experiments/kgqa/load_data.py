import random
import torch
import networkx as nx

from ...utils import TextGraphDataset

# The prompt node's text is "{QUESTION_PREFIX}{degree}".
# We mask the question prefix tokens to -100 and train on the answer (the degree digit).
QUESTION_PREFIX = "What is my out-degree? Answer: "


def load_data(tokenizer, num_samples=80, seed=42):
    """
    Generate a small synthetic graph dataset for demonstration.

    Task: predict the out-degree of a designated prompt node given its local
    graph neighbourhood encoded as node text.

    Returns (train_dataset, eval_dataset, test_dataset) — a 75/12.5/12.5 split.
    """
    rng = random.Random(seed)
    graphs = []

    for _ in range(num_samples):
        n_nodes = rng.randint(4, 7)
        g = nx.gnp_random_graph(n_nodes, 0.35, seed=rng.randint(0, 100_000), directed=True)

        prompt_node = rng.randint(0, n_nodes - 1)
        degree = g.out_degree(prompt_node)

        for node in g.nodes():
            if node == prompt_node:
                g.nodes[node]['text'] = f"{QUESTION_PREFIX}{degree}"
            else:
                g.nodes[node]['text'] = f"Node {node}."
        g.graph['prompt_node'] = prompt_node
        graphs.append(g)

    ds = TextGraphDataset(graphs)
    ds.compute_shortest_path_distances()
    ds.compute_rrwp(max_rrwp_steps=16)
    ds.compute_magnetic_lap(q=0.25)
    ds.tokenize(tokenizer, max_length=64, add_eos=True)

    # precompute question prefix length once so the closure is picklable
    q_len = len(tokenizer(QUESTION_PREFIX, add_special_tokens=False)['input_ids'])

    def get_labels(example):
        input_ids = example['input_ids'][example['prompt_node']]
        labels = torch.tensor(input_ids, dtype=torch.long)
        labels[:q_len] = -100  # mask the question, supervise only the answer
        return labels

    ds.compute_labels(get_labels)

    n = len(ds)
    n_train = int(0.75 * n)
    n_eval = int(0.125 * n)

    train_ds = ds[:n_train]
    eval_ds  = ds[n_train : n_train + n_eval]
    test_ds  = ds[n_train + n_eval :]

    return train_ds, eval_ds, test_ds

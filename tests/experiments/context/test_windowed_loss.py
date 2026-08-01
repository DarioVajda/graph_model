"""The windowed loss must equal the full-logits loss it replaces.

At L = 65k the ordinary path computes an ``(B, L, V)`` logit tensor — 16.5 GB of
bf16 — of which all but ~4 rows multiply into ``-100``. ``evaluate.windowed_*``
slices that away with ``logits_to_keep``, which means the training objective is
now computed by code this repo owns rather than by HuggingFace's loss function.
If the slice or the shift is off by one, nothing raises: the model simply trains
on the wrong target, or scores exact match against the wrong positions.

So this file pins both halves against the unsliced reference on a tiny model:
the loss value, and the argmax positions the exact-match metric reads.
"""

import networkx as nx
import pytest
import torch

from src.experiments.context.evaluate import (
    window_start, windowed_forward, windowed_loss,
)
from src.models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from src.utils.text_graph_collator_v2 import GraphCollatorV2
from src.utils.text_graph_dataset import TextGraphDataset

_BASE = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=256, max_position_embeddings=512,
    pad_token_id=0, _attn_implementation="eager",
)
_BIAS = dict(spd=True, max_spd=8, magnetic=True, magnetic_dim=8)

CODE_LEN = 3          # supervised tokens before EOS, as in the real config


def _star_graph(n_content=4, t=8, seed=0):
    """A miniature of the real topology: QUESTION centre, content leaves, isolated PROMPT."""
    rng = torch.Generator().manual_seed(seed)
    g = nx.Graph()
    g.add_node(0, text="q")
    for i in range(1, n_content + 1):
        g.add_node(i, text=f"c{i}")
        g.add_edge(0, i)
    prompt = n_content + 1
    g.add_node(prompt, text="a")
    g.graph["prompt_node"] = prompt
    g.graph["question_node"] = 0

    # Token ids are assigned directly (this test is about the loss window, not the
    # text builder): content nodes get t tokens, the prompt node gets a 2-token
    # prefix + CODE_LEN code tokens + EOS.
    ids = [torch.randint(1, 256, (3,), generator=rng).tolist()]
    ids += [torch.randint(1, 256, (t,), generator=rng).tolist() for _ in range(n_content)]
    ids += [torch.randint(1, 256, (2 + CODE_LEN + 1,), generator=rng).tolist()]
    return g, ids, prompt


def _batch(n_graphs=2, n_content=4, t=8):
    graphs, all_ids, prompts = [], [], []
    for s in range(n_graphs):
        g, ids, prompt = _star_graph(n_content=n_content, t=t, seed=s)
        graphs.append(g)
        all_ids.append(ids)
        prompts.append(prompt)

    ds = TextGraphDataset(graphs)
    ds.compute_shortest_path_distances(cutoff=8, use_gpu=False)
    ds.compute_magnetic_lap(q=0.25, use_gpu=False, m=0)

    items = []
    for i in range(len(ds)):
        item = dict(ds[i])
        item["input_ids"] = all_ids[i]
        labels = torch.full((len(all_ids[i][prompts[i]]),), -100, dtype=torch.long)
        labels[2:] = torch.tensor(all_ids[i][prompts[i]][2:], dtype=torch.long)
        item["labels"] = labels
        items.append(item)

    collator = GraphCollatorV2(pad_token_id=0, k_hop=0, magnetic_m=0, pad_to_block=False)
    return collator(items)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = GTLMLlamaConfig(k_hop=0, graph_attn_impl="eager", **_BIAS, **_BASE)
    return GTLMLlamaForCausalLM(cfg).eval()


def test_window_starts_one_before_the_first_supervised_label():
    """The logit at t predicts token t+1, so the window must open one position early."""
    labels = torch.full((2, 20), -100, dtype=torch.long)
    labels[:, 15:19] = 7
    assert window_start(labels) == 14


def test_window_takes_the_earliest_start_in_the_batch():
    labels = torch.full((2, 20), -100, dtype=torch.long)
    labels[0, 15:19] = 7
    labels[1, 12:16] = 7
    assert window_start(labels) == 11


def test_windowed_loss_matches_the_full_logits_loss(model):
    batch = _batch()
    labels = batch.pop("labels")

    with torch.no_grad():
        reference = model(**batch, labels=labels).loss
        logits, window_labels = windowed_forward(model, batch, labels)
        windowed = windowed_loss(logits, window_labels)

    assert logits.shape[1] < batch["input_ids"].shape[1], "the window did not shrink anything"
    torch.testing.assert_close(windowed, reference, rtol=1e-5, atol=1e-6)


def test_windowed_argmax_matches_the_full_logits_argmax_on_supervised_positions(model):
    """What exact match actually reads: the predictions at the label positions."""
    batch = _batch()
    labels = batch.pop("labels")

    with torch.no_grad():
        full = model(**batch).logits
        logits, window_labels = windowed_forward(model, batch, labels)

    full_pred = full[:, :-1, :].argmax(-1)
    full_gold = labels[:, 1:]
    win_pred = logits[:, :-1, :].argmax(-1)
    win_gold = window_labels[:, 1:]

    for row in range(labels.shape[0]):
        assert torch.equal(full_pred[row][full_gold[row] != -100],
                           win_pred[row][win_gold[row] != -100])
        assert torch.equal(full_gold[row][full_gold[row] != -100],
                           win_gold[row][win_gold[row] != -100])


def test_num_items_in_batch_normalizes_like_hf(model):
    """Gradient accumulation passes a token count; the loss must be sum/count."""
    batch = _batch()
    labels = batch.pop("labels")
    with torch.no_grad():
        logits, window_labels = windowed_forward(model, batch, labels)
    n_tokens = int((window_labels[:, 1:] != -100).sum())
    mean_loss = windowed_loss(logits, window_labels)
    sum_loss = windowed_loss(logits, window_labels, num_items_in_batch=n_tokens)
    torch.testing.assert_close(mean_loss, sum_loss, rtol=1e-5, atol=1e-6)

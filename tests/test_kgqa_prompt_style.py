"""
Pin the D3 chat prompt style (instruct backbone) against its locked design.

Design (TODO 2026-07-08): graph nodes unchanged; ONE prompt node holds the full
chat-templated string with a pinned minimal system turn; the assistant-header
token sequence replaces "Answer:" as ``question_end`` (label mask + generation
cut); the instruct tokenizer's EOS *is* ``<|eot_id|>`` so ``add_eos=True``
already appends the right terminator; the plain style must be byte-identical
to the dfv3 behavior it replaces.

Tokenizer-dependent tests use the real Llama-3.2 tokenizer (cached on this
machine); they are skipped when it can't be loaded (e.g. offline CI).
"""

import pytest

from src.experiments.kgqa.config import (
    RunConfig, ASSISTANT_HEADER, PINNED_SYSTEM_PROMPT)
from src.experiments.kgqa.process_dataset import (
    ANSWER_DELIM, AnswerLabelMasker, add_prompt_node, chat_prompt_text)
from src.experiments.kgqa.evaluate import _find_prefix_len

INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"
BASE = "meta-llama/Llama-3.2-1B"


def _tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(INSTRUCT)
    except Exception as e:                                    # pragma: no cover
        pytest.skip(f"instruct tokenizer unavailable: {e}")


# ── config resolution ────────────────────────────────────────────────────────
def test_prompt_style_auto_default():
    assert RunConfig(model_name=BASE).resolved_prompt_style == "plain"
    assert RunConfig(model_name=INSTRUCT).resolved_prompt_style == "chat"
    # explicit override wins (the instruct+plain control arm)
    assert RunConfig(model_name=INSTRUCT,
                     prompt_style="plain").resolved_prompt_style == "plain"


def test_prompt_style_validated():
    with pytest.raises(ValueError):
        RunConfig(prompt_style="chatml").validate()
    RunConfig(prompt_style="chat").validate()
    RunConfig().validate()                                    # None (auto) is fine


def test_question_end_str_tracks_style():
    assert RunConfig(model_name=BASE).question_end_str == "Answer:"
    assert RunConfig(model_name=INSTRUCT).question_end_str == ASSISTANT_HEADER


def test_data_config_key_suffix_only_for_chat():
    plain_key = RunConfig(model_name=BASE).data_config_key("webqsp")
    assert "_ps" not in plain_key                             # old caches keep their names
    chat_key = RunConfig(model_name=INSTRUCT).data_config_key("webqsp")
    assert chat_key.endswith("_pschat")
    # instruct+plain control: separate cache via model_name, no style suffix
    ctrl_key = RunConfig(model_name=INSTRUCT, prompt_style="plain").data_config_key("webqsp")
    assert "Instruct" in ctrl_key and "_ps" not in ctrl_key


# ── prompt-node construction ─────────────────────────────────────────────────
def _mini_graph():
    import networkx as nx
    G = nx.DiGraph()
    G.add_node("m.1", text="Bulgaria")
    return G


REC = {"question": "what is nina dobrev nationality", "entities": ["m.1"]}


def test_chat_prompt_node_text_structure():
    g = add_prompt_node(_mini_graph(), REC, "Bulgaria\nCanada", ["Bulgaria"],
                        prompt_style="chat")
    text = g.nodes["PROMPT"]["text"]
    assert text.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                           + PINNED_SYSTEM_PROMPT + "<|eot_id|>")
    assert "<|start_header_id|>user<|end_header_id|>\n\n" + REC["question"] + "<|eot_id|>" in text
    # answers follow the assistant header DIRECTLY (no leading space, unlike plain)
    assert text.endswith(ASSISTANT_HEADER + "Bulgaria\nCanada")
    # never the template default system turn (its auto "Today Date" would make
    # cache content depend on build date)
    assert "Today Date" not in text and "Cutting Knowledge" not in text


def test_chat_unanswerable_row_ends_at_assistant_header():
    g = add_prompt_node(_mini_graph(), REC, "", ["William Roache"], prompt_style="chat")
    assert g.nodes["PROMPT"]["text"].endswith(ASSISTANT_HEADER)
    assert g.graph["unanswerable"] is True


def test_plain_style_unchanged_dfv3_golden():
    g = add_prompt_node(_mini_graph(), REC, "Bulgaria\nCanada", ["Bulgaria"])
    assert g.nodes["PROMPT"]["text"] == (
        REC["question"] + ANSWER_DELIM + " Bulgaria\nCanada")
    g_empty = add_prompt_node(_mini_graph(), REC, "", ["x"])
    assert g_empty.nodes["PROMPT"]["text"] == REC["question"] + ANSWER_DELIM


# ── tokenizer-level invariants (real instruct tokenizer) ─────────────────────
def test_special_tokens_roundtrip_as_single_ids():
    tok = _tok()
    ids = tok(ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
    assert ids == [128006, 78191, 128007, 271]                # header + "\n\n"
    for s, want in (("<|begin_of_text|>", 128000), ("<|eot_id|>", 128009)):
        got = tok(s, add_special_tokens=False)["input_ids"]
        assert got == [want], f"{s} -> {got}"
    assert tok.eos_token_id == 128009                         # add_eos appends <|eot_id|>


def test_mask_starts_exactly_after_assistant_header():
    tok = _tok()
    question_end = tok(ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
    text = chat_prompt_text(REC["question"]) + "Bulgaria\nCanada"
    ids = tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    labels = AnswerLabelMasker(question_end)(
        {"input_ids": [ids], "prompt_node": 0})
    cut = _find_prefix_len(ids, question_end)
    assert cut is not None
    assert all(l == -100 for l in labels[:cut])               # prompt fully masked
    assert labels[cut:] == ids[cut:]                          # answers + <|eot_id|> supervised
    # the supervised span decodes back to exactly the target + terminator
    assert tok.decode(ids[cut:], skip_special_tokens=True) == "Bulgaria\nCanada"
    assert ids[-1] == 128009
    # generation cut: the prefix ends exactly at the header's "\n\n"
    assert ids[cut - len(question_end):cut] == question_end


def test_answers_after_header_keep_standalone_newline_boundaries():
    tok = _tok()
    # dates/digits directly after the header "\n\n" and after each "\n" must not
    # merge into multi-char whitespace tokens (the boundary the model learns)
    text = chat_prompt_text("how old is sacha baron cohen") + "1971-10-13\nMonarchy"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    q_end = tok(ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
    cut = _find_prefix_len(ids, q_end)
    tail = ids[cut:]
    assert 198 in tail                                        # standalone "\n" boundary
    assert tok.decode(tail) == "1971-10-13\nMonarchy"


def test_chat_unanswerable_labels_are_terminator_only():
    tok = _tok()
    question_end = tok(ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
    ids = tok(chat_prompt_text(REC["question"]),
              add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    labels = AnswerLabelMasker(question_end)(
        {"input_ids": [ids], "prompt_node": 0})
    assert labels[:-1] == [-100] * (len(ids) - 1)
    assert labels[-1] == 128009


def test_header_subsequence_unambiguous_even_with_tricky_answers():
    tok = _tok()
    question_end = tok(ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
    # an answer containing the word "assistant" must not re-trigger the masker
    text = chat_prompt_text("who is it") + "assistant\nBulgaria"
    ids = tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    AnswerLabelMasker(question_end)({"input_ids": [ids], "prompt_node": 0})

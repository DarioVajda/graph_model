"""Pin the three document strategies and the invariant that makes them safe to compare.

`key_value` labels every field on every row and drops the fields a row does not populate.
The other two hoist a table's column list into a header node and render its rows as bare
positional values -- which is only meaningful if every row emits a slot for every column,
including the ones it lacks. A dropped slot shifts every later value into the wrong column,
silently: the document still reads fine, it just says the wrong things about the row.
"""

import pandas as pd
import pytest

from src.experiments.relbench.data import build_flat_graph, build_graph, schema_node_id
from src.experiments.relbench.row_text import NULL_SLOT, RowRenderer


class _Table:
    def __init__(self, df, pkey_col=None, time_col=None, fkeys=None):
        self.df = df
        self.pkey_col = pkey_col
        self.time_col = time_col
        self.fkey_col_to_pkey_table = fkeys or {}


class _DB:
    """Two `results` rows, one of which leaves `rank` and `laps` unset."""

    def __init__(self):
        self.table_dict = {
            "drivers": _Table(pd.DataFrame({"id": [0], "surname": ["Gerard"]}),
                              pkey_col="id"),
            "results": _Table(
                pd.DataFrame({"id": [0, 1], "grid": [13, 4],
                              "laps": [67, None], "rank": [None, 2.0]}),
                pkey_col="id", fkeys={}),
        }


class _Sampled:
    nodes = [("drivers", 0, 0), ("results", 0, 1), ("results", 1, 1)]
    edges = [(1, 0), (2, 0)]


def _renderer():
    return RowRenderer(_DB(), null_threshold=1.0)


def _texts(graph):
    return {n: graph.nodes[n]["text"] for n in graph.nodes}


# -- the alignment invariant --------------------------------------------------

def test_aligned_row_emits_a_slot_for_every_column():
    """The header promises column i at slot i. A sparse row must still fill its slots."""
    r = _renderer()
    header_cols = r.header("results").split(" | ")[1:]
    for row in (0, 1):
        values = r.render("results", row, 0, aligned=True).split(" | ")
        assert len(values) == len(header_cols), (
            f"row {row} has {len(values)} values against {len(header_cols)} columns")


def test_aligned_row_marks_missing_fields_rather_than_dropping_them():
    r = _renderer()
    assert NULL_SLOT in r.render("results", 0, 0, aligned=True).split(" | ")  # `rank` unset


def test_labelled_row_still_drops_missing_fields():
    """key_value's behaviour is unchanged -- it is the control and must not move."""
    r = _renderer()
    assert "rank" not in r.render("results", 0, 0)
    assert "grid: 13" in r.render("results", 0, 0)


def test_aligned_row_omits_the_table_name_the_header_carries():
    r = _renderer()
    assert not r.render("results", 0, 0, aligned=True).startswith("results")
    assert r.render("results", 0, 0).startswith("results")


# -- mode selection -----------------------------------------------------------

def test_key_value_hoists_nothing():
    g = build_graph(_Sampled(), _renderer(), 0, "q", "Answer: yes", text_mode="key_value")
    assert not any(str(n).startswith("__schema__") for n in g.nodes)


def test_schema_node_hoists_every_sampled_table():
    g = build_graph(_Sampled(), _renderer(), 0, "q", "Answer: yes", text_mode="schema_node")
    for table in ("drivers", "results"):
        assert schema_node_id(table) in g.nodes


def test_rows_point_at_their_header():
    g = build_graph(_Sampled(), _renderer(), 0, "q", "Answer: yes", text_mode="schema_node")
    for position in (1, 2):                       # the two `results` rows
        assert (position, schema_node_id("results")) in g.edges
    assert (0, schema_node_id("drivers")) in g.edges


def test_shortest_never_produces_a_longer_document_than_key_value():
    """The whole point of the mode. `drivers` contributes one row, so hoisting it cannot
    pay; `results` contributes two, so it might."""
    args = (_Sampled(), _renderer(), 0, "q", "Answer: yes")
    kv = build_flat_graph(*args, text_mode="key_value")
    sh = build_flat_graph(*args, text_mode="shortest")
    kv_len = len(next(iter(_texts(kv).values())))
    sh_len = len(next(iter(_texts(sh).values())))
    assert sh_len <= kv_len


def test_shortest_leaves_singleton_tables_labelled():
    g = build_graph(_Sampled(), _renderer(), 0, "q", "Answer: yes", text_mode="shortest")
    assert schema_node_id("drivers") not in g.nodes, (
        "one row can never amortize a header")


# -- the flat control ---------------------------------------------------------

def test_flat_puts_headers_before_the_rows():
    """Flat is read causally, so a header after its rows is useless."""
    text = next(iter(_texts(build_flat_graph(
        _Sampled(), _renderer(), 0, "q", "Answer: yes", text_mode="schema_node")).values()))
    lines = text.split("\n")
    header_at = next(i for i, l in enumerate(lines) if l.startswith("TABLE results"))
    row_at = next(i for i, l in enumerate(lines) if l.startswith("TARGET "))
    assert header_at < row_at


def test_arms_stay_byte_identical_under_the_default_mode():
    """The headline comparison runs at `key_value`, where there are no header lines and the
    two arms must carry exactly the same characters."""
    args = (_Sampled(), _renderer(), 0, "QUESTION | q", "Answer: yes")
    graph = build_graph(*args, text_mode="key_value")
    flat = build_flat_graph(*args, text_mode="key_value")
    graph_chars = sum(len(t) for t in _texts(graph).values())
    flat_chars = len(next(iter(_texts(flat).values())))
    # flat joins with newlines; the graph arm's node texts are the same strings.
    assert flat_chars == graph_chars + len(_texts(graph)) - 1


@pytest.mark.parametrize("mode", ["key_value", "schema_node", "shortest"])
def test_every_mode_keeps_the_seed_at_node_zero(mode):
    g = build_graph(_Sampled(), _renderer(), 0, "q", "Answer: yes", text_mode=mode)
    assert g.nodes[0]["text"].startswith("TARGET ")


# -- truncation ---------------------------------------------------------------

def test_node_cap_below_field_cap_is_rejected():
    """A node cap under the field cap makes `max_value_chars` unreachable: every node is cut
    before one field can spend its budget. That is how 95.5% of rel-trial's `studies` rows
    got truncated while the field cap looked like the knob that mattered."""
    from src.experiments.relbench.config import RunConfig
    with pytest.raises(ValueError, match="unreachable"):
        RunConfig(max_node_chars=100, max_value_chars=200).validate()


def test_no_node_cap_by_default():
    from src.experiments.relbench.config import RunConfig
    assert RunConfig().validate().max_node_chars is None


def test_uncapped_renderer_does_not_truncate():
    long_db = _DB()
    long_db.table_dict["drivers"].df.loc[0, "surname"] = "x" * 5000
    assert not RowRenderer(long_db, max_node_chars=None,
                           max_value_chars=5000).render("drivers", 0, 0).endswith("…")
    assert RowRenderer(long_db, max_node_chars=600,
                       max_value_chars=600).render("drivers", 0, 0).endswith("…")

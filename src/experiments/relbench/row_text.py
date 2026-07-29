"""One database row -> the text of one graph node.

This is the surface with the biggest lever on final numbers and the least theory behind it
(kgqa's data-format v3 was worth +4.6 F1 over v2 on nothing but serialization choices), so
every knob here is part of the cache key and is meant to be ablated rather than argued about.

Column selection is **derived from the schema**, never written by hand (PLAN.md 5.0 A):

* drop the primary key -- an id with no meaning outside the database;
* drop every foreign key -- the edges carry that information structurally, and leaving raw
  ids in the text both wastes tokens and invites the model to memorize entity ids, which on
  a temporal split is pure overfitting;
* drop columns that are almost entirely null -- nothing to say in most rows;
* render the time column specially (relative to the seed), never as a raw value;
* keep everything else.

**A rule that was specified and then rejected:** PLAN.md 6.1 also called for dropping columns
whose cardinality equals the row count, as "free-form ids". Measured against rel-f1 that rule
flags `circuits.name`, `constructors.name` and `circuits.lat`/`lng` -- real signal, and in the
case of the names *precisely* the memorizable content that the `anonymize` arm (PLAN.md 8.6)
exists to control deliberately rather than delete by accident. What it was meant to catch is
slug duplicates like `driverRef`, worth a handful of tokens. Wrong trade; the rule is not
implemented. `max_value_chars` handles the free-text-blob case it was also aimed at.
"""

import hashlib

import numpy as np
import pandas as pd

# Strings that are unambiguously "no value" and never a real category, so they are dropped at
# the *value* level. `\N` is MySQL's null marker and covers 88% of `rel-f1 drivers.code`.
_EMPTY = {"", "nan", "NaN", "NaT", "null", "NULL", "<NA>", r"\N", "\\N"}

# Strings that *usually* mean "no value" but can legitimately be a category -- "None" is the
# open-label answer in `rel-trial designs.masking`, present in 20% of rows and meaningful.
# These are counted toward a column's missing fraction (a column that is 100% "None" carries
# nothing however you read it) but are still rendered where the column survives, because
# discarding a real category is worse than spending four tokens.
_PLACEHOLDER = _EMPTY | {"None", "N/A", "NA", "Not Applicable", "unknown", "Unknown"}

DAY = 86_400


def _missing_fraction(series):
    """Fraction of rows carrying no value, counting placeholder strings as missing.

    `pandas.isna` is not enough. rel-trial encodes nulls as the literal string `"None"`
    throughout: `studies.is_ppsd` and `studies.fdaaa801_violation` are 100% `"None"`,
    `is_unapproved_device` 99.5%, `baseline_type_units_analyzed` 99.9%. All of them look
    completely populated to `isna()` and would otherwise render in every document forever.
    """
    if series.dtype == object:
        as_str = series.astype(str).str.strip()
        return float((series.isna() | as_str.isin(_PLACEHOLDER)).mean())
    return float(series.isna().mean())


def build_column_spec(db, null_threshold=0.95):
    """`{table: [renderable columns]}`, derived from the schema alone.

    `null_threshold` drops columns missing in more than this fraction of rows. Because missing
    values are omitted from the output anyway, a mostly-empty column costs tokens only in the
    rows where it is present, so this is a noise filter rather than a budget one -- keep it
    loose.
    """
    spec = {}
    for name, table in db.table_dict.items():
        structural = {table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table}
        structural.discard(None)
        cols = [col for col in table.df.columns
                if col not in structural
                and _missing_fraction(table.df[col]) <= null_threshold]
        spec[name] = cols
    return spec


def anonymizable_columns(db, cardinality_ratio=0.5):
    """Columns the `anonymize: entities` arm replaces with stable per-row tokens.

    The contamination control (PLAN.md 8.6) targets *identities* -- the names a pretrained
    model can recognize -- while leaving topology and every numeric field untouched, so the
    node count and graph shape are identical across arms and only memorizable content moves.

    Rule: string columns on **static** tables (no `time_col`, i.e. dimension tables) whose
    cardinality exceeds `cardinality_ratio` of the row count. On rel-f1 that selects
    `drivers.forename/surname/driverRef/code`, `circuits.name/circuitRef/location` and
    `constructors.name/constructorRef` while leaving `nationality` (42 values over 857 rows)
    and `country` (35 over 77) alone -- low-cardinality categoricals are attributes, not
    identities.

    **This rule does not transfer to rel-trial and the arm is rel-f1-only in phase 1.** There
    it would also hash `conditions.mesh_term` and `interventions.mesh_term`, which are the
    medical condition and the treatment -- the signal the label depends on, not trivia a model
    might have memorized. Contamination is not a live concern there anyway: a frozen 1B scores
    55.72 on `study-outcome` against 88.47 on rel-f1 `driver-top3` (PLAN.md 1.1).
    """
    out = {}
    for name, table in db.table_dict.items():
        if table.time_col is not None:
            continue
        n = len(table.df)
        cols = [c for c in table.df.columns
                if table.df[c].dtype == object
                and table.df[c].nunique(dropna=True) > cardinality_ratio * n]
        if cols:
            out[name] = cols
    return out


def _format_value(value, max_value_chars):
    """One cell -> a short string, or None if it carries nothing."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        # %.4g keeps 4 significant figures and drops trailing zeros, so 6.0 -> 6 and
        # 1.23456e6 -> 1.235e+06 without a per-column format decision.
        return f"{value:.4g}"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return f"{value:%Y-%m-%d}"

    text = str(value).strip()
    if text in _EMPTY:
        return None
    # Postgres dumps booleans as 't'/'f' (rel-trial's `has_dmc`, `adult`, `child`,
    # `subject_masked`, ...). One character is not a word a language model has any prior over.
    if text in ("t", "f"):
        return "true" if text == "t" else "false"
    # `~` is the ClinicalTrials.gov line separator inside `eligibilities.criteria`; left as-is
    # it reads as noise mid-sentence.
    text = text.replace("~", "; ")
    text = " ".join(text.split())            # collapse newlines/runs of whitespace
    if max_value_chars and len(text) > max_value_chars:
        text = text[:max_value_chars].rstrip() + "…"
    return text


def _format_time(value, seed_ts, time_encoding):
    """The time column, rendered against the seed timestamp.

    `relative` is the default and the interesting arm: an offset from the prediction date is
    what RDL gets for free and a serialized document does not, and absolute dates additionally
    let the model memorize eras -- which a temporal split punishes (PLAN.md 6.1).
    """
    if value is None or value is pd.NaT or pd.isna(value):
        return None
    absolute = f"{value:%Y-%m-%d}"
    days = int((seed_ts - int(value.timestamp())) // DAY)
    # Negative would mean a future row, which the sampler forbids; surface it loudly rather
    # than rendering "-3d before".
    relative = f"{days}d before" if days >= 0 else f"FUTURE+{-days}d"
    if time_encoding == "absolute":
        return absolute
    if time_encoding == "relative":
        return relative
    return f"{absolute} ({relative})"


# Document strategies. `key_value` repeats the column name on every row; the other two hoist
# the schema into a per-table header node and render rows as positional values (PLAN.md 6.2).
TEXT_MODES = ("key_value", "schema_node", "shortest")

# Stands in for a field a row does not populate. Required, not cosmetic: `key_value` can drop
# a null field because its neighbours are self-labelling, but a positional row cannot -- the
# header promises column i is at slot i, and a dropped field silently shifts every later
# value into the wrong column. One character, and it tokenizes as one token.
NULL_SLOT = "-"


class RowRenderer:
    """Renders `(table, row)` pairs against a fixed configuration.

    Holds the derived column spec and per-table frames so a dump or a dataset build does not
    re-derive them per row.
    """

    def __init__(self, db, text_mode="key_value", time_encoding="relative",
                 anonymize="none", max_value_chars=200, max_node_chars=600,
                 null_threshold=0.95, column_spec=None):
        if text_mode not in TEXT_MODES:
            raise ValueError(f"unknown text_mode {text_mode!r} (expected {TEXT_MODES})")
        if time_encoding not in ("relative", "absolute", "both"):
            raise ValueError(f"unknown time_encoding {time_encoding!r}")
        if anonymize not in ("none", "entities", "all"):
            raise ValueError(f"unknown anonymize {anonymize!r}")

        self.db = db
        self.text_mode = text_mode
        # `anonymize: all` additionally strips absolute dates, removing the era cue that lets
        # a model date a row from its content rather than reason about it.
        self.time_encoding = "relative" if anonymize == "all" else time_encoding
        self.anonymize = anonymize
        self.max_value_chars = max_value_chars
        self.max_node_chars = max_node_chars
        self.spec = column_spec or build_column_spec(db, null_threshold)
        self.hidden = anonymizable_columns(db) if anonymize != "none" else {}
        self._frames = {name: t.df for name, t in db.table_dict.items()}
        self._time_cols = {name: t.time_col for name, t in db.table_dict.items()}

    def column_report(self):
        """What the derivation kept and dropped, per table -- for the run record."""
        report = {}
        for name, table in self.db.table_dict.items():
            structural = {table.pkey_col, table.time_col, *table.fkey_col_to_pkey_table}
            structural.discard(None)
            kept = self.spec[name]
            report[name] = {
                "kept": kept,
                "dropped_structural": sorted(structural),
                "dropped_null": [c for c in table.df.columns
                                 if c not in structural and c not in kept],
                "anonymized": self.hidden.get(name, []),
            }
        return report

    def render(self, table, row, seed_ts, aligned=False):
        """One row -> one node's text.

        ``aligned=False`` labels every field (``grid: 13``) and omits the ones the row does
        not populate. ``aligned=True`` emits values only, in column order, with
        :data:`NULL_SLOT` for the missing ones, so the row lines up against the header
        produced by :meth:`header`. The two are not interchangeable: an aligned row is
        unreadable without its header.
        """
        frame = self._frames[table]
        series = frame.iloc[row]
        hidden = self.hidden.get(table, ())

        parts = []
        time_col = self._time_cols[table]
        if time_col is not None:
            stamp = _format_time(series[time_col], seed_ts, self.time_encoding)
            if aligned:
                parts.append(stamp or NULL_SLOT)
            elif stamp:
                parts.append(f"date: {stamp}")

        for col in self.spec[table]:
            if col in hidden:
                # Replaced, not deleted: the field still occupies its slot so node count and
                # token count barely move between arms, and only the identity is gone.
                parts.append(f"{table}_{row}" if aligned else f"{col}: {table}_{row}")
                continue
            text = _format_value(series[col], self.max_value_chars)
            if text is None:
                if aligned:
                    parts.append(NULL_SLOT)
                continue
            parts.append(text if aligned else f"{col}: {text}")

        body = " | ".join(parts)
        # An aligned row carries no table name: the header node already names the table, and
        # repeating it per row would give back part of what the scheme is saving.
        out = body if aligned else (f"{table} | {body}" if body else table)
        if self.max_node_chars and len(out) > self.max_node_chars:
            out = out[:self.max_node_chars].rstrip() + "…"
        return out

    def header(self, table):
        """The schema line for a table: what column each slot of an aligned row holds."""
        cols = []
        if self._time_cols[table] is not None:
            cols.append("date")
        cols.extend(self.spec[table])
        return f"TABLE {table} | " + " | ".join(cols)

    def cheaper_as_schema_node(self, table, rows, seed_ts):
        """Would a header plus aligned rows be shorter than labelling every row?

        Compared in characters, on the rows actually sampled -- so the answer adapts to how
        many rows this table contributed and how sparsely they are populated, which are the
        two things that decide it (PLAN.md 6.2). Deterministic given the data, so it does not
        make the cache ambiguous.
        """
        labelled = sum(len(self.render(table, r, seed_ts)) for r in rows)
        hoisted = len(self.header(table)) + sum(
            len(self.render(table, r, seed_ts, aligned=True)) for r in rows)
        return hoisted < labelled


def stable_token(*parts):
    """Short deterministic token, for anonymization schemes that need one across tables."""
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return digest[:8]

"""
Sweep config loading + expansion.

A sweep config is a JSONC document (JSON + ``//`` / ``/* */`` comments and
trailing commas, so templates can be self-documenting). It carries two kinds of
content:

  * reserved keys the runner consumes directly — never swept: ``execution``
    (local/sbatch dispatch), ``name`` (sweep directory), ``results_dir`` (where
    that directory is created);
  * everything else: the **sweep parameters**, whose JSON *shape* decides how
    they expand into individual runs.

Expansion semantics (decided with the user — type-driven, no marker keywords):

    scalar                  ->  fixed: present in every resolved run unchanged.
    list of scalars         ->  a sweep AXIS: one run per element.
    list of lists           ->  a sweep axis whose values are themselves lists
                                (the way a list-valued param such as len_buckets
                                is swept).
    list of objects (dicts) ->  a BUNDLE: a group of params that vary *together*.
                                Each object is one coherent setting; the object's
                                keys are flattened up to the top level of the
                                resolved run (the bundle's own label disappears).

The full run set is the **cartesian product** of every axis (a bundle counts as
one axis whose values are its objects). Fixed scalars are added to every run.

Because bundle keys flatten to the top level, a key may be "owned" by exactly
one place — top-level *or* a single bundle. A key appearing in two places (two
bundles, or a bundle and the top level) is a collision and raises ``SweepError``
at expansion time, rather than silently picking a winner.

The runner is schema-less: it never checks key *names* against any known set, so
arbitrary experiment-specific params pass straight through to the resolved dict.
"""

import itertools
import json

# Keys consumed by the runner itself; pulled out before expansion, never swept.
RESERVED_KEYS = ("name", "execution", "results_dir")


class SweepError(ValueError):
    """A malformed sweep config (collisions, bad bundle shape, …)."""


# ── JSONC loading ────────────────────────────────────────────────────────────
def _strip_jsonc(text):
    """Strip ``//`` line comments, ``/* */`` block comments and trailing commas.

    Comment markers inside string literals are preserved (a naive regex would
    corrupt e.g. a Windows path or a URL), so this walks the text tracking string
    context rather than pattern-matching.
    """
    out = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:        # keep escaped char verbatim
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        # not in a string
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":   # to end of line
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2                             # skip the closing */
        else:
            out.append(c)
            i += 1
    stripped = "".join(out)
    return _drop_trailing_commas(stripped)


def _drop_trailing_commas(text):
    """Remove commas that immediately precede a ``}`` or ``]`` (outside strings)."""
    out = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":      # trailing comma -> drop it
                i += 1
            else:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def loads(text):
    """Parse a JSONC string into a dict."""
    obj = json.loads(_strip_jsonc(text))
    if not isinstance(obj, dict):
        raise SweepError(f"Sweep config must be a JSON object, got {type(obj).__name__}.")
    return obj


def load_config(path):
    """Load a JSONC sweep config file into a dict."""
    with open(path) as f:
        return loads(f.read())


# ── expansion ────────────────────────────────────────────────────────────────
def split_meta(config):
    """Split a config into ``(meta, sweep_params)``.

    ``meta`` holds the reserved runner keys (``name`` / ``execution``);
    ``sweep_params`` is everything else (what gets expanded into runs).
    """
    meta = {k: config[k] for k in RESERVED_KEYS if k in config}
    sweep = {k: v for k, v in config.items() if k not in RESERVED_KEYS}
    return meta, sweep


def _is_bundle(value):
    """A bundle is a non-empty list whose elements are all objects (dicts)."""
    if not isinstance(value, list) or not value:
        return False
    dicts = sum(isinstance(e, dict) for e in value)
    if dicts == 0:
        return False
    if dicts != len(value):
        raise SweepError(
            "A bundle (list of objects) must contain only objects; "
            "found a mix of objects and non-objects in the same list.")
    return True


def _axis_values(key, value):
    """Return this param's list of axis-values, or ``None`` if it is a fixed scalar.

    Bundle objects are returned verbatim (their keys are flattened later); plain
    lists become one value per element; anything else is a single fixed value.
    """
    if _is_bundle(value):
        return list(value)                     # each object is one axis value
    if isinstance(value, list):
        return list(value)                     # scalar-axis / list-of-lists axis
    return None                                # fixed scalar


def _key_owners(sweep):
    """Map every (flattened) key to the param that owns it, raising on collisions.

    A bundle owns the union of keys across its objects; every other param owns
    its own name. Two owners claiming the same key is a collision.
    """
    owners = {}

    def claim(flat_key, owner):
        if flat_key in owners and owners[flat_key] != owner:
            raise SweepError(
                f"Key {flat_key!r} is defined in both {owners[flat_key]!r} and "
                f"{owner!r}; a key may be owned by exactly one place "
                f"(top-level or a single bundle).")
        owners[flat_key] = owner

    for key, value in sweep.items():
        if _is_bundle(value):
            for obj in value:
                for flat_key in obj:
                    claim(flat_key, key)
        else:
            claim(key, key)
    return owners


def expand(sweep):
    """Expand sweep params into a list of flat, fully-resolved run dicts.

    Raises ``SweepError`` on a key collision or malformed bundle.
    """
    _key_owners(sweep)                         # validate collisions up front

    fixed = {}                                 # scalar params, in every run
    axes = []                                  # list of (key, [values...])
    for key, value in sweep.items():
        vals = _axis_values(key, value)
        if vals is None:
            fixed[key] = value
        else:
            axes.append((key, vals))

    runs = []
    axis_keys = [k for k, _ in axes]
    axis_value_lists = [v for _, v in axes]
    for combo in itertools.product(*axis_value_lists) if axes else [()]:
        run = dict(fixed)
        for key, chosen in zip(axis_keys, combo):
            if isinstance(chosen, dict):       # bundle object -> flatten its keys
                run.update(chosen)
            else:
                run[key] = chosen
        runs.append(run)
    return runs


def load_and_expand(path):
    """Convenience: load a config file and return ``(meta, runs)``."""
    meta, sweep = split_meta(load_config(path))
    return meta, expand(sweep)

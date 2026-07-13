"""Raw SR record access + the CWQ int->mid normalization adapter.

Every consumer of raw SR records (graph data prep, the flat arm, the coverage
analysis, the naming-v2 builder) goes through ``load_sr_records(dataset, split)``
and sees ONE record shape — WebQSP's:

    id / question : str
    entities      : [mid, ...]                       (topic entities)
    answers       : [{"kb_id": ..., "text": ...}, ...]
    subgraph      : {"tuples": [[h_mid, rel, t_mid], ...], "entities": [...]}
    paths         : [[root_mid, [rel, ..., "END OF HOP"]], ...]

WebQSP records already have this shape and pass through untouched. CWQ records
(GNN-RAG's ``sr-cwq``) instead carry opaque integers for every tuple endpoint
and path root, plus ``entities_cid``/``answers_cid`` parallels: SR's CWQ
retriever ran on an int-coded Freebase cache whose vocabulary was pickled as
``ent2id.pickle`` (56,382,529 entities; RUCKBReasoning/SubgraphRetrievalKBQA
``src/CWQ/generate_kb_cache/convert_all_triple_to_int.py``). That dict was
built by ``enumerate`` over a Python set, so the order — and hence the mapping —
is irreproducible from source; the decode table

    data/cwq_ent_id2mid.txt        (line number = global id, line = mid)

was extracted once from the authors' pickle (SR Drive folder
1ZHSBxGE-vjcvrfkmJPgcXfme_b2Ni6k6, via SR issue #8) and verified against all
16,960 in-record ``*_cid`` <-> ``kb_id`` pairs (16,959 match; the one miss is
the cid=-1 unlinkable-entity sentinel). GNN-RAG itself trains its GNN entirely
in int space and only decodes at eval output (``gnn/evaluate.py``); we need
node *text* at build time, so we decode here instead.

Only the ~750k ids that actually occur in sr-cwq are needed, so a trimmed
``data/sr-cwq/id2mid.trimmed.json`` is built from the full table on first use
(flock-guarded, like the .gtds builds) and loaded thereafter.
"""

import fcntl
import json
import os

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = ("webqsp", "cwq")
SPLITS = {"train": "train.json", "dev": "dev.json", "test": "test.json"}

CWQ_ID2MID_TABLE = os.path.join(EXPERIMENT_DIR, "data", "cwq_ent_id2mid.txt")
CWQ_ID2MID_TRIMMED = os.path.join(EXPERIMENT_DIR, "data", "sr-cwq", "id2mid.trimmed.json")

# cid -1 = SR's unlinkable-entity sentinel (its kb_id in-record is malformed);
# any other unmapped int would indicate a truncated/wrong decode table.
_UNLINKED = "UNLINKED_ENTITY"


def sr_dir(dataset):
    return os.path.join(EXPERIMENT_DIR, "data", f"sr-{dataset}")


def split_path(dataset, split):
    return os.path.join(sr_dir(dataset), SPLITS[split])


def _iter_raw(dataset, split):
    with open(split_path(dataset, split)) as f:
        for line in f:
            yield json.loads(line)


# --------------------------------------------------------------------------- #
# CWQ int -> mid decode
# --------------------------------------------------------------------------- #
def _collect_cwq_ids():
    """Every global int id occurring anywhere in the sr-cwq splits."""
    ids = set()
    for split in SPLITS:
        for rec in _iter_raw("cwq", split):
            for h, _, t in rec["subgraph"]["tuples"]:
                if isinstance(h, int):
                    ids.add(h)
                if isinstance(t, int):
                    ids.add(t)
            for e in rec["subgraph"].get("entities", []):
                if isinstance(e, int):
                    ids.add(e)
            ids.update(c for c in rec.get("entities_cid", []) if isinstance(c, int))
            ids.update(c for c in rec.get("answers_cid", []) if isinstance(c, int))
            for path in rec.get("paths", []):
                if isinstance(path[0], int):
                    ids.add(path[0])
    return ids


def _build_trimmed_id2mid():
    if not os.path.exists(CWQ_ID2MID_TABLE):
        raise FileNotFoundError(
            f"CWQ decode table missing: {CWQ_ID2MID_TABLE}\n"
            "It is extracted from SR's ent2id.pickle (Drive folder "
            "1ZHSBxGE-vjcvrfkmJPgcXfme_b2Ni6k6, RUCKBReasoning/SubgraphRetrievalKBQA "
            "issue #8): line number = global entity id, line = mid. See sr_records.py."
        )
    needed = _collect_cwq_ids()
    print(f"[sr-cwq] building trimmed id->mid map for {len(needed)} ids ...")
    id2mid = {}
    with open(CWQ_ID2MID_TABLE) as f:
        for i, line in enumerate(f):
            if i in needed:
                id2mid[i] = line.rstrip("\n")
    unmapped = {i for i in needed if i not in id2mid and i >= 0}
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} sr-cwq ids missing from the decode table "
            f"(e.g. {sorted(unmapped)[:5]}) — table truncated or wrong version.")
    with open(CWQ_ID2MID_TRIMMED, "w") as f:
        json.dump({str(k): v for k, v in id2mid.items()}, f)
    print(f"[sr-cwq] wrote {CWQ_ID2MID_TRIMMED} ({len(id2mid)} entries)")
    return id2mid


def load_cwq_id2mid():
    """The trimmed {int id -> mid} map, built from the full table on first use."""
    if os.path.exists(CWQ_ID2MID_TRIMMED):
        return {int(k): v for k, v in json.load(open(CWQ_ID2MID_TRIMMED)).items()}
    with open(CWQ_ID2MID_TRIMMED + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(CWQ_ID2MID_TRIMMED):     # built by a concurrent job
            return {int(k): v for k, v in json.load(open(CWQ_ID2MID_TRIMMED)).items()}
        return _build_trimmed_id2mid()


def _decode(x, id2mid):
    if not isinstance(x, int):
        return x                                   # already a mid / literal string
    return id2mid.get(x, _UNLINKED)                # -1 (and only -1, per build check)


def normalize_cwq_record(rec, id2mid):
    """Reshape one raw sr-cwq record into the WebQSP record shape (see module doc)."""
    out = {k: v for k, v in rec.items() if k not in ("entities_cid", "answers_cid")}
    out["subgraph"] = dict(rec["subgraph"])
    out["subgraph"]["tuples"] = [
        [_decode(h, id2mid), r, _decode(t, id2mid)]
        for h, r, t in rec["subgraph"]["tuples"]]
    # Topic entities: decode entities_cid (same id space as the tuples, so prompt
    # edges attach); the record's own `entities` dicts are the fallback for the
    # cid=-1 sentinel, matching GNN-RAG's skip-if-absent behavior.
    ents = []
    cids = rec.get("entities_cid", [])
    for i, e in enumerate(rec["entities"]):
        kb = e["kb_id"] if isinstance(e, dict) else e
        mid = _decode(cids[i], id2mid) if i < len(cids) else kb
        ents.append(kb if mid == _UNLINKED else mid)
    out["entities"] = ents
    out["paths"] = [[_decode(p[0], id2mid), p[1]] for p in rec.get("paths", [])]
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_sr_records(dataset, split):
    """All records of one split, in the normalized (WebQSP) shape."""
    if dataset == "webqsp":
        return list(_iter_raw(dataset, split))
    if dataset == "cwq":
        id2mid = load_cwq_id2mid()
        return [normalize_cwq_record(r, id2mid) for r in _iter_raw(dataset, split)]
    raise ValueError(f"dataset must be one of {DATASETS}; got {dataset!r}")

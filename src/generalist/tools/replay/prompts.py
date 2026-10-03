"""Select the prompt set `text/replay` generates its targets for (`MOLECULE_GENERALIST.md` §9.1).

    src/generalist/tools/launch/run_py.sh src/generalist/tools/replay/prompts.py --review
    src/generalist/tools/launch/run_py.sh src/generalist/tools/replay/prompts.py --version v1

Reads the raw downloads under ``results/raw/replay/`` and writes
``results/replay/<version>/prompts.jsonl`` plus ``selection.json``, which records
every count and every parameter the selection used. ``--review`` stops after
filtering and writes random samples of each chemistry tier instead, which is how
the chemistry rule's precision is read before anything is generated.

**Sources.** English first user turns only, because a first turn is the one
shape a single-node prompt can carry faithfully:

* OASST2 ready-tree roots (Apache-2.0) — human-written, every one kept.
* Dolly-15k (CC BY-SA 3.0) — every instruction without a context passage, plus
  a capped, category-stratified share of the passage-reading ones. A passage
  answer is short and extractive; left uncapped those categories would be a
  quarter of Dolly and teach the backbone's least typical register.
* WildChat-1M (ODC-BY) — real traffic, and the only source large enough to fill
  the pool. It is also the noisiest, so it carries its own filters.

**Filters on every row**, in this order: overlap with the 48 `text_probes.json`
prompts (exact, or any shared 8-gram — those stay a clean test set); exact
duplicates across all three sources, earliest source wins; under 12 characters;
over :data:`MAX_PROMPT_TOKENS` tokens; and any token that RDKit parses as a
molecule of three or more atoms. The last one is not about quality: the graph
arm must never see a molecule string (§1), and replay is the one source of
training text nobody wrote with that rule in mind.

**WildChat on top.** Mostly non-Latin text under an English tag; jailbreak and
role-play openers, which train the backbone's refusal register rather than its
assistant one; image-generator prompt templates; and template floods — hundreds
of prompts differing only in a filled-in slot — capped at one per 60-character
opening. Pasted-text edits ("rewrite this paragraph: …") are kept but capped at
:data:`PASTED_EDIT_SHARE` of the WildChat sample, because they are a large share
of real traffic and a small share of what the backbone is *for*.

**The chemistry slice.** The nearest neighbour of the caption failure (§7.6) is
chemistry in prose, so it is a tagged slice with its own sample count. Detection
is a keyword rule in two tiers — ``strict`` needs two chemistry cues, ``loose``
one strong cue — with exclusions for the two false-positive shapes the census
found: fiction (invented elements, alien planets) and pasted-text edits of a
chemistry paper, which are editing requests and not chemistry questions.

**Only ``strict`` is the slice, sampled twice as often.** Hand-checked on 60
random prompts per tier and source (job 153770, `--review`): ``strict`` is about
80 % chemistry on WildChat and two-thirds on Dolly, where biology and nutrition
passages share its cues; ``loose`` is about half, the rest being weather
forecasts, caffeine merchandise and superhero power lists. Loosening therefore
buys roughly 600 real chemistry prompts at the price of 600 that are not, and
``strict`` alone (1,316 before the template cap, ~1,050 of them chemistry) cannot
carry 5 % of a 35k pass at one draw each. So each ``strict`` prompt gets
``2 × answers`` samples and weight 2. ``loose`` and ``excluded`` rows are
ordinary general prompts.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import random
import re
import sys
from multiprocessing import Pool

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO)

RAW_DIR = os.path.join(REPO, "src", "generalist", "results", "raw", "replay")
OUT_ROOT = os.path.join(REPO, "src", "generalist", "results", "replay")
PROBES_PATH = os.path.join(REPO, "src", "generalist", "evaluate", "text_probes.json")
TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"

#: The prompt half of the text task's 1,024-token node; the answer gets 768.
MAX_PROMPT_TOKENS = 256
MIN_CHARS = 12
#: Share of a WildChat sample that may be an edit of pasted text.
PASTED_EDIT_SHARE = 0.05


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def prompt_id(text: str) -> str:
    """The partition key: stable across processes, blind to whitespace and case."""
    return hashlib.sha256(norm(text).encode()).hexdigest()[:16]


def ngrams(text: str, n: int = 8) -> set:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_oasst(raw: str) -> list:
    path = os.path.join(raw, "oasst2", "2023-11-05_oasst2_ready.trees.jsonl.gz")
    out = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            prompt = json.loads(line)["prompt"]
            if prompt.get("lang") == "en":
                out.append({"source": "oasst2", "sub": "root", "text": prompt["text"]})
    return out


def load_dolly(raw: str) -> list:
    out = []
    with open(os.path.join(raw, "dolly", "databricks-dolly-15k.jsonl")) as fh:
        for line in fh:
            row = json.loads(line)
            text, context = row["instruction"].strip(), row["context"].strip()
            if context:
                text = f"{text}\n\n{context}"
            out.append({"source": "dolly", "sub": row["category"], "text": text,
                        "context": bool(context)})
    return out


def load_wildchat(raw: str, stats: collections.Counter) -> list:
    import pyarrow.parquet as pq

    out = []
    for path in sorted(glob.glob(os.path.join(raw, "wildchat", "*.parquet"))):
        table = pq.read_table(path, columns=["conversation", "language", "toxic",
                                             "redacted", "turn"])
        for row in table.to_pylist():
            stats["rows"] += 1
            if row["language"] != "English":
                continue
            stats["english"] += 1
            if row["toxic"] or row["redacted"]:
                stats["toxic_or_redacted"] += 1
                continue
            first = next((m for m in row["conversation"] if m["role"] == "user"), None)
            text = (first or {}).get("content") or ""
            if text.strip():
                out.append({"source": "wildchat", "sub": f"turn{min(row['turn'], 3)}",
                            "text": text})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────

SMILES_TOKEN = re.compile(r"(?<![\w/])[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{6,}(?![\w/])")


def has_smiles(text: str) -> bool:
    """Any whitespace-delimited token RDKit reads as a molecule of 3+ heavy atoms."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    for token in SMILES_TOKEN.findall(text):
        token = token.rstrip(".")
        if not re.search(r"[()=\[\]#]|\d", token):
            continue
        if re.fullmatch(r"https?:.*|[\d.,\-+%]+|[A-Za-z]+\d*|\d+[A-Za-z]*", token):
            continue
        mol = Chem.MolFromSmiles(token)
        if mol is not None and mol.GetNumHeavyAtoms() >= 3:
            return True
    return False


def non_latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > 0x24F) / len(letters)


ROLEPLAY = re.compile(
    r"ignore (all |any )?(the |your )?(previous|prior|above) (instructions|prompts)"
    r"|\bact as (a|an|my|the)\b|\bpretend (to be|you are|you're|that you)"
    r"|\bjailbreak|\bdeveloper mode\b|\bDAN\b|\bstay in character\b"
    r"|\brole-?play\b|from now on,? you (are|will|must)"
    r"|\byou are (now )?going to (act|pretend)",
    re.I)

IMAGE_PROMPTS = re.compile(
    r"midjourney|stable diffusion|/imagine|\bprompt generator\b"
    r"|as a prompt generator|dall-?e", re.I)

EDIT_VERBS = re.compile(
    r"\b(rewrite|re-write|paraphrase|proofread|polish|rephrase|reword|edit|"
    r"improve|correct (the )?grammar|make it (more )?(academic|formal|concise)|"
    r"summari[sz]e|translate)\b", re.I)

CODE = re.compile(r"```|\bdef \w+\(|#include|\bfunction\s*\(|;\s*$|</?\w+>|\bimport \w+",
                  re.M)


def is_pasted_edit(text: str, tokens: int) -> bool:
    """An editing request over pasted text: a verb near the top, and a long body."""
    return tokens > 120 and bool(EDIT_VERBS.search(text[:300]))


# ─────────────────────────────────────────────────────────────────────────────
# The chemistry rule
# ─────────────────────────────────────────────────────────────────────────────

#: Near-unambiguous chemistry cues. One is enough under the ``loose`` rule.
STRONG = [
    r"\bchemi(cal|cals|stry|st|sts)\b", r"\bmolecul(e|es|ar)\b", r"\bsolvents?\b",
    r"\bcatalys(t|ts|is)\b", r"\boxidi[sz](e|ed|es|ing|ation)\b", r"\boxidation\b",
    r"\besters?\b", r"\balk(ane|ene|yne)s?\b", r"\baromatic (ring|compound|hydrocarbon)",
    r"\bbenzene\b", r"\bethanol\b", r"\bmethanol\b", r"\bmethane\b",
    r"\bpolymers?\b", r"\bpolymeri[sz]", r"\btitrat", r"\bstoichiometr",
    r"\bmolarity\b", r"\bmolar mass\b", r"\bisotopes?\b", r"\bcovalent\b",
    r"\bionic bond", r"\bperiodic table\b", r"\bpka\b", r"\bcaffeine\b",
    r"\baspirin\b", r"\bibuprofen\b", r"\bparacetamol\b", r"\bacetaminophen\b",
    r"\bacetone\b", r"\bammonia\b", r"\bsulfuric\b", r"\bsulphuric\b",
    r"\bhydrochloric\b", r"\bchlorine\b", r"\bsodium\b", r"\bnitrogen\b",
    r"\bhydrogen\b", r"\bcarbon dioxide\b", r"\bchiral(ity)?\b",
    r"\benantiomers?\b", r"\bisomers?\b", r"\bfunctional groups?\b",
    r"\bpharmacolog", r"\bdrug design\b", r"\bamino acids?\b", r"\bglucose\b",
    r"\bph (of|level|value|scale)\b", r"\bacidic\b", r"\balkaline\b",
    r"\bprecipitat", r"\bdistillation\b", r"\bredox\b", r"\bexothermic\b",
    r"\bendothermic\b", r"\bnucleophil", r"\belectrophil", r"\bs[nN][12]\b",
    r"\benthalpy\b", r"\bhydrocarbons?\b", r"\bchromatograph", r"\bspectroscop",
    r"\bnmr\b", r"\bmass spectromet", r"\bsolubility\b", r"\belectrolysis\b",
    r"\bcombustion\b", r"\bavogadro", r"\bideal gas law\b", r"\ble chatelier",
    r"\bequilibrium constant\b", r"\breagents?\b", r"\bnicotine\b",
    r"\bethylene\b", r"\bpropane\b", r"\bbutane\b", r"\bozone\b",
    r"\bnitrates?\b", r"\bsulfates?\b", r"\bcarbonates?\b", r"\bhydroxides?\b",
    r"\bperoxides?\b", r"\bchlorides?\b", r"\bfluorides?\b", r"\blithium\b",
    r"\bpotassium\b", r"\bmagnesium\b", r"\bcarcinogen", r"\bpesticides?\b",
    r"\bpfas\b", r"\bbicarbonate\b", r"\bvalence\b", r"\belectronegativ",
    r"\borganic chemistry\b", r"\bbiochemi", r"\bmolecular (formula|weight|structure)",
    r"\bsynthesi[sz](e|ed|ing)? (of|the|a)\b", r"\breaction (mechanism|rate|yield)",
    r"\bmoles? of\b", r"\bgrams? per mole\b", r"\belectron configuration",
]
#: Chemistry cues that are ordinary English on their own ("compound interest",
#: "a reaction to the news", "base case"). They only count beside a second cue.
WEAK = [r"\bcompounds?\b", r"\breactions?\b", r"\bacids?\b", r"\bbases?\b",
        r"\belements?\b", r"\borganic\b", r"\bbonds?\b", r"\batoms?\b",
        r"\bdrugs?\b", r"\btoxic\b", r"\bsalts?\b", r"\bmetals?\b",
        r"\bions?\b", r"\belectrons?\b", r"\bproteins?\b", r"\benzymes?\b",
        r"\bsolutions?\b", r"\bconcentration\b", r"\bdissolv", r"\bvitamins?\b",
        r"\bcarbon\b", r"\boxygen\b", r"\bplastics?\b", r"\blab(oratory)?\b",
        r"\bexperiment", r"\bformula\b"]
STRONG_RE = [re.compile(p, re.I) for p in STRONG]
WEAK_RE = [re.compile(p, re.I) for p in WEAK]

#: The census's false-positive shapes. A hit that matches one of these is not a
#: chemistry question, whatever vocabulary it uses.
FICTION = re.compile(
    r"\b(story|stories|fictional|fantasy|sci-?fi|planet|alien|species|"
    r"character|novel|anime|pokemon|minecraft|video game|lore|kingdom|dialogue|"
    r"screenplay|episode|fanfic|worldbuilding|civilization|superhero|villain)\b",
    re.I)


def chemistry_tier(text: str, tokens: int) -> str:
    """``"strict"``, ``"loose"``, ``"excluded"`` or ``""`` (not chemistry)."""
    strong = sum(1 for r in STRONG_RE if r.search(text))
    weak = sum(1 for r in WEAK_RE if r.search(text))
    if strong >= 2 or (strong >= 1 and strong + weak >= 2):
        tier = "strict"
    elif strong >= 1:
        tier = "loose"
    else:
        return ""
    if FICTION.search(text) or is_pasted_edit(text, tokens):
        return "excluded"
    return tier


# ─────────────────────────────────────────────────────────────────────────────
# Per-row work, parallel
# ─────────────────────────────────────────────────────────────────────────────

_TOKENIZER = None


def _init_worker():
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER)


def _measure(rows: list) -> list:
    """Tokens, SMILES, code, chemistry tier and the WildChat flags, for a chunk."""
    texts = [r["text"].strip() for r in rows]
    ids = _TOKENIZER(texts, add_special_tokens=False)["input_ids"]
    out = []
    for row, text, tok in zip(rows, texts, ids):
        n = len(tok)
        row = dict(row, text=text, prompt_tokens=n)
        if n <= MAX_PROMPT_TOKENS:
            row["smiles"] = has_smiles(text)
            row["code"] = bool(CODE.search(text))
            row["chem"] = chemistry_tier(text, n)
            row["pasted_edit"] = is_pasted_edit(text, n)
            if row["source"] == "wildchat":
                row["non_latin"] = non_latin_share(text) > 0.10
                row["roleplay"] = bool(ROLEPLAY.search(text[:600]))
                row["image_prompt"] = bool(IMAGE_PROMPTS.search(text))
        out.append(row)
    return out


def filter_rows(sources: dict, workers: int, drops: dict) -> dict:
    """Probe overlap, dedup and length on the main process; the rest in a pool."""
    probes = json.load(open(PROBES_PATH))["prompts"]
    probe_exact = {norm(p["text"]) for p in probes}
    probe_grams = set().union(*(ngrams(p["text"]) for p in probes))

    seen = set()
    staged = {}
    for name in ("oasst2", "dolly", "wildchat"):
        d = drops.setdefault(name, collections.Counter())
        keep = []
        for row in sources[name]:
            text = row["text"].strip()
            key = norm(text)
            if key in probe_exact or ngrams(text) & probe_grams:
                d["probe_overlap"] += 1
                continue
            if key in seen:
                d["exact_dup"] += 1
                continue
            seen.add(key)
            if len(text) < MIN_CHARS:
                d["too_short"] += 1
                continue
            keep.append(row)
        staged[name] = keep

    out = {}
    with Pool(workers, initializer=_init_worker) as pool:
        for name, rows in staged.items():
            d = drops[name]
            chunks = [rows[i:i + 2000] for i in range(0, len(rows), 2000)]
            measured = [r for chunk in pool.imap(_measure, chunks) for r in chunk]
            keep = []
            for row in measured:
                if row["prompt_tokens"] > MAX_PROMPT_TOKENS:
                    d[f"prompt_over_{MAX_PROMPT_TOKENS}_tokens"] += 1
                elif row["smiles"]:
                    d["smiles_in_prompt"] += 1
                elif row.get("non_latin"):
                    d["non_latin"] += 1
                elif row.get("roleplay"):
                    d["roleplay_or_jailbreak"] += 1
                elif row.get("image_prompt"):
                    d["image_prompt_template"] += 1
                else:
                    keep.append(row)
            out[name] = keep
    return out


def cap_templates(rows: list, drops: collections.Counter, seed: int) -> list:
    """One prompt per 60-character opening, chosen at random, not first-come."""
    rng = random.Random(f"templates|{seed}")
    order = list(range(len(rows)))
    rng.shuffle(order)
    seen, keep = set(), []
    for i in order:
        opening = norm(rows[i]["text"])[:60]
        if opening in seen:
            drops["template_flood"] += 1
            continue
        seen.add(opening)
        keep.append(rows[i])
    keep.sort(key=lambda r: prompt_id(r["text"]))
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Review and selection
# ─────────────────────────────────────────────────────────────────────────────

def write_review(rows_by_source: dict, out_dir: str, seed: int, per: int = 60) -> dict:
    """Random samples of every (tier, source) cell, for reading precision by hand."""
    rng = random.Random(f"review|{seed}")
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for tier in ("strict", "loose", "excluded"):
        for name, rows in rows_by_source.items():
            hits = [r for r in rows if r["chem"] == tier]
            counts[f"{tier}/{name}"] = len(hits)
            sample = rng.sample(hits, min(per, len(hits)))
            path = os.path.join(out_dir, f"{tier}_{name}.txt")
            with open(path, "w") as fh:
                for i, row in enumerate(sample):
                    body = row["text"][:400].replace("\n", " ")
                    fh.write(f"[{i:02d}] ({row['sub']}, {row['prompt_tokens']} tok) "
                             f"{body}\n")
    return counts


def select(rows_by_source: dict, args, drops: dict) -> tuple:
    """The pool: every row a rule admits, then WildChat sampled to the target."""
    rng = random.Random(f"select|{args.seed}")
    chem_tiers = set(args.chem_tiers.split(","))

    oasst = list(rows_by_source["oasst2"])
    dolly_plain = [r for r in rows_by_source["dolly"] if not r["context"]]
    dolly_context = [r for r in rows_by_source["dolly"] if r["context"]]
    by_category = collections.defaultdict(list)
    for row in dolly_context:
        by_category[row["sub"]].append(row)
    per_category = args.dolly_context // max(len(by_category), 1)
    dolly_capped = []
    for category in sorted(by_category):
        rows = by_category[category]
        rng.shuffle(rows)
        dolly_capped.extend(rows[:per_category])
    drops["dolly"]["context_cap"] = len(dolly_context) - len(dolly_capped)

    wild = cap_templates(rows_by_source["wildchat"], drops["wildchat"], args.seed)
    wild_chem = [r for r in wild if r["chem"] in chem_tiers]
    wild_rest = [r for r in wild if r["chem"] not in chem_tiers]
    rng.shuffle(wild_rest)

    fixed = oasst + dolly_plain + dolly_capped + wild_chem
    n_chem = sum(1 for r in fixed if r["chem"] in chem_tiers)
    chem_weight = 1 if n_chem >= args.chem_prompts_for_weight_one else 2
    fixed_examples = sum(chem_weight if r["chem"] in chem_tiers else 1 for r in fixed)
    want_wild = max(0, int(args.train_examples * args.overselect) + args.val
                    - fixed_examples)

    edits_cap = int(PASTED_EDIT_SHARE * want_wild)
    wild_take, edits = [], 0
    for row in wild_rest:
        if len(wild_take) >= want_wild:
            break
        if row["pasted_edit"]:
            if edits >= edits_cap:
                drops["wildchat"]["pasted_edit_cap"] += 1
                continue
            edits += 1
        wild_take.append(row)

    pool = fixed + wild_take
    pool.sort(key=lambda r: prompt_id(r["text"]))
    # The val split never carries chemistry: the slice is small, and the val loss
    # is a curve for the general register rather than a chemistry read.
    general = [r for r in pool if r["chem"] not in chem_tiers]
    val_ids = {prompt_id(r["text"]) for r in rng.sample(general, args.val)}

    out = []
    for row in pool:
        pid = prompt_id(row["text"])
        chem = row["chem"] in chem_tiers
        role = "val" if pid in val_ids else "train"
        samples = 1 if role == "val" else (
            args.answers * (chem_weight if chem else 1))
        out.append({
            "id": pid, "source": row["source"], "sub": row["sub"],
            "text": row["text"], "prompt_tokens": row["prompt_tokens"],
            "chem": chem, "chem_tier": row["chem"], "code": row["code"],
            "role": role, "samples": samples,
            "weight": chem_weight if (chem and role == "train") else 1,
        })
    return out, {"chem_weight": chem_weight, "chem_prompts": n_chem,
                 "wildchat_after_template_cap": len(wild),
                 "wildchat_taken": len(wild_take), "pasted_edits_taken": edits}


def summarise(pool: list) -> dict:
    train = [r for r in pool if r["role"] == "train"]
    by = lambda key: dict(collections.Counter(key(r) for r in train))  # noqa: E731
    examples = sum(r["weight"] for r in train)
    chem_examples = sum(r["weight"] for r in train if r["chem"])
    return {
        "prompts": len(pool), "train_prompts": len(train),
        "val_prompts": len(pool) - len(train),
        "train_examples_per_pass": examples,
        "chem_prompts": sum(1 for r in train if r["chem"]),
        "chem_share_of_examples": chem_examples / max(examples, 1),
        "samples_to_generate": sum(r["samples"] for r in pool),
        "by_source": by(lambda r: r["source"]),
        "by_source_sub": by(lambda r: f"{r['source']}/{r['sub']}"),
        "chem_by_source": dict(collections.Counter(
            r["source"] for r in train if r["chem"])),
        "code_like": sum(1 for r in train if r["code"]),
        "prompt_tokens_mean": sum(r["prompt_tokens"] for r in train) / max(len(train), 1),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw", default=RAW_DIR)
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--version", default="v1")
    p.add_argument("--review", action="store_true",
                   help="write chemistry review samples and counts, then stop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--train-examples", type=int, default=35000,
                   help="examples per replay pass the pool should carry")
    p.add_argument("--overselect", type=float, default=1.15,
                   help="headroom for prompts whose every sample fails to terminate")
    p.add_argument("--val", type=int, default=500)
    p.add_argument("--dolly-context", type=int, default=1500)
    p.add_argument("--answers", type=int, default=3)
    p.add_argument("--chem-tiers", default="strict")
    p.add_argument("--chem-prompts-for-weight-one", type=int, default=1750,
                   help="below this many chemistry prompts, each is sampled twice "
                        "as often (option b): 5 %% of a 35k pass is 1,750")
    args = p.parse_args(argv)

    stats = collections.Counter()
    sources = {"oasst2": load_oasst(args.raw), "dolly": load_dolly(args.raw),
               "wildchat": load_wildchat(args.raw, stats)}
    print("wildchat raw:", dict(stats), flush=True)
    drops: dict = {}
    rows = filter_rows(sources, args.workers, drops)
    for name in rows:
        print(f"{name}: {len(sources[name])} raw -> {len(rows[name])} kept; "
              f"drops {dict(drops[name])}", flush=True)

    out_dir = os.path.join(args.out_root, args.version)
    if args.review:
        counts = write_review(rows, os.path.join(out_dir, "review"), args.seed)
        print(json.dumps(counts, indent=1))
        with open(os.path.join(out_dir, "review", "counts.json"), "w") as fh:
            json.dump({"counts": counts, "drops": {k: dict(v) for k, v in drops.items()},
                       "wildchat_raw": dict(stats)}, fh, indent=1)
        return 0

    pool, meta = select(rows, args, drops)
    summary = summarise(pool)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prompts.jsonl"), "w") as fh:
        for row in pool:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    record = {"version": args.version, "args": vars(args), "wildchat_raw": dict(stats),
              "drops": {k: dict(v) for k, v in drops.items()}, "selection": meta,
              "summary": summary, "tokenizer": TOKENIZER,
              "max_prompt_tokens": MAX_PROMPT_TOKENS}
    with open(os.path.join(out_dir, "selection.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print(json.dumps({"selection": meta, "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

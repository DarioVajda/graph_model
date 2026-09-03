"""
T8 — the evaluation plugin system (`src/generalist/evaluate/`, DESIGN.md §D7).

What is pinned here is the *plumbing*, in the sense that D7 means it: a metric
that appears under a name nobody declared, a validator that takes a run down with
it, a cadence that fires on the wrong step and a selection made on test are all
failures of the harness rather than of a model, and all four are cheap to make.
So:

* every built-in declares its keys and returns **exactly** them — the test walks
  the registry, so a validator added later is covered without editing this file;
* a validator that raises is logged and skipped, and the others' metrics survive
  intact (the `_per_example` contract: measurement never loses a run that already
  cost GPU-hours);
* an undeclared key fails that validator and drops all of its metrics;
* an unmet ``needs`` names the field, rather than surfacing as an ``AttributeError``
  from three frames inside a metric;
* ``should_run`` fires on each cadence form and on nothing else;
* a selection key naming ``test`` is refused;
* the scorers are right on cases computed by hand.

The model is a stub: an embedding and a head, no attention, with one
``graph_bias`` parameter so the bias-norm and adapters-off paths have something
real to read. That is deliberate — the built-ins' *arithmetic* is exercised
against the real batching, the real collator and the real metric implementations,
while the thing a tiny transformer would add (attention) is not what any of these
validators measure. The GPU smoke run (T10) is where they meet a real model and a
real dataset.
"""

import math

import pytest
import torch

from src.generalist import evaluate as ev
from src.generalist.evaluate import builtin
from src.generalist.registry import Registry, TaskSpec, resolve
from src.generalist.schema import Example, render
from src.utils.text_graph_collator_v2 import GraphCollatorV2

# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────

YES, NO = " Yes", " No"
YES_ID, NO_ID = 1000, 1001
VOCAB = 1100


class FakeTokenizer:
    """One token per character, with the two label words as single tokens.

    The label words *must* be single tokens or the margin readout compares first
    tokens rather than classes — `answer_token_ids` asserts it, and this
    tokenizer is what makes that assertion pass here without touching the HF
    cache.
    """

    eos_token_id = 0
    pad_token_id = 0
    name_or_path = "fake"

    def _ids(self, text):
        ids, i = [], 0
        while i < len(text):
            if text.startswith(YES, i):
                ids.append(YES_ID)
                i += len(YES)
            elif text.startswith(NO, i):
                ids.append(NO_ID)
                i += len(NO)
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def __call__(self, texts, padding=False, truncation=True, max_length=512,
                 add_special_tokens=False, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = [self._ids(t)[:max_length] for t in texts]
        if return_tensors != "pt":
            return {"input_ids": ids}
        width = max(len(x) for x in ids)
        return {
            "input_ids": torch.tensor([x + [0] * (width - len(x)) for x in ids]),
            "attention_mask": torch.tensor(
                [[1] * len(x) + [0] * (width - len(x)) for x in ids]),
        }

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in [int(x) for x in ids]:
            if i == YES_ID:
                out.append(YES)
            elif i == NO_ID:
                out.append(NO)
            elif i == 0 and skip_special_tokens:
                continue
            else:
                out.append(chr(i))
        return "".join(out)


class StubLM(torch.nn.Module):
    """Logits from the token alone, plus a graph-bias parameter and a canned generate.

    Deterministic and position-independent, which is what makes the tied-margin
    case below a *hand-computable* one: two prompts that agree up to the scored
    position produce the same margin, which is exactly the tie
    ``tied_pair_fraction`` exists to report.
    """

    def __init__(self, generation=" CCO", seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(VOCAB, 8)
        self.head = torch.nn.Linear(8, VOCAB)
        for p in list(self.emb.parameters()) + list(self.head.parameters()):
            p.requires_grad_(False)
        # The one trainable tensor, named the way the bias channel is named so
        # `bias_norm` and `base_exact` see what they would see on a real run.
        self.graph_bias_weights = torch.nn.Parameter(torch.full((4, 8), 0.25))
        self.generation = generation
        self.tokenizer = FakeTokenizer()

    def forward(self, input_ids=None, **kwargs):
        from transformers.modeling_outputs import CausalLMOutput

        return CausalLMOutput(logits=self.head(self.emb(input_ids)))

    def generate(self, input_ids=None, max_new_tokens=16, **kwargs):
        new = self.tokenizer.encode(self.generation)[:max_new_tokens]
        tail = torch.tensor([new] * input_ids.shape[0], dtype=input_ids.dtype)
        return torch.cat([input_ids, tail], dim=1)


class StubSource:
    """A built ``(task, split, arm)``, satisfying `adapters.TaskSource`."""

    def __init__(self, task, split, arm, items):
        self.task, self.split, self.arm, self.pass_id = task, split, arm, 0
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return dict(self._items[i])

    def lengths(self):
        return ([int(x["num_nodes"]) for x in self._items],
                [sum(len(ids) for ids in x["input_ids"]) for x in self._items])


TOKENIZER = FakeTokenizer()


def flat_item(task, split, kind, question, smiles, answer, key, meta=None):
    """One flat-arm item, built the way the molecules adapter builds one.

    The prompt text is `dataset.build_flat_example`'s verbatim, so
    `perm_spread`'s SMILES rewriting and `schema.render`'s span both meet the
    string they were written against.
    """
    lead = "" if kind in ("token", "yesno") else " "
    text = f"{question}\nSMILES: {smiles}\nA:{lead}{answer}"
    stub = Example(task=task, domain="molecules", split=split, arm="flat",
                   graph={"text": [text], "prompt_node": 0, "num_nodes": 1},
                   question=question, answer=answer, answer_kind=kind, key=key)
    rendered = render(stub, TOKENIZER)
    # `labels` and `shortest_path_dists` are tensors, as
    # `TextGraphDataset.__getitem__` hands them over — the collator asserts on
    # the label tensor's shape, so a list here would test a shape nothing produces.
    graph = {
        "text": [text], "num_nodes": 1, "prompt_node": 0, "question_node": -1,
        "edges": [], "input_ids": [list(rendered.input_ids[0])],
        "labels": torch.tensor(rendered.labels, dtype=torch.long),
        "shortest_path_dists": torch.zeros((1, 1), dtype=torch.int32),
    }
    return Example(task=task, domain="molecules", split=split, arm="flat",
                   graph=graph, question=question, answer=answer,
                   answer_kind=kind, key=key, meta=dict(meta or {})).to_item()


YESNO_Q = "Question: does this molecule inhibit the target?"
TOXIC_Q = "Question: is this molecule active on endpoint NR-AR?"


def _yesno_items(task, split):
    """Two molecules whose prompts agree up to the scored position.

    One positive, one negative, so AUROC is defined; identical scoring contexts,
    so the pair is tied and ``tied_pair_fraction`` must come back at 1.0.
    """
    return [
        flat_item(task, split, "yesno", YESNO_Q, "CCO", YES, "CCO",
                  {"endpoint": "NR-AR"}),
        flat_item(task, split, "yesno", YESNO_Q, "CCO", NO, "CCO",
                  {"endpoint": "NR-AR"}),
    ]


def _multi_endpoint_items(task, split):
    return [
        flat_item(task, split, "yesno", TOXIC_Q, "CCO", YES, "CCO",
                  {"endpoint": "NR-AR"}),
        flat_item(task, split, "yesno", TOXIC_Q, "CCO", NO, "CCO",
                  {"endpoint": "NR-AR"}),
        flat_item(task, split, "yesno", TOXIC_Q, "c1ccccc1", YES, "c1ccccc1",
                  {"endpoint": "SR-MMP"}),
        flat_item(task, split, "yesno", TOXIC_Q, "c1ccccc1", NO, "c1ccccc1",
                  {"endpoint": "SR-MMP"}),
    ]


TASKS = {
    "mol/ring_count": ("token", "exact_match"),
    "mol/bace": ("yesno", "roc_auc"),
    "mol/tox21": ("yesno", "roc_auc"),
    "mol/chebi20": ("text", "bleu2"),
    "mol/g2s": ("smiles", "roundtrip_match"),
    "mol/bond_path": ("token", "exact_match"),          # held out (§4)
}


@pytest.fixture(scope="module")
def registry():
    reg = Registry()
    for name, (kind, metric) in TASKS.items():
        reg.register(TaskSpec(
            name=name, domain="molecules", adapter="molecules",
            kind="generator" if name in ("mol/ring_count", "mol/g2s") else "corpus",
            answer_kind=kind, metric=metric, held_out=(name == "mol/bond_path"),
            weight=1.0, mean_tokens=40.0, train_size=32,
            max_new_tokens=8, build_version="test"))
    return reg


@pytest.fixture(scope="module")
def eval_sets():
    q_a = "Question: how many rings does this molecule have?"
    sets = {
        "mol/ring_count": {
            "test": StubSource("mol/ring_count", "test", "flat", [
                flat_item("mol/ring_count", "test", "token", q_a, "CCO", " 0", "CCO"),
                flat_item("mol/ring_count", "test", "token", q_a, "c1ccccc1", " 1",
                          "c1ccccc1"),
            ])},
        "mol/bace": {
            "val": StubSource("mol/bace", "val", "flat", _yesno_items("mol/bace", "val")),
            "test": StubSource("mol/bace", "test", "flat",
                               _yesno_items("mol/bace", "test")),
        },
        "mol/tox21": {
            "test": StubSource("mol/tox21", "test", "flat",
                               _multi_endpoint_items("mol/tox21", "test"))},
        "mol/chebi20": {
            "test": StubSource("mol/chebi20", "test", "flat", [
                flat_item("mol/chebi20", "test", "text",
                          "Question: describe this molecule.", "CCO",
                          "The molecule is an alcohol.", "CCO")])},
        "mol/g2s": {
            "test": StubSource("mol/g2s", "test", "flat", [
                flat_item("mol/g2s", "test", "smiles",
                          "Question: write the canonical SMILES for this molecule.",
                          "OCC", "CCO", "CCO")])},
        "mol/bond_path": {
            "held_out": StubSource("mol/bond_path", "held_out", "flat", [
                flat_item("mol/bond_path", "held_out", "token",
                          "Question: how many bonds separate atom 1 and atom 3?",
                          "CCO", " 2", "CCO")])},
    }
    return sets


@pytest.fixture
def ctx(registry, eval_sets, tmp_path):
    """A context every built-in can run against."""
    mixture = resolve(
        registry,
        [{"name": "mol/bace", "weight": 0.6}, {"name": "mol/chebi20", "weight": 0.4}],
        tokens_per_step=512, steps=10, min_examples_per=0)
    model = StubLM()
    shares = mixture.shares

    def loss_fn(task):
        # A gradient whose norm is proportional to the task's configured share,
        # so `grad_share` must report the shares back and `max_abs_error` ~ 0.
        return (model.graph_bias_weights * float(shares[task])).sum()

    return ev.EvalContext(
        step=100, model=model, tokenizer=TOKENIZER, registry=registry,
        mixture=mixture, arm="flat", schedule_position=(1, 40),
        eval_sets=eval_sets, train_sampler=object(),
        base_model_name="fake/base", collator=GraphCollatorV2(pad_token_id=0),
        device=None, scratch_dir=str(tmp_path),
        config={builtin.ACTIVE_PARAMS: ["graph_bias"],
                builtin.GRAD_SHARE_LOSS_FN: loss_fn,
                builtin.MAX_SPD: 32})


#: Per-validator construction options for the walk over the registry. A built-in
#: that needs something no stub context can carry declares it here; anything not
#: listed is built with its defaults, so a validator added later is covered by
#: the walk without this file changing.
OPTIONS = {
    "perm_spread": {"n_permutations": 3},                # 10 is the run's default
}


def all_builtins(model=None):
    """Every registered validator, built at cadence ``manual``."""
    out = []
    for name in ev.names():
        options = dict(OPTIONS.get(name, {}))
        if name == "base_exact":
            # Loading real base weights belongs to the GPU smoke; what is under
            # test here is that the comparison runs and reports its applicability.
            options["base_model"] = model
        out.append(ev.get(name)(cadence="manual", **options))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cadence
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_cadence_accepts_the_four_forms_and_nothing_else():
    assert ev.parse_cadence("steps:500") == ("steps", 500)
    assert ev.parse_cadence("milestone") == ("milestone", None)
    assert ev.parse_cadence("end") == ("end", None)
    assert ev.parse_cadence("manual") == ("manual", None)
    for bad in ("steps:", "steps:0", "steps:x", "every", "", None, 500):
        with pytest.raises(ev.EvalError):
            ev.parse_cadence(bad)


def test_should_run_fires_on_each_cadence_form_and_not_otherwise():
    # steps:<n> — on its multiples, on step events only, never at step 0.
    assert ev.should_run("steps:50", 100, "step")
    assert not ev.should_run("steps:50", 99, "step")
    assert not ev.should_run("steps:50", 0, "step")
    assert not ev.should_run("steps:50", 100, "milestone")
    assert not ev.should_run("steps:50", 100, "end")

    assert ev.should_run("milestone", 7, "milestone")
    assert not ev.should_run("milestone", 7, "step")
    assert not ev.should_run("milestone", 7, "end")

    assert ev.should_run("end", 7, "end")
    assert not ev.should_run("end", 7, "step")
    assert not ev.should_run("end", 7, "milestone")

    # `manual` cadence fires on nothing but a manual event; a manual event fires
    # everything, because naming a validator is already the decision.
    for event in ("step", "milestone", "end"):
        assert not ev.should_run("manual", 7, event)
    for cadence in ("steps:50", "milestone", "end", "manual"):
        assert ev.should_run(cadence, 7, "manual")

    with pytest.raises(ev.EvalError):
        ev.should_run("end", 7, "epoch")


def test_a_bad_cadence_fails_when_the_validator_list_is_built():
    """`validate` mode, on the login node — not at step 500 on a GPU."""
    with pytest.raises(ev.EvalError):
        ev.build_validators([{"name": "bias_norm", "cadence": "steps:nope"}])
    with pytest.raises(ev.EvalError):
        ev.build_validators([{"name": "no_such_validator"}])
    with pytest.raises(ev.EvalError):
        ev.build_validators([{"name": "bias_norm"}, {"name": "bias_norm"}])
    built = ev.build_validators(["bias_norm", {"name": "throughput", "cadence": "end"}])
    assert [v.name for v in built] == ["bias_norm", "throughput"]
    assert built[1].cadence == "end"


def test_an_option_a_validator_will_not_honour_is_refused_by_name():
    """The smoke set capped `per_example` at 32 and got all 1000 rows, silently.

    A cost knob that some validators read and others ignore is a config that
    lies about what it costs and what it measured. It fails at build time, which
    is `validate` mode on the login node.
    """
    with pytest.raises(ev.EvalError) as exc:
        ev.build_validators([{"name": "per_example", "max_samples": 32}])
    assert "max_samples" in str(exc.value)
    # The ones that do honour it are unaffected.
    assert ev.build_validators([{"name": "in_mixture", "max_samples": 32}])[0] \
        .option("max_samples") == 32


# ─────────────────────────────────────────────────────────────────────────────
# The runner's three contracts
# ─────────────────────────────────────────────────────────────────────────────

class Fine(ev.BaseValidator):
    name = "fine"
    cadence = "manual"

    def keys(self, ctx=None):
        return {"value"}

    def run(self, ctx):
        return {"value": 1.0, "mol/bace/value": 2.0}


class Boom(ev.BaseValidator):
    name = "boom"
    cadence = "manual"

    def keys(self, ctx=None):
        return {"value"}

    def run(self, ctx):
        raise RuntimeError("this validator is nonsense")


class Undeclared(ev.BaseValidator):
    name = "undeclared"
    cadence = "manual"

    def keys(self, ctx=None):
        return {"declared"}

    def run(self, ctx):
        return {"declared": 1.0, "sneaked_in": 2.0}


class WantsEverything(ev.BaseValidator):
    name = "wants_everything"
    cadence = "manual"
    needs = frozenset({"model", "train_sampler", "base_model"})

    def keys(self, ctx=None):
        return {"value"}

    def run(self, ctx):
        return {"value": 1.0}


def test_a_raising_validator_is_skipped_and_the_others_survive(ctx, capsys):
    run = ev.run_validators(ctx, [Fine(), Boom(), Fine(cadence="manual")],
                            event="manual")
    assert run.metrics == {"fine/value": 1.0, "fine/mol/bace/value": 2.0}
    assert [s.state for s in run.statuses] == ["ran", "error", "ran"]
    assert "this validator is nonsense" in run.status("boom").message
    # Logged, not swallowed: the traceback is on stdout for the job log.
    assert "boom failed at step 100" in capsys.readouterr().out
    with pytest.raises(ev.EvalError):
        run.raise_for_errors()


def test_a_raising_validator_is_fatal_only_under_strict(ctx):
    """The smoke run (T10) asks for the failure; a training run never does."""
    with pytest.raises(RuntimeError):
        ev.run_validators(ctx, [Boom()], event="manual", strict=True)


def test_an_undeclared_key_fails_the_validator_and_drops_its_metrics(ctx):
    run = ev.run_validators(ctx, [Undeclared(), Fine()], event="manual")
    assert run.status("undeclared").state == "error"
    assert "sneaked_in" in run.status("undeclared").message
    assert not any(k.startswith("undeclared/") for k in run.metrics)
    assert run.metrics["fine/value"] == 1.0


def test_unmet_needs_name_the_missing_field(ctx):
    bare = ev.EvalContext(step=3)
    with pytest.raises(ev.EvalNeedsError) as excinfo:
        ev.check_needs(WantsEverything(), bare)
    assert "wants_everything" in str(excinfo.value)
    assert "base_model_name" in str(excinfo.value)      # the alias resolves

    run = ev.run_validators(bare, [WantsEverything(), Fine()], event="manual")
    assert run.status("wants_everything").state == "error"
    assert "base_model_name" in run.status("wants_everything").message
    assert run.metrics["fine/value"] == 1.0             # the run continues


def test_a_validator_cannot_ask_for_a_field_the_context_has_no_room_for():
    class Typo(ev.BaseValidator):
        name = "typo"
        needs = frozenset({"modle"})

    with pytest.raises(ev.EvalError):
        Typo()


def test_metric_values_must_survive_the_run_record(ctx):
    class Dicty(ev.BaseValidator):
        name = "dicty"
        cadence = "manual"

        def keys(self, ctx=None):
            return {"value"}

        def run(self, ctx):
            return {"value": {"nested": 1}}

    run = ev.run_validators(ctx, [Dicty()], event="manual")
    assert run.status("dicty").state == "error"
    assert "dict" in run.status("dicty").message


def test_the_protocol_version_is_in_the_record(ctx):
    validators = [Fine(), builtin.BiasNorm(cadence="manual")]
    run = ev.run_validators(ctx, validators, event="manual")
    record = run.record()
    assert ["bias_norm", "1"] in record["protocol_versions"]
    assert ["fine", "1"] in record["protocol_versions"]
    versions = {s["name"]: s["protocol_version"] for s in record["statuses"]}
    assert versions["bias_norm"] == "1"
    # D7.2: the set covers what the run was *configured* with, not only what fired.
    quiet = ev.run_validators(ctx, [Fine(cadence="end")], event="step")
    assert quiet.status("fine").state == "not_due"
    assert quiet.record()["protocol_versions"] == [["fine", "1"]]


def test_only_restricts_the_set_without_hiding_the_rest(ctx):
    run = ev.run_validators(ctx, [Fine(), Boom()], event="manual", only=["fine"])
    assert run.status("boom").state == "not_due"
    assert run.metrics["fine/value"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# D7.4 — selection
# ─────────────────────────────────────────────────────────────────────────────

def test_a_selection_key_naming_test_is_refused():
    for key in ("in_mixture/mol/bace/test/roc_auc", "test_roc_auc",
                "eval_test_f1", "TEST/roc_auc"):
        with pytest.raises(ev.EvalError) as excinfo:
            ev.check_selection({"metric": key}, mode="anneal")
        assert "test" in str(excinfo.value)
    with pytest.raises(ev.EvalError):
        ev.check_selection({"metric": "in_mixture/mol/bace/val/roc_auc",
                            "split": "test"}, mode="adapt")


def test_a_training_run_never_selects_and_a_fork_may():
    with pytest.raises(ev.EvalError):
        ev.check_selection({"metric": "in_mixture/mol/bace/val/roc_auc"})
    assert ev.check_selection(None) is None
    ok = ev.check_selection({"metric": "held_out/mol/bond_path/held_out/em_accuracy",
                             "split": "val"}, mode="adapt")
    assert ok["split"] == "val"
    # A name that merely contains the four letters is not a test key.
    assert ev.check_selection({"metric": "latest_loss"}, mode="anneal")


# ─────────────────────────────────────────────────────────────────────────────
# Every built-in, over the registry
# ─────────────────────────────────────────────────────────────────────────────

def test_every_builtin_declares_its_keys_and_returns_exactly_them(ctx):
    """The walk is over the registry so a validator added later is covered here.

    ``keys(ctx)`` is the context-aware declaration: for a scoring validator it
    depends on which answer kinds are actually in the eval sets, which is the
    only honest way to say "exactly these" about a plugin whose output is a
    function of the mixture.
    """
    validators = all_builtins(model=ctx.model)
    assert {v.name for v in validators} == set(ev.names())

    run = ev.run_validators(ctx, validators, event="manual", strict=True)
    assert not run.errors(), [(s.name, s.message) for s in run.errors()]

    for validator in validators:
        declared = set(validator.keys(ctx))
        assert declared, f"{validator.name} declares no keys"
        returned = {k.split("/", 1)[1].rsplit("/", 1)[-1]
                    for k in run.metrics if k.startswith(f"{validator.name}/")}
        assert returned == declared, (
            f"{validator.name}: returned {sorted(returned)}, declared "
            f"{sorted(declared)}")


def test_the_runner_namespaces_by_validator_task_and_metric(ctx):
    run = ev.run_validators(ctx, [builtin.InMixture(cadence="manual")],
                            event="manual", strict=True)
    assert "in_mixture/mol/bace/test/roc_auc" in run.metrics
    assert "in_mixture/mol/bace/val/roc_auc" in run.metrics
    assert "in_mixture/mol/ring_count/test/em_accuracy" in run.metrics
    # Held out never appears in the in-mixture readout, and vice versa.
    assert not any("bond_path" in k for k in run.metrics)

    held = ev.run_validators(ctx, [builtin.HeldOut(cadence="manual")],
                             event="manual", strict=True)
    assert "held_out/mol/bond_path/held_out/em_accuracy" in held.metrics
    assert not any("bace" in k for k in held.metrics)


def test_a_yesno_task_always_reports_both_tie_diagnostics(ctx):
    """`n_distinct` alone is a bf16 artifact; `tied_pair_fraction` is the bound.

    The two molecules here share a scoring context, so the single
    (positive, negative) pair is tied: AUROC is 0.5 by a coin flip and
    ``tied_pair_fraction`` is 1.0, which is the whole of what the AUROC rests on.
    Reporting one number without the other is what makes that unreadable.
    """
    run = ev.run_validators(ctx, [builtin.InMixture(cadence="manual")],
                            event="manual", strict=True)
    prefix = "in_mixture/mol/bace/test/"
    assert run.metrics[prefix + "tied_pair_fraction"] == 1.0
    assert run.metrics[prefix + "n_distinct"] == 1.0
    assert run.metrics[prefix + "roc_auc"] == 0.5
    assert run.metrics[prefix + "pos_rate"] == 0.5
    for name in ("average_precision", "accuracy", "f1", "margin_mean", "n"):
        assert prefix + name in run.metrics


def test_a_multi_endpoint_corpus_is_broken_out_and_a_single_one_is_not(ctx):
    run = ev.run_validators(ctx, [builtin.InMixture(cadence="manual")],
                            event="manual", strict=True)
    assert "in_mixture/mol/tox21/test/endpoint:NR-AR/roc_auc" in run.metrics
    assert "in_mixture/mol/tox21/test/endpoint:SR-MMP/roc_auc" in run.metrics
    assert run.metrics["in_mixture/mol/tox21/test/endpoint:NR-AR/n"] == 2
    # BACE has one endpoint; a breakdown there would restate the pooled number
    # under a second name and read as an independent measurement.
    assert not any(k.startswith("in_mixture/mol/bace/test/endpoint:")
                   for k in run.metrics)


def test_bias_norm_reads_the_channel_and_says_when_there_is_none(ctx):
    run = ev.run_validators(ctx, [builtin.BiasNorm(cadence="manual")],
                            event="manual", strict=True)
    assert run.metrics["bias_norm/present"] == 1.0
    # 32 entries of 0.25 -> sqrt(32 * 0.0625) = sqrt(2).
    assert run.metrics["bias_norm/l2"] == pytest.approx(math.sqrt(2.0))

    flat = ev.run_validators(
        ctx, [builtin.BiasNorm(cadence="manual", active_params=["no_such_thing"])],
        event="manual", strict=True)
    assert flat.metrics["bias_norm/present"] == 0.0
    assert math.isnan(flat.metrics["bias_norm/l2"])


def test_grad_share_reports_the_configured_weights_back(ctx):
    run = ev.run_validators(ctx, [builtin.GradShare(cadence="manual")],
                            event="manual", strict=True)
    assert run.metrics["grad_share/mol/bace/share"] == pytest.approx(0.6, abs=1e-6)
    assert run.metrics["grad_share/mol/bace/weight"] == pytest.approx(0.6, abs=1e-6)
    assert run.metrics["grad_share/max_abs_error"] < 1e-6
    assert run.metrics["grad_share/n_tasks"] == 2.0
    assert run.metrics["grad_share/n_measured"] == 2.0


def test_grad_share_reports_an_absent_task_as_absent_not_as_zero(ctx):
    """The first smoke run reported a task the sample never drew as 0.0.

    A zero share reads as "this task contributes nothing to the gradient", which
    is a finding; "the step did not sample it" is not. The two must not look the
    same, and an absent task must not drag ``max_abs_error`` with it.
    """
    from dataclasses import replace

    absent = "mol/chebi20"
    loss_fn = ctx.config[builtin.GRAD_SHARE_LOSS_FN]
    counts = {"mol/bace": 5}
    config = dict(ctx.config)
    config[builtin.GRAD_SHARE_LOSS_FN] = (
        lambda task: None if task == absent else loss_fn(task))
    config[builtin.GRAD_SHARE_COUNTS_FN] = lambda: counts

    run = ev.run_validators(replace(ctx, config=config),
                            [builtin.GradShare(cadence="manual")],
                            event="manual", strict=True)
    assert math.isnan(run.metrics[f"grad_share/{absent}/share"])
    assert math.isnan(run.metrics[f"grad_share/{absent}/abs_error"])
    assert run.metrics[f"grad_share/{absent}/examples"] == 0.0
    assert run.metrics["grad_share/n_measured"] == 1.0
    # The one task that was drawn took the whole normalised share, and the
    # absent one contributed nothing to the worst error.
    assert run.metrics["grad_share/mol/bace/share"] == pytest.approx(1.0, abs=1e-6)
    assert run.metrics["grad_share/mol/bace/step_share"] == pytest.approx(1.0)
    assert run.metrics["grad_share/max_abs_error"] == pytest.approx(0.4, abs=1e-6)


def test_grad_share_says_what_the_trainer_must_install(ctx):
    """The closure cannot be reconstructed here — drawing from the sampler would
    advance the cursor of the run being measured."""
    from dataclasses import replace

    stripped = replace(ctx, config={})
    run = ev.run_validators(ctx=stripped,
                            validators=[builtin.GradShare(cadence="manual")],
                            event="manual")
    assert run.status("grad_share").state == "error"
    assert builtin.GRAD_SHARE_LOSS_FN in run.status("grad_share").message


def test_base_exact_reports_itself_inapplicable_rather_than_failing(ctx):
    from dataclasses import replace

    same = builtin.BaseExact(cadence="manual", base_model=ctx.model)
    run = ev.run_validators(ctx, [same], event="manual", strict=True)
    assert run.metrics["base_exact/applicable"] == 1.0
    assert run.metrics["base_exact/max_abs_diff"] == 0.0
    assert run.metrics["base_exact/within_tolerance"] == 1.0

    # A run whose forward pass moves unconditionally: the property does not hold
    # and its failure is not a defect, so `applicable` says so.
    moved = replace(ctx, config=dict(ctx.config, unconditional_forward_change="arm C"))
    out = ev.run_validators(moved, [same], event="manual", strict=True)
    assert out.metrics["base_exact/applicable"] == 0.0
    assert "arm C" in out.metrics["base_exact/reason"]

    # And automatically, when a trainable parameter sits outside LoRA and the bias.
    bare = replace(ctx, config={builtin.ACTIVE_PARAMS: ["nothing_matches"]})
    auto = ev.run_validators(bare, [same], event="manual", strict=True)
    assert auto.metrics["base_exact/applicable"] == 0.0
    assert "graph_bias" in auto.metrics["base_exact/reason"]


def test_base_exact_sees_a_backbone_weight_that_moved(ctx):
    """The property is about the weights, so a moved weight must fail it.

    The comparison is exact, not within a bf16 tolerance: a frozen tensor is
    bit-identical to the one it was loaded from, and anything else is a backbone
    that trained.
    """
    other = StubLM()
    with torch.no_grad():
        other.head.weight[0, 0] += 0.5
    moved = builtin.BaseExact(cadence="manual", base_model=other)
    run = ev.run_validators(ctx, [moved], event="manual", strict=True)
    assert run.metrics["base_exact/max_abs_diff"] == pytest.approx(0.5)
    assert run.metrics["base_exact/within_tolerance"] == 0.0
    # The graph-bias parameter is what this project adds; it has no counterpart
    # in a base model and must not be compared or counted as missing.
    assert run.metrics["base_exact/n_unmatched"] == 0.0
    assert run.metrics["base_exact/n_tensors"] == 3.0     # emb + head weight/bias


def test_base_exact_reports_a_backbone_tensor_the_base_model_does_not_have(ctx):
    """A renamed or added backbone module must not pass by comparing nothing."""
    from dataclasses import replace

    class Extra(StubLM):
        def __init__(self):
            super().__init__()
            self.side_channel = torch.nn.Parameter(torch.zeros(3))

    theirs = builtin.BaseExact(cadence="manual", base_model=StubLM())
    run = ev.run_validators(replace(ctx, model=Extra()), [theirs],
                            event="manual", strict=True)
    assert run.metrics["base_exact/n_unmatched"] == 1.0
    assert run.metrics["base_exact/within_tolerance"] == 0.0


def test_base_exact_refuses_when_nothing_matched(ctx):
    """Comparing zero tensors is a wrong name mapping, not a passing check."""
    from dataclasses import replace

    nothing = builtin.BaseExact(cadence="manual", base_model=torch.nn.Linear(1, 1),
                                added_fragments=("emb", "head", "graph_bias"))
    run = ev.run_validators(ctx, [nothing], event="manual")
    assert run.status("base_exact").state == "error"
    assert "nothing was checked" in run.status("base_exact").message


def test_base_exact_strips_pefts_wrapper_and_base_layer_paths():
    """`base_model.model.…` and `.base_layer.` are paths to the same storage."""
    assert builtin._backbone_name(
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
    ) == "model.layers.0.self_attn.q_proj.weight"
    assert builtin._backbone_name("model.embed_tokens.weight") == \
        "model.embed_tokens.weight"


def test_throughput_is_wall_clock_between_firings(ctx):
    """Not a mean of per-step millisecond timers (`feedback-throughput-metric`)."""
    from dataclasses import replace

    validator = builtin.Throughput(cadence="manual")
    first = ev.run_validators(ctx, [validator], event="manual", strict=True)
    assert math.isnan(first.metrics["throughput/s_per_it"])

    later = ev.run_validators(replace(ctx, step=ctx.step + 20), [validator],
                              event="manual", strict=True)
    assert later.metrics["throughput/steps_measured"] == 20.0
    assert later.metrics["throughput/wall_s"] > 0.0
    assert later.metrics["throughput/s_per_it"] == pytest.approx(
        later.metrics["throughput/wall_s"] / 20.0)
    assert later.metrics["throughput/tokens_per_s"] == pytest.approx(
        ctx.mixture.tokens_per_step / later.metrics["throughput/s_per_it"])


def test_per_example_writes_one_row_per_example(ctx):
    import json
    import os

    run = ev.run_validators(ctx, [builtin.PerExample(cadence="manual")],
                            event="manual", strict=True)
    path = run.metrics["per_example/mol/bace/per_example_path"]
    assert os.path.exists(path)
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 2
    assert {"i", "correct", "margin", "y_true", "y_score"} <= set(rows[0])
    # The report checks itself: the AUROC recomputed from the rows equals the
    # one the margin readout reported from the same predictions.
    assert run.metrics["per_example/mol/bace/per_example_roc_auc"] == 0.5
    assert "per_example/mol/ring_count/per_example_accuracy" in run.metrics


# ─────────────────────────────────────────────────────────────────────────────
# perm_spread
# ─────────────────────────────────────────────────────────────────────────────

def test_perm_spread_stratifies_by_symmetry_class(ctx):
    """Benzene has one atom symmetry class, so randomisation cannot move it and a
    pooled spread would understate the effect (`molecules/PLAN.md` §6)."""
    run = ev.run_validators(ctx, [builtin.PermSpread(cadence="manual",
                                                     n_permutations=3)],
                            event="manual", strict=True)
    assert run.metrics["perm_spread/mol/tox21/n_permutations"] == 3.0
    assert run.metrics["perm_spread/mol/tox21/all/n_molecules"] == 4.0
    # CCO has three classes, benzene one.
    assert run.metrics["perm_spread/mol/tox21/asymmetric/n_molecules"] == 2.0
    assert run.metrics["perm_spread/mol/tox21/symmetric/n_molecules"] == 2.0


def test_the_flat_arm_reports_a_spread_but_no_verdict(ctx):
    """`within_tolerance` asserts Property 1, which is a graph-arm property.

    The flat arm's re-writing is a different SMILES string for the same
    molecule, and a nonzero spread there is the measurement rather than a
    failure — the first flat cross-check read a `margin_spread_max` of 31.6
    beside a `within_tolerance` of 0.0, which is the arm doing exactly what it
    is described as doing and reads at a glance as a broken run. The spread and
    the measured quantum are reported on both arms; the verdict is not.
    """
    run = ev.run_validators(ctx, [builtin.PermSpread(cadence="manual",
                                                     n_permutations=3)],
                            event="manual", strict=True)
    assert "perm_spread/mol/tox21/margin_spread_max" in run.metrics
    assert "perm_spread/mol/tox21/margin_quantum" in run.metrics
    assert "perm_spread/mol/tox21/within_tolerance" not in run.metrics
    assert "perm_spread/mol/tox21/margin_control_max" not in run.metrics


def test_the_graph_arm_measures_the_floor_it_asserts_against(ctx):
    """Property 1 is about the function; the margin is that function in bf16.

    Re-batching alone moves the margin — the BACE cross-check measured 0.375 at
    the end of the stable phase and 0.750 after the anneal, on inputs that were
    bit-identical and with no relabelling anywhere. A fixed absolute tolerance
    cannot be right at both ends of a run, so the run measures its own floor and
    the assertion is "relabelling moves the margin no more than re-batching
    does". One permutation keeps the relabelling out of a flat fixture while
    still walking the whole graph-arm branch.
    """
    import dataclasses

    graph = dataclasses.replace(ctx, arm="graph")
    run = ev.run_validators(graph, [builtin.PermSpread(cadence="manual",
                                                       n_permutations=1,
                                                       n_control=3)],
                            event="manual", strict=True)
    assert run.metrics["perm_spread/mol/tox21/margin_control_max"] == 0.0
    assert run.metrics["perm_spread/mol/tox21/within_tolerance"] == 1.0


def test_the_control_can_be_turned_off(ctx):
    """`n_control: 0` falls back to the quantum alone — the behaviour before the
    floor was measured, kept reachable because the extra passes are not free."""
    import dataclasses

    graph = dataclasses.replace(ctx, arm="graph")
    run = ev.run_validators(graph, [builtin.PermSpread(cadence="manual",
                                                       n_permutations=1,
                                                       n_control=0)],
                            event="manual", strict=True)
    assert run.metrics["perm_spread/mol/tox21/margin_control_max"] == 0.0
    assert "perm_spread/mol/tox21/within_tolerance" in run.metrics


def test_the_tolerance_cannot_be_finer_than_the_margins_own_grid():
    """A 1e-4 assertion on a bf16 margin quantised to 0.125 can only pass by luck.

    The graph arm's spread is asserted against ``max(tolerance, margin_quantum)``
    and the quantum is measured, not assumed: on the smoke run's own rows the
    margin took 21 distinct values over 152 examples with a minimum gap of
    exactly 0.125, and a spread of one such step is not something this
    instrument can tell apart from rounding.
    """
    import numpy as np

    on_a_grid = np.array([[0.0, 0.25], [0.125, 0.25], [0.0, 0.375]])
    assert builtin._margin_quantum(on_a_grid) == 0.125
    # Every margin identical: there is no grid to measure and no spread either.
    assert builtin._margin_quantum(np.zeros((3, 2))) == 0.0
    assert builtin._margin_quantum(np.array([])) == 0.0


def test_a_graph_arm_relabeling_permutes_every_node_indexed_column():
    item = {
        "text": ["a", "b", "c"], "num_nodes": 3, "prompt_node": 2,
        "question_node": 1, "edges": [(0, 1), (1, 2)],
        "input_ids": [[1], [2], [3]],
        # Aligned to the prompt node's tokens; its length equals the node count
        # here on purpose, since that coincidence is what a shape-based rule
        # would get wrong.
        "labels": torch.tensor([-100, -100, 7]),
        "shortest_path_dists": torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
        "magnetic_V": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        # (M,), indexed by eigenvector — and M is the node count whenever the
        # spectrum is not truncated, which is what made a shape-based rule
        # permute it.
        "magnetic_lambdas": torch.tensor([0.0, 1.5, 2.5]),
        "original_ids": {"a": 0, "b": 1, "c": 2},
    }
    out = builtin._relabelled_graph_item(item, perm_id=1)
    order = [item["text"].index(t) for t in out["text"]]      # new -> old
    where = {old: new for new, old in enumerate(order)}
    assert order != [0, 1, 2], "the relabeling did not move anything"

    assert out["num_nodes"] == 3
    assert torch.equal(out["labels"], item["labels"])          # not node-indexed
    assert out["prompt_node"] == where[2]
    assert out["question_node"] == where[1]
    assert sorted(out["edges"]) == sorted(
        [(where[0], where[1]), (where[1], where[2])])
    assert out["input_ids"] == [item["input_ids"][old] for old in order]
    assert out["original_ids"] == {"a": where[0], "b": where[1], "c": where[2]}
    assert torch.equal(out["magnetic_V"],
                       torch.stack([item["magnetic_V"][old] for old in order]))
    # The spectrum belongs to the graph. Permuting it pairs every eigenvector
    # with another one's eigenvalue, and the graph arm then reports a nonzero
    # margin spread that reads as a violation of Property 1 — the smoke run's
    # 0.75 came from exactly this.
    assert torch.equal(out["magnetic_lambdas"], item["magnetic_lambdas"])

    spd, original = out["shortest_path_dists"], item["shortest_path_dists"]
    for i in range(3):
        for j in range(3):
            assert spd[where[i], where[j]] == original[i, j]


def test_a_truncated_spectrum_is_not_mistaken_for_a_node_column():
    """``magnetic_m`` truncates to M < N, and the shape rule then *rejects* it.

    Both halves of the bug are the same mistake: the eigenvalue vector is not
    indexed by node. At M == N it was permuted and produced a fake Property 1
    violation; at M != N it would have failed the validator outright.
    """
    item = {"text": ["a", "b", "c"], "num_nodes": 3, "prompt_node": 0,
            "edges": [], "input_ids": [[1], [2], [3]],
            "magnetic_lambdas": torch.tensor([0.0, 1.5])}      # M = 2, N = 3
    out = builtin._relabelled_graph_item(item, perm_id=1)
    assert torch.equal(out["magnetic_lambdas"], item["magnetic_lambdas"])


def test_a_relabeling_refuses_a_column_it_cannot_place():
    """A silently mis-permuted feature reads as a violation of Property 1, which
    is the one failure here that would look like a real result."""
    item = {"text": ["a", "b"], "num_nodes": 2, "prompt_node": 0,
            "edges": [], "input_ids": [[1], [2]],
            "mystery": torch.zeros((3, 4))}
    with pytest.raises(ev.EvalError) as excinfo:
        builtin._relabelled_graph_item(item, perm_id=1)
    assert "mystery" in str(excinfo.value)


def test_a_rewritten_flat_prompt_is_the_same_molecule(ctx):
    from rdkit import Chem

    item = flat_item("mol/bace", "test", "yesno", YESNO_Q, "OCCCC", YES, "CCCCO")
    out = builtin._rewritten_flat_item(item, TOKENIZER, perm_id=2)
    start = out["text"][0].find(builtin.SMILES_MARKER) + len(builtin.SMILES_MARKER)
    end = out["text"][0].find("\n", start)
    rewritten = out["text"][0][start:end]
    assert Chem.MolToSmiles(Chem.MolFromSmiles(rewritten)) == "CCCCO"
    # The label convention comes from `schema.render`, not from a copy of it.
    assert int(out["labels"][-1]) == out["input_ids"][0][-1] == YES_ID
    assert set(out["labels"][:-1].tolist()) == {-100}


# ─────────────────────────────────────────────────────────────────────────────
# The scorers, on cases computed by hand
# ─────────────────────────────────────────────────────────────────────────────

def test_bleu_rouge_and_meteor_on_a_hand_computed_caption():
    from src.generalist.evaluate.captions import bleu, caption_metrics, meteor, rouge_l

    hyp, ref = ["the cat"], ["the cat sat"]
    # p1 = 2/2, p2 = 1/1, BP = exp(1 - 3/2).
    assert bleu(hyp, ref, max_n=2) == pytest.approx(math.exp(-0.5))
    # No 3-grams in a 2-token hypothesis, so BLEU-4 is 0 without smoothing.
    assert bleu(hyp, ref, max_n=4) == 0.0
    # LCS 2, precision 1, recall 2/3 -> F1 0.8.
    assert rouge_l(hyp, ref) == pytest.approx(0.8)
    # m = 2, P = 1, R = 2/3, F = PR/(0.9P + 0.1R), one chunk -> penalty 0.5 * 0.5^3.
    f_mean = (1.0 * (2 / 3)) / (0.9 * 1.0 + 0.1 * (2 / 3))
    assert meteor(hyp, ref) == pytest.approx(f_mean * (1 - 0.5 * 0.5 ** 3))

    identical = caption_metrics(["the cat sat"], ["the cat sat"])
    assert identical["bleu2"] == pytest.approx(1.0)
    assert identical["rouge_l"] == pytest.approx(1.0)
    assert identical["n"] == 1
    empty = caption_metrics([], [])
    assert set(empty) == {"bleu2", "bleu4", "rouge_l", "meteor", "n"}


def test_a_stereo_mark_in_a_smiles_prediction_is_an_error():
    """Under the §5 target stereo is an error, not a harmless extra: the graph
    carries parity words without the neighbour ordering that gives them meaning."""
    from src.generalist.adapters.molecules import smiles_scores

    out = smiles_scores(["C[C@H](N)O", "CC(N)O", "not a molecule"],
                        ["CC(N)O", "CC(N)O", "CC(N)O"])
    assert out["stereo_marks_emitted"] == pytest.approx(1 / 3)
    assert out["roundtrip_match"] == pytest.approx(1 / 3)   # the marked one fails
    assert out["exact_match"] == pytest.approx(1 / 3)
    assert out["validity"] == pytest.approx(2 / 3)


def test_generation_starts_exactly_at_the_answer_boundary(ctx):
    """The evaluation prompt is byte-identical to the training prompt up to the
    answer — anything else scores the model against a prompt it never saw."""
    from src.generalist.evaluate.scorers import answer_start, generate_predictions

    source = ctx.eval_sets["mol/g2s"]["test"]
    item = source[0]
    start = answer_start(item)
    prompt = TOKENIZER.decode(item["input_ids"][0][:start])
    # The answer's leading space belongs to the prefix under a tokenizer that
    # does not merge across the boundary, and to the answer under one that does
    # (`schema.render` widens the span in that case, deliberately). What has to
    # hold under both is that the prompt stops at "\nA:" and carries none of the
    # answer.
    assert prompt.rstrip(" ").endswith("\nA:")
    assert "CCO" not in prompt

    predictions, targets = generate_predictions(
        ctx.model, TOKENIZER, ctx.collator, source, [0], max_new_tokens=8)
    assert targets == ["CCO"]
    assert predictions == ["CCO"]        # the stub's canned continuation, stripped


@pytest.mark.slow
def test_the_answer_boundary_holds_under_the_real_tokenizer():
    """The one thing a character-level fake cannot show: a BPE merge at the boundary.

    Every scorer here locates the answer by reading the first supervised position
    off the ``labels`` column, and a real tokenizer merges the prompt's trailing
    ``":"`` with the answer's leading space into one token — which `render`
    handles by *widening* the span rather than narrowing it. Under a fake that
    never merges, that branch is never taken. This runs the real one, which is
    why it is marked slow: it wants the HF cache.
    """
    from transformers import AutoTokenizer

    from src.experiments.molecules.config import MODEL_NAME
    from src.generalist.evaluate.scorers import answer_start

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip(f"the real tokenizer is not available here: {exc}")

    question = "Question: write the canonical SMILES for this molecule."
    for kind, answer in (("yesno", YES), ("smiles", "CCO")):
        lead = "" if kind == "yesno" else " "
        text = f"{question}\nSMILES: OCC\nA:{lead}{answer}"
        example = Example(
            task="mol/g2s", domain="molecules", split="test", arm="flat",
            graph={"text": [text], "prompt_node": 0, "num_nodes": 1},
            question=question, answer=answer, answer_kind=kind, key="CCO")
        rendered = render(example, tokenizer)
        item = {"input_ids": [list(rendered.input_ids[0])],
                "labels": torch.tensor(rendered.labels), "prompt_node": 0}

        assert answer_start(item) == rendered.answer_start
        prompt = tokenizer.decode(item["input_ids"][0][:rendered.answer_start])
        assert answer.strip() not in prompt
        span = tokenizer.decode(item["input_ids"][0][rendered.answer_start:])
        assert span.strip() == answer.strip()


def test_the_token_scorer_is_exact_match_over_the_supervised_span(ctx):
    from src.generalist.evaluate.scorers import score_source

    spec = ctx.registry.get("mol/ring_count")
    out = score_source(ctx.model, TOKENIZER, ctx.collator,
                       ctx.eval_sets["mol/ring_count"]["test"], spec)
    assert out["n"] == 2
    assert 0.0 <= out["em_accuracy"] <= 1.0
    assert set(out) == {"em_accuracy", "n"}

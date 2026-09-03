"""
D3 — the adapter protocol: an existing dataset package becomes schema Examples.

An adapter owns *data*, never training logic. It answers three questions and
nothing else:

``partition(config) -> Partition``
    Which role — ``train`` / ``val`` / ``test`` / ``held_out`` — does every key
    this adapter will ever emit belong to? Computed from the raw sources, once,
    before a single example is generated, so the train/test boundary is a
    property of the *molecules* rather than of whichever task happened to be
    built first (`MOLECULE_GENERALIST.md` §3).

``build(config, roles) -> None``
    Materialise every split of every task it owns, on disk, under a cache key
    that includes the build version (D3.2). Validation runs here: an adapter that
    emits an item the schema rejects fails the build, not the run.

``load(task, split, arm, pass_id=0) -> TaskSource``
    Hand back what was built. It never regenerates — a resume that had to
    rebuild a pass would be a resume that changes the data under the sampler.

A :class:`TaskSource` is what the mixture (D4) batches over. The protocol is
deliberately small: ``__len__``, ``__getitem__`` returning a
``TextGraphDataset`` item with the schema sidecar attached
(``Example.to_item()``), :meth:`lengths` for the bucket table, and the four
identity fields. Everything else about a source — how it was drawn, what it
cost, what it dropped — lives in the adapter's manifest, not in this object.

Nothing is imported eagerly. ``get_adapter("molecules")`` pulls RDKit,
networkx, pandas and ``TextGraphDataset``; ``validate`` mode on the login node
must be able to import this package without any of them.
"""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

#: Adapter module names, resolved under this package. One entry per domain; the
#: trunk's graphqa / kgqa / relbench adapters join it as they land (DESIGN.md §9).
ADAPTERS = ("molecules",)


class AdapterError(ValueError):
    """An adapter that cannot be resolved, or one asked for a task it does not own."""


@runtime_checkable
class TaskSource(Protocol):
    """One (task, split, arm, pass) of built data, ready to batch.

    ``__getitem__`` returns an ``Example.to_item()`` dict: the
    ``TextGraphDataset`` item — ``text``, ``num_nodes``, ``prompt_node``,
    ``edges``, ``input_ids``, ``labels``, ``shortest_path_dists``,
    ``magnetic_V``, … — plus ``ds_label`` and the schema sidecar under
    ``schema.SIDECAR_KEY``. ``GraphCollatorV2`` reads named keys only, so the
    sidecar rides along for free and the mixture can route a batch by task
    without a registry lookup.
    """

    task: str
    split: str
    arm: str
    pass_id: int

    def __len__(self) -> int: ...

    def __getitem__(self, i: int) -> dict: ...

    def lengths(self) -> tuple:
        """``(num_nodes, num_tokens)``, one entry per example, for bucketing.

        ``num_tokens`` is the total over *all* nodes as ``TextGraphDataset``
        tokenized them — not the prompt node's length. That is the quantity
        ``tokens_per_step`` (D4.4) is a budget of, because every node's tokens
        are in the packed sequence.
        """


@runtime_checkable
class Adapter(Protocol):
    """The three functions of D3. Modules, not classes — there is no state."""

    def build(self, config, roles) -> None: ...

    def load(self, task: str, split: str, arm: str, pass_id: int = 0): ...

    def partition(self, config): ...


def get_adapter(name: str):
    """The adapter module called ``name``, imported on demand.

    A module rather than an instance: an adapter has no state that outlives a
    call, and importing lazily is what keeps ``schema`` / ``registry`` — and so
    ``validate`` mode — free of RDKit and torch.
    """
    if name not in ADAPTERS:
        raise AdapterError(
            f"{name!r}: no such adapter (have {ADAPTERS}). An adapter is a module "
            f"under src/generalist/adapters/ with build / load / partition.")
    module = importlib.import_module(f".{name}", __package__)
    missing = [fn for fn in ("build", "load", "partition")
               if not callable(getattr(module, fn, None))]
    if missing:
        raise AdapterError(
            f"{name!r}: adapter module is missing {missing}; D3 requires all three.")
    return module

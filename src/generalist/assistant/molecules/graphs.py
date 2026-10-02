"""A composed molecule row as the graph `mol/assistant` trains on.

`MoleculeDomain.example_builder` is what `pipeline/graphs.py` calls, and
`adapters/molecules.py` reads `named_atoms_for` and `shot_molecules` when it
builds the same examples for training, so the two cannot drift apart.
"""

from __future__ import annotations

from ..domain import Unbuildable
from ..facts import as_facts


def named_atoms_for(facts) -> list:
    """The atoms an assistant example is about, as **0-based RDKit indices**.

    Empty unless *every* drawn fact is atom-scoped. This mirrors
    `dataset.build_graph_example`: an atom-level question wires the prompt to
    the atoms it names, and a molecule-level one wires it to the whole
    molecule, because a prompt whose edges reach only one atom of a question
    about the whole molecule is pointing at the wrong thing. An assistant
    example can draw both kinds at once, and then the molecule is what it is
    about.

    **The indices are converted.** A fact sheet writes atoms 1-based, because
    that is the convention the Tier-A questions use ("atom 14"); the graph's
    node keys are RDKit's own 0-based indices. Wiring the prompt without
    subtracting one would point it at the neighbouring atom — silently, and
    only on the families that name an atom.
    """
    facts = as_facts(facts)
    if not facts or not all(f.atoms for f in facts):
        return []
    return sorted({a - 1 for f in facts for a in f.atoms})


def shot_molecules(row) -> list:
    """A composed row's demonstrations as ``(mol, question, answer)``.

    The molecule is rebuilt from the demonstration's partition key, which is the
    stereo-free canonical SMILES (`adapters.molecules.partition_key`) — the same
    string the §3 partition treats as the molecule's identity, so a demonstration
    cannot smuggle in a stereoisomer the partition holds at a different role.
    """
    from rdkit import Chem

    out = []
    for shot in row.get("shots") or []:
        mol = Chem.MolFromSmiles(shot["key"])
        if mol is None:                  # unparseable key: drop the shot, keep the row
            continue
        out.append((mol, shot["question"], shot["answer"]))
    return out


def example_builder(config):
    """``(build, tokenizer_name)`` for a generalist run config.

    ``build(row)`` returns ``(graph, n_shots)``, and raises `Unbuildable` for a
    row whose key RDKit cannot parse.
    """
    from rdkit import Chem

    from ...adapters.molecules import resolved_prompt_style
    from ....experiments.molecules.config import RunConfig as MolConfig
    from ....experiments.molecules.dataset import build_assistant_example
    from ..shots import question_text

    adapter = config.adapter_config()
    # `mol/assistant` is a corpus task with a free-text answer, so `task` here is
    # only a stand-in that carries the encoding: the scope of an example comes
    # from its own facts, through `named_atoms_for`, not from the task name.
    cfg = MolConfig(task="ring_count", arm="graph",
                    encoding=adapter.encoding,
                    stereo_tags=adapter.stereo_tags,
                    model_name=adapter.model_name,
                    question_node=adapter.question_node,
                    prompt_style=resolved_prompt_style(adapter)).validate()

    def build(row):
        mol = Chem.MolFromSmiles(row["key"])
        if mol is None:
            raise Unbuildable("unparseable key")
        shots = shot_molecules(row)
        graph = build_assistant_example(mol, question_text(row), row["answer"],
                                        named_atoms_for(row["facts"]), shots, cfg)
        return graph, len(shots)

    return build, adapter.model_name

"""The assistant set: graph-conditioned questions written from computed facts.

Layout:

* ``domain.py`` — the `Domain` contract and its registry. Everything that differs
  between subjects lives behind it;
* ``facts.py`` — `Fact` and the text helpers on fact sentences;
* ``intents.py`` — the draw: situation, task, facts, twist, brief;
* ``render.py`` — the deterministic reply the writer must keep the claims of;
* ``shots.py`` — the few-shot demonstration rules compose applies;
* ``molecules/`` — the first domain: fact sheet, vocabulary, prompts, checks,
  pool, situations, and the graph builder for ``mol/assistant``;
* ``pipeline/`` — the stages, in order: build, write (ask, voice), judge,
  accept, compose, graphs. ``pipeline/run.sh`` drives them end to end;
* ``analysis/`` — audits, calibration, scoring and the case study, run on a
  finished set.

Adding a domain: subclass `domain.Domain`, fill in its attributes and hooks
(`source` and `example_builder` are the two that touch data), and call
`domain.register` on an instance at import. Then either add the module to
`domain._MODULES` so ``--domain <name>`` finds it lazily, or import it before the
pipeline resolves the name. Every stage takes ``--domain``; without it, a stage
runs molecules.
"""

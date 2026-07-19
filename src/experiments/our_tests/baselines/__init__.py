"""
Baseline pipelines for the our_tests experiment (flat-LLM and LLaGA/RGLM).

Kept apart from the GTLM path deliberately: these consume *text* or LLaGA-format
datasets rather than the ``.gtds`` text-attributed graphs, so they share the data
*generators* (``kgqa_gen`` / ``family_gen``) but nothing downstream of them.
"""

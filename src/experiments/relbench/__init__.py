"""RelBench x GTLM experiment (see PLAN.md).

Predict a relational-database label at a timestamp by feeding the temporally-sampled rows to
GTLM as full text, with the foreign-key topology carried in the attention bias instead of
being compressed into per-row vectors.
"""

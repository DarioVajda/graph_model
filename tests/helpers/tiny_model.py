"""Tiny model configuration shared across GTLM v2 tests."""

BASE_CONFIG = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    intermediate_size=128, vocab_size=256, max_position_embeddings=256,
    pad_token_id=0, _attn_implementation="eager",
)

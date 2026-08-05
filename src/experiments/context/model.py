"""
Model construction for the context experiment — fresh, and from a checkpoint.

Both entry points that touch a GPU (``train`` and ``grid``) need the same GTLM
stack wired the same way, so it lives here once. Nothing in ``src/models/`` is
modified or subclassed: this is plain construction from ``RunConfig``.
"""

import json
import os


from transformers import AutoTokenizer

from ...models import GTLMLlamaConfig, GTLMLlamaForCausalLM
from ...models.io import load_bias_parameters
from ...utils import GraphCollatorV2


def build_config(cfg, **overrides):
    """The GTLM config for this run (bias modules, mask, backend)."""
    return GTLMLlamaConfig.from_pretrained(
        cfg.model_name, **cfg.bias_params(),
        k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        graph_attn_impl=cfg.graph_attn_impl,
        flex_compile_mode=cfg.compile_mode,
        # One compiled shape per (L bucket, N bucket, batch-size) family; the grid
        # has ~25 L buckets x 5 N buckets, and an exhausted dynamo cache is fatal
        # under flex (uncompiled flex has no working backward), so leave headroom.
        flex_cache_size_limit=256,
        **overrides,
    )


def build_model(cfg, device):
    """A fresh GTLM model + tokenizer, backbone frozen (LoRA is added later)."""
    config = build_config(cfg)
    model = GTLMLlamaForCausalLM.from_pretrained(
        cfg.model_name, config=config, graph_attn_impl=cfg.graph_attn_impl,
        torch_dtype=cfg.torch_dtype)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    return model, tokenizer


def load_checkpoint_model(checkpoint_path, cfg, device):
    """Rebuild a trained model from a checkpoint written by ``GraphTrainerV2``.

    The checkpoint holds the adapter (PEFT), the model config, and the graph-bias
    tensors in ``bias_parameters.pt``. The bias file must be loaded **after** the
    PEFT wrapping, because the saved parameter names carry PEFT's
    ``base_model.model.`` prefix (the same ordering ``_load_best_model`` relies on).
    """
    with open(os.path.join(checkpoint_path, "graph_bias_config.json")) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    config = GTLMLlamaConfig.from_pretrained(checkpoint_path)
    # Backend / compile knobs are run-time choices, not checkpoint properties.
    config.graph_attn_impl = cfg.graph_attn_impl
    config.flex_compile_mode = cfg.compile_mode
    config.flex_cache_size_limit = 256

    model = GTLMLlamaForCausalLM.from_pretrained(
        base_model_name, config=config, graph_attn_impl=cfg.graph_attn_impl,
        torch_dtype=cfg.torch_dtype)

    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)

    if load_bias_parameters(model, checkpoint_path) is None:
        raise FileNotFoundError(
            f"{checkpoint_path} has no bias_parameters.pt — the graph-bias weights "
            "would be freshly initialized, so this checkpoint cannot be scored.")

    # The collator is built from ``cfg``, the model from the checkpoint. Where the
    # two describe the SAME thing they must agree, or the model is scored on
    # inputs it was never trained on — silently, with plausible-looking numbers.
    for field in ("k_hop", "k_hop_directed", "magnetic_dim", "max_spd"):
        want, got = getattr(cfg, field, None), getattr(config, field, None)
        if want is not None and got is not None and want != got:
            raise ValueError(
                f"--{field.replace('_', '-')}={want} disagrees with the checkpoint's "
                f"{field}={got}. The collator follows the flags and the model follows the "
                "checkpoint, so scoring would silently mismatch. Pass the trained value.")

    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer, config


def build_collator(cfg, tokenizer, for_grid=False):
    """The v2 collator, with this experiment's explicit flex bucket ladders.

    Explicit ladders (rather than the default 512-multiple one) keep the number of
    compiled flex shapes equal to the number of cells: batching is cell-homogeneous,
    so every batch's raw length already sits on a cell length and block alignment is
    the only padding added.
    """
    magnetic_m = cfg.collate_magnetic_m
    len_buckets = cfg.grid_len_buckets() if for_grid else cfg.len_buckets()
    return GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop, k_hop_directed=cfg.k_hop_directed,
        magnetic_m=magnetic_m,
        pad_to_block=(cfg.graph_attn_impl == "flex"),
        len_buckets=len_buckets, node_buckets=cfg.node_buckets(),
        node_position_mode="reset", max_spd=cfg.max_spd,
    )

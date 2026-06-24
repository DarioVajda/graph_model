"""
Implementation dispatch (v0 vs v2) + trainable-parameter selection.

The two GTLM implementations differ in their model class, collator, forward
calling convention, and the names of their trainable graph-bias modules; this
module centralizes those differences behind a small ``parse_impl`` /
``build_*`` / ``forward_loss`` surface so the training and bench loops are
implementation-agnostic.
"""

import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate.utils import send_to_device

from ...models.legacy.modeling_gtlm_llama_v0 import GraphLlamaForCausalLM, GraphLlamaConfig
from ...models import GTLMLlamaForCausalLM, GTLMLlamaConfig
from ...utils import GraphCollator, GraphCollatorV2


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Parameter-name substrings of the trainable graph-bias parameters (everything
# else stays frozen). The two implementations name these modules differently:
#   v0  ...self_attn.{spd_weights, rrwp_proj, magnetic_*}
#   v2  ...self_attn.graph_bias.bias_modules.* (one umbrella prefix)
ACTIVE_PARAMS_V0 = ["spd_weights", "laplacian_weights", "rwse_weights", "rrwp_proj", "magnetic_"]
ACTIVE_PARAMS_V2 = ["graph_bias"]


def parse_impl(impl):
    """Split an IMPL string like 'v2-flex' into ('v2', 'flex') with validation."""
    version, _, backend = impl.partition("-")
    if version not in ("v0", "v2"):
        raise ValueError(f"Unknown version '{version}' in IMPL='{impl}' (expected v0|v2).")
    if backend not in ("eager", "flex"):
        raise ValueError(f"Unknown backend '{backend}' in IMPL='{impl}' (expected eager|flex).")
    if version == "v0" and backend != "eager":
        raise ValueError(f"v0 only supports the eager backend; got IMPL='{impl}'.")
    return version, backend


def active_params_for(impl):
    """Return the trainable-bias name substrings for ``impl``'s parameter layout."""
    version, _ = parse_impl(impl)
    return ACTIVE_PARAMS_V0 if version == "v0" else ACTIVE_PARAMS_V2


def build_model(impl, model_name, bias_params, k_hop, k_hop_directed, device,
                flex_compile_mode=None, dtype=torch.bfloat16):
    """Construct the model for ``impl`` and freeze every parameter.

    v0 is dense and eager-only (its config has no ``k_hop`` / backend fields); the
    ``k_hop`` / backend knobs are v2 concepts and are simply not passed to v0.

    ``dtype`` is the weight dtype (default ``bfloat16``). The backbone is frozen,
    so loading it in bf16 halves its footprint and runs every matmul/einsum —
    including the O(N²·M) magnetic-bias einsums — in bf16 rather than fp32; this
    is what the flex benchmark suite measured. Pass ``dtype=torch.float32`` to
    reproduce the old full-precision numbers.
    """
    version, backend = parse_impl(impl)
    if version == "v0":
        config = GraphLlamaConfig.from_pretrained(model_name, **bias_params)
        model = GraphLlamaForCausalLM.from_pretrained(
            model_name, config=config, attn_implementation="eager", torch_dtype=dtype)
    else:
        extra = dict(k_hop=k_hop, k_hop_directed=k_hop_directed, graph_attn_impl=backend)
        if flex_compile_mode is not None:
            extra["flex_compile_mode"] = flex_compile_mode
        config = GTLMLlamaConfig.from_pretrained(model_name, **bias_params, **extra)
        model = GTLMLlamaForCausalLM.from_pretrained(
            model_name, config=config, graph_attn_impl=backend, torch_dtype=dtype)

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def build_collator(impl, tokenizer, pad_token_id, bias_params, k_hop, k_hop_directed):
    """Build the collator matching ``impl`` (v0 ragged batch vs v2 packed batch)."""
    version, backend = parse_impl(impl)
    if version == "v0":
        return GraphCollator()
    magnetic_m = bias_params["magnetic_dim"] if bias_params.get("magnetic") else 0
    return GraphCollatorV2(
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        k_hop=k_hop,
        k_hop_directed=k_hop_directed,
        magnetic_m=magnetic_m,
        pad_to_block=(backend == "flex"),   # flex needs block-aligned L and bucketed N
    )


def forward_loss(impl, model, batch, device):
    """Run a single forward and return the loss, using the right calling convention."""
    version, _ = parse_impl(impl)
    batch = send_to_device(batch, device)
    if version == "v0":
        labels = batch.pop("labels", None)
        if labels is None:
            raise ValueError("Collated batch is missing 'labels'.")
        outputs = model(input_ids=None, input_graph_batch=batch, labels=labels)
    else:
        outputs = model(**batch)
    return outputs.loss


def select_active_params(model, active_params=None, lora=None):
    """
    Applies LoRA if a configuration dictionary is provided.
    Sets requires_grad=True for parameters whose names contain any of the substrings in active_params.
    If active_params is None, no additional parameters are unfrozen.
    If active_params is "all", all parameters are set to requires_grad=True.
    """
    if lora is not None:
        print("Applying LoRA with config:", lora)
        lora_config = LoraConfig(
            r=lora.get("r", 8),
            lora_alpha=lora.get("lora_alpha", 16),
            target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=lora.get("lora_dropout", 0.05),
            bias=lora.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if active_params == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif active_params is not None:
        for name, param in model.named_parameters():
            if any(active_param in name for active_param in active_params):
                param.requires_grad = True

    return model


def print_trainable_parameters(model):
    """Prints the number of trainable parameters, splitting LoRA vs custom bias params."""
    trainable_lora_params = 0
    trainable_custom_params = 0
    all_param = 0

    print("List of custom active parameters (requires_grad=True):")
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            if "lora" in name.lower():
                trainable_lora_params += num_params
                print(f" - {name} ({num_params:,} params) [LoRA Adapter]")
            else:
                trainable_custom_params += num_params
                print(f" - {name} ({num_params:,} params)")

    total_trainable = trainable_lora_params + trainable_custom_params
    print("\n" + "=" * 50)
    print("TRAINABLE PARAMETER SUMMARY")
    print("=" * 50)
    print(f"LoRA Adapters:       {trainable_lora_params:>15,}")
    print(f"Custom Graph Biases: {trainable_custom_params:>15,}")
    print("-" * 50)
    print(f"Total Trainable:     {total_trainable:>15,}")
    print(f"Total Model Params:  {all_param:>15,}")
    print(f"Trainable %:         {100 * total_trainable / all_param:>14.4f}%")
    print("=" * 50 + "\n")

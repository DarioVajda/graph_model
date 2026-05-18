import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

from ..models.llama_k_hop import KHopLlamaConfig, KHopGraphLlamaForCausalLM


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(model_name, device, bias_params, k_hop=0, attn_implementation="eager"):
    config = KHopLlamaConfig.from_pretrained(model_name, k_hop=k_hop, **bias_params)
    model = KHopGraphLlamaForCausalLM.from_pretrained(
        model_name, config=config, attn_implementation=attn_implementation
    )
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def select_active_params(model, active_params=None, lora=None):
    """
    Applies LoRA if a configuration dict is provided.
    Sets requires_grad=True for parameters whose names contain any substring in active_params.
    Pass active_params="all" to unfreeze everything, or None to leave all frozen.
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
            if any(p in name for p in active_params):
                param.requires_grad = True

    return model


def print_trainable_parameters(model):
    trainable_lora = 0
    trainable_custom = 0
    total = 0

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            if "lora" in name.lower():
                trainable_lora += n
                print(f"  {name} ({n:,}) [LoRA]")
            else:
                trainable_custom += n
                print(f"  {name} ({n:,})")

    trainable = trainable_lora + trainable_custom
    print("\n" + "=" * 50)
    print("TRAINABLE PARAMETER SUMMARY")
    print("=" * 50)
    print(f"LoRA Adapters:       {trainable_lora:>15,}")
    print(f"Custom Graph Biases: {trainable_custom:>15,}")
    print("-" * 50)
    print(f"Total Trainable:     {trainable:>15,}")
    print(f"Total Model Params:  {total:>15,}")
    print(f"Trainable %:         {100 * trainable / total:>14.4f}%")
    print("=" * 50 + "\n")

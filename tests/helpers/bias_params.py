"""Old<->new graph-bias parameter correspondence.

Shared by the legacy compatibility tests and the GTLM v2 tests.
"""

from src.models.legacy.modeling_gtlm_llama_v0 import GraphAttnBiasConfig


def iter_bias_param_pairs(old_model, new_model, bias_cfg: GraphAttnBiasConfig):
    """
    Yield ``(name, old_param, new_param)`` for every graph-bias leaf parameter,
    pairing the old model's flat ``self_attn`` attributes (e.g. ``spd_weights``)
    with the new model's ``self_attn.graph_bias.bias_modules[i]`` submodules,
    ordered per BIAS_TYPES = [SPDBias, LaplacianBias, RWSEBias, RRWPBias, MagneticBias].

    Single source of truth for the old<->new bias-parameter correspondence: used
    both to transfer weights (copy ``.data``) and to compare gradients (`.grad`).
    """
    for i in range(len(old_model.model.layers)):
        old_attn = old_model.model.layers[i].self_attn
        new_attn = new_model.model.layers[i].self_attn
        tag = f"layers.{i}.self_attn"
        mod_idx = 0

        if bias_cfg.spd:
            yield (f"{tag}.spd_weights", old_attn.spd_weights,
                   new_attn.graph_bias.bias_modules[mod_idx].weights)
            mod_idx += 1

        if bias_cfg.laplacian:
            yield (f"{tag}.laplacian_weights", old_attn.laplacian_weights,
                   new_attn.graph_bias.bias_modules[mod_idx].weights)
            mod_idx += 1

        if bias_cfg.rwse:
            yield (f"{tag}.rwse_weights", old_attn.rwse_weights,
                   new_attn.graph_bias.bias_modules[mod_idx].weights)
            mod_idx += 1

        if bias_cfg.rrwp:
            new_mod = new_attn.graph_bias.bias_modules[mod_idx]
            for j in range(len(old_attn.rrwp_proj)):
                if hasattr(old_attn.rrwp_proj[j], 'weight'):
                    yield (f"{tag}.rrwp_proj.{j}.weight",
                           old_attn.rrwp_proj[j].weight, new_mod.proj[j].weight)
                    yield (f"{tag}.rrwp_proj.{j}.bias",
                           old_attn.rrwp_proj[j].bias, new_mod.proj[j].bias)
            mod_idx += 1

        if bias_cfg.magnetic:
            new_mod = new_attn.graph_bias.bias_modules[mod_idx]
            # lambda_lin: R -> R^head_dim
            yield (f"{tag}.magnetic_lambda_lin.weight",
                   old_attn.magnetic_lambda_lin.weight, new_mod.lambda_lin.weight)
            yield (f"{tag}.magnetic_lambda_lin.bias",
                   old_attn.magnetic_lambda_lin.bias, new_mod.lambda_lin.bias)
            # deep_set: R^(2*head_dim) -> R^magnetic_dim
            for j in range(len(old_attn.magnetic_lambda_deep_set)):
                if hasattr(old_attn.magnetic_lambda_deep_set[j], 'weight'):
                    yield (f"{tag}.magnetic_lambda_deep_set.{j}.weight",
                           old_attn.magnetic_lambda_deep_set[j].weight, new_mod.deep_set[j].weight)
                    yield (f"{tag}.magnetic_lambda_deep_set.{j}.bias",
                           old_attn.magnetic_lambda_deep_set[j].bias, new_mod.deep_set[j].bias)
            # proj: R^(2*magnetic_dim) -> R^num_heads
            for j in range(len(old_attn.magnetic_bias_proj)):
                if hasattr(old_attn.magnetic_bias_proj[j], 'weight'):
                    yield (f"{tag}.magnetic_bias_proj.{j}.weight",
                           old_attn.magnetic_bias_proj[j].weight, new_mod.proj[j].weight)
                    yield (f"{tag}.magnetic_bias_proj.{j}.bias",
                           old_attn.magnetic_bias_proj[j].bias, new_mod.proj[j].bias)
            mod_idx += 1


def transfer_bias_weights(old_model, new_model, bias_cfg: GraphAttnBiasConfig):
    """
    Copy graph-bias parameters from old_model into new_model.

    Old model stores them directly on self_attn (e.g. spd_weights).
    New model stores them in self_attn.graph_bias.bias_modules[i], ordered
    according to BIAS_TYPES = [SPDBias, LaplacianBias, RWSEBias, RRWPBias, MagneticBias].
    """
    for _name, old_p, new_p in iter_bias_param_pairs(old_model, new_model, bias_cfg):
        new_p.data.copy_(old_p.data)



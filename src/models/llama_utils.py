import os
import torch

def save_bias_parameters(model, save_dir, params):
    """
    Save the bias parameters of the model to a specified directory.

    Args:
        model: The PyTorch model containing bias parameters.
        save_dir: The directory where the bias parameters will be saved.
        params: A list of parameter names (actually, substrings of parameter names) to identify which bias parameters to save.
    """
    os.makedirs(save_dir, exist_ok=True)

    custom_state_dict = {}
    for name, param in model.named_parameters():
        if any(param_name in name for param_name in params):
            custom_state_dict[name] = param.data

    save_path = os.path.join(save_dir, "bias_parameters.pt")
    torch.save(custom_state_dict, save_path)

def load_bias_parameters(model, path):
    bias_path = os.path.join(path, "bias_parameters.pt")
    if os.path.exists(bias_path):
        state_dict = torch.load(bias_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
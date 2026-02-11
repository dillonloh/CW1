import os
import torch

def save_model(model, path, extra=None):
    """
    Save a PyTorch model safely using state_dict.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)


def load_model(model, path, map_location="cpu"):
    """
    Load a PyTorch model safely from state_dict.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("extra", None)
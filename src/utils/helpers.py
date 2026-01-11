import os
import json
import torch


def create_directory(path):
    """creates directory if it doesn't exist."""

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"directory created: {path}")

    else:
        print(f"directory already exists: {path}")


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """saves model checkpoint to disk."""

    # create checkpoint directory if it doesn't exist.
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # save checkpoint.
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }

    torch.save(checkpoint, filepath)
    print(f"checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """loads model checkpoint from disk."""

    # check if checkpoint exists.
    if not os.path.exists(filepath):
        print(f"checkpoint not found: {filepath}")
        return None

    # load checkpoint.
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"checkpoint loaded: {filepath}")

    return epoch, loss
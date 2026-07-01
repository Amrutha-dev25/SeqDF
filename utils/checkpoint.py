import os
import torch


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, checkpoint_dir: str, tag: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{tag}_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
    }, path)
    print(f"Saved checkpoint -> {path}")
    return path


def load_checkpoint(model, path: str, optimizer=None, map_location="cuda"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']}, metrics={ckpt['metrics']})")
    return ckpt


def find_latest_checkpoint(checkpoint_dir: str, tag: str) -> str:
    """Finds the highest-epoch checkpoint file matching a given tag prefix."""
    matches = [f for f in os.listdir(checkpoint_dir) if f.startswith(tag) and f.endswith(".pt")]
    if not matches:
        raise FileNotFoundError(f"No checkpoints found for tag '{tag}' in {checkpoint_dir}")

    def epoch_of(fname):
        # filename pattern: {tag}_epoch{N}.pt
        return int(fname.replace(tag + "_epoch", "").replace(".pt", ""))

    matches.sort(key=epoch_of)
    return os.path.join(checkpoint_dir, matches[-1])

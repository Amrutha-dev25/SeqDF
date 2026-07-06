import os
import torch


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, checkpoint_dir: str, tag: str, scheduler=None):
    """
    Writes to a temp file first, then atomically renames into place. This avoids
    two failure modes we hit in practice:
      1. A crash mid-write (e.g. disk full) leaving a corrupt/truncated .pt file
         that would silently break a later resume (torch.load raising on a
         half-written zip archive).
      2. A checkpoint-write failure raising an exception that kills the entire
         training process AFTER a full epoch of real GPU work already completed
         and validation already ran - losing that epoch's results for something
         that has nothing to do with model correctness.
    If the write fails (e.g. disk full), this now prints a clear warning and
    returns None instead of raising - training continues to the next epoch.
    You should still fix the underlying cause (free disk space) since repeated
    failures mean you are silently not checkpointing at all.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{tag}_epoch{epoch}.pt")
    tmp_path = path + ".tmp"
    try:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
        }, tmp_path)
        os.replace(tmp_path, path)  # atomic on same filesystem
        print(f"Saved checkpoint -> {path}")
        return path
    except OSError as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        print(f"[WARNING] Failed to save checkpoint for epoch {epoch}: {e}")
        print(f"[WARNING]   This is very likely low disk space - check with "
              f"'Get-PSDrive C' in PowerShell. Training will continue, but this "
              f"epoch was NOT checkpointed.")
        return None


def load_checkpoint(model, path: str, optimizer=None, scheduler=None, map_location="cuda"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']}, metrics={ckpt['metrics']})")
    return ckpt


def find_best_metric_so_far(checkpoint_dir: str, tag: str, metric_key: str, map_location="cuda") -> float:
    """
    Scans every saved per-epoch checkpoint for a tag and returns the best value
    seen for `metric_key` in its stored `metrics` dict. Used on resume to recover
    `best_exact_match` exactly, instead of guessing or re-establishing it from
    scratch on the next validation pass.
    """
    best = 0.0
    if not os.path.isdir(checkpoint_dir):
        return best
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith(tag) and fname.endswith(".pt"):
            ckpt = torch.load(os.path.join(checkpoint_dir, fname), map_location=map_location, weights_only=False)
            val = ckpt.get("metrics", {}).get(metric_key, 0.0)
            if val > best:
                best = val
    return best


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
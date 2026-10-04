"""Atomic, rotating checkpoints for formal teacher training."""
import os
import time
from pathlib import Path

import torch


def _replace_with_retry(source, destination, *, attempts=10, initial_delay=0.25):
    """Retry transient Windows/WSL sharing violations without hiding hard failures."""
    if attempts < 1 or initial_delay < 0:
        raise ValueError("invalid checkpoint replacement retry settings")
    delay = initial_delay
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


def steps_per_epoch(sample_count, batch_size, *, drop_last):
    if sample_count < 1 or batch_size < 1:
        raise ValueError("sample_count and batch_size must be positive")
    if drop_last:
        return sample_count // batch_size
    return (sample_count + batch_size - 1) // batch_size


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    _replace_with_retry(temporary, path)


def save_rotating_checkpoint(output_dir, payload, *, keep_previous=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "latest.pt"
    previous = output_dir / "previous.pt"
    temporary = output_dir / "latest.pt.tmp"
    torch.save(payload, temporary)
    if keep_previous and latest.exists():
        _replace_with_retry(latest, previous)
    _replace_with_retry(temporary, latest)
    return latest


def save_ema_milestone(output_dir, *, subset, epoch, global_step, config, ema_model):
    path = Path(output_dir) / f"ema_epoch_{epoch:04d}.pt"
    payload = {
        "format": 1,
        "kind": "ema_model_only",
        "subset": subset,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "ema_model": ema_model.state_dict(),
    }
    _atomic_torch_save(payload, path)
    return path

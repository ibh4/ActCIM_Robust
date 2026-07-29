"""Shared CIFAR-10 test-set loader that does not depend on torchvision
(the local torchvision build is incompatible with the installed torch)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10_test(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized test images (N,3,32,32) float32 and labels (N,) int64."""
    batch_file = Path(root) / "cifar-10-batches-py" / "test_batch"
    with open(batch_file, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
    data = entry["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.asarray(entry["labels"], dtype=np.int64)
    mean = np.asarray(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    data = (data - mean) / std
    return torch.from_numpy(data), torch.from_numpy(labels)


def iterate_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int = 500):
    for i in range(0, x.shape[0], batch_size):
        yield x[i : i + batch_size], y[i : i + batch_size]

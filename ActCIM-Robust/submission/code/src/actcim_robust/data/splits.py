from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets

from ..constants import TRAIN_SIZE, VAL_SIZE
from .utils import ensure_raw_data, create_dataloader, get_cifar10_transforms


def create_splits(data_dir="data", seed=42):
    raw_dir = ensure_raw_data(data_dir)
    splits_dir = Path(data_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_indices_path = splits_dir / "cifar10_train_indices.npy"
    val_indices_path = splits_dir / "cifar10_val_indices.npy"

    if train_indices_path.exists() and val_indices_path.exists():
        return (
            np.load(train_indices_path).tolist(),
            np.load(val_indices_path).tolist(),
        )

    full_train = datasets.CIFAR10(
        root=raw_dir, train=True, download=False, transform=None
    )
    total = len(full_train)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:TRAIN_SIZE]
    val_indices = indices[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

    np.save(train_indices_path, np.array(train_indices))
    np.save(val_indices_path, np.array(val_indices))

    return train_indices, val_indices


def load_splits(data_dir="data", seed=42):
    splits_dir = Path(data_dir) / "splits"
    train_path = splits_dir / "cifar10_train_indices.npy"
    val_path = splits_dir / "cifar10_val_indices.npy"

    if not train_path.exists() or not val_path.exists():
        return create_splits(data_dir=data_dir, seed=seed)

    return (
        np.load(train_path).tolist(),
        np.load(val_path).tolist(),
    )


def get_split_loaders(batch_size=128, num_workers=4, data_dir="data", seed=42):
    raw_dir = ensure_raw_data(data_dir)

    train_transform = get_cifar10_transforms(train=True)
    test_transform = get_cifar10_transforms(train=False)

    full_train = datasets.CIFAR10(
        root=raw_dir, train=True, download=False, transform=train_transform
    )
    val_full = datasets.CIFAR10(
        root=raw_dir, train=True, download=False, transform=test_transform
    )
    test_dataset = datasets.CIFAR10(
        root=raw_dir, train=False, download=False, transform=test_transform
    )

    train_indices, val_indices = load_splits(data_dir=data_dir, seed=seed)

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(val_full, val_indices)

    train_loader = create_dataloader(train_dataset, batch_size, num_workers, train=True)
    val_loader = create_dataloader(val_dataset, batch_size, num_workers, train=False)
    test_loader = create_dataloader(test_dataset, batch_size, num_workers, train=False)

    return train_loader, val_loader, test_loader

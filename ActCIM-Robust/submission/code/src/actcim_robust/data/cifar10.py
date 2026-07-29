import torch
from torch.utils.data import Subset
from torchvision import datasets

from ..constants import TRAIN_SIZE, VAL_SIZE
from .utils import ensure_raw_data, create_dataloader
from .utils import get_cifar10_transforms as _get_cifar10_transforms


def _fallback_split(total_size, train_size, val_size, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    return indices[:train_size], indices[train_size:train_size + val_size]


def get_cifar10_loaders(
    batch_size=128,
    num_workers=4,
    data_dir="data",
    use_splits=True,
    seed=42,
):
    raw_dir = ensure_raw_data(data_dir)

    train_transform = _get_cifar10_transforms(train=True)
    test_transform = _get_cifar10_transforms(train=False)

    full_train = datasets.CIFAR10(
        root=raw_dir, train=True, download=False, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=raw_dir, train=False, download=False, transform=test_transform
    )

    test_loader = create_dataloader(test_dataset, batch_size, num_workers, train=False)

    if use_splits:
        from .splits import load_splits
        train_indices, val_indices = load_splits(data_dir, seed=seed)
    else:
        train_indices, val_indices = _fallback_split(
            len(full_train), TRAIN_SIZE, VAL_SIZE, seed
        )

    train_dataset = Subset(full_train, train_indices)
    val_dataset_full = datasets.CIFAR10(
        root=raw_dir, train=True, download=False, transform=test_transform
    )
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = create_dataloader(train_dataset, batch_size, num_workers, train=True)
    val_loader = create_dataloader(val_dataset, batch_size, num_workers, train=False)

    return train_loader, val_loader, test_loader


get_cifar10_transforms = _get_cifar10_transforms

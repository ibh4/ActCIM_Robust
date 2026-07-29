import os
import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ..constants import CIFAR10_MEAN, CIFAR10_STD

_CIFAR10_TAR_PATH = Path(
    r"I:\比赛项目\存算一体高校挑战赛\cifar-10-python.tar.gz"
)


def ensure_raw_data(data_dir):
    raw_dir = Path(data_dir) / "raw"
    cifar10_dir = raw_dir / "cifar-10-batches-py"

    if cifar10_dir.exists():
        batch_files = list(cifar10_dir.glob("data_batch_*"))
        if len(batch_files) >= 5:
            return raw_dir

    raw_dir.mkdir(parents=True, exist_ok=True)

    if not _CIFAR10_TAR_PATH.exists():
        raise FileNotFoundError(
            f"CIFAR-10 tar.gz not found at {_CIFAR10_TAR_PATH}. "
            "Please download it from https://www.cs.toronto.edu/~kriz/cifar.html"
        )

    with tarfile.open(_CIFAR10_TAR_PATH, "r:gz") as tar:
        tar.extractall(path=raw_dir)

    extracted_dir = raw_dir / "cifar-10-batches-py"
    if not extracted_dir.exists():
        for item in raw_dir.iterdir():
            if item.is_dir() and item.name != "cifar-10-batches-py":
                if (item / "data_batch_1").exists():
                    os.rename(str(item), str(extracted_dir))
                    break

    return raw_dir


def create_dataloader(dataset, batch_size, num_workers, train):
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=train,
        pin_memory=True,
    )

    if os.name == "nt":
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = True if (train and num_workers > 0) else False
        try:
            loader = DataLoader(dataset, **loader_kwargs)
            _ = next(iter(loader))
            return loader
        except (RuntimeError, OSError):
            pass

    loader_kwargs["num_workers"] = 0
    loader_kwargs["persistent_workers"] = False
    return DataLoader(dataset, **loader_kwargs)


def get_cifar10_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])

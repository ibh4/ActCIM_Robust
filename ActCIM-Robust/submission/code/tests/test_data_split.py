import os
import pytest
from actcim_robust.data.splits import create_splits, load_splits


def test_splits_create_and_load(data_dir="data"):
    train_idx, val_idx = create_splits(data_dir=data_dir, seed=42)
    assert len(train_idx) == 45000
    assert len(val_idx) == 5000
    assert len(set(train_idx) & set(val_idx)) == 0
    assert os.path.exists(os.path.join(data_dir, "splits", "cifar10_train_indices.npy"))
    assert os.path.exists(os.path.join(data_dir, "splits", "cifar10_val_indices.npy"))


def test_splits_reproducibility():
    idx1, _ = create_splits(seed=42)
    idx2, _ = create_splits(seed=42)
    assert idx1 == idx2

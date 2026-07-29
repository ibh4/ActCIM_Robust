import os
import pytest
from actcim_robust.config import load_config
from actcim_robust.constants import PROJECT_ROOT


def test_load_base_config():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    assert cfg.project == "ActCIM-Robust"
    assert cfg.data.batch_size == 128
    assert cfg.optimizer.learning_rate == 0.1


def test_load_smoke_config():
    cfg = load_config(os.path.join(PROJECT_ROOT, "configs", "smoke.yaml"))
    assert cfg.model == "tinycnn"
    assert cfg.training.epochs == 1
    assert cfg.data.train_subset == 2048

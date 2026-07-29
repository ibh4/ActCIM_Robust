from __future__ import annotations

from pathlib import Path

import torch

from .trainer import Trainer
from ..data.cifar10 import get_cifar10_loaders
from ..models.factory import create_model
from ..reproducibility import set_seed, set_deterministic
from ..utils.paths import ensure_dir
from ..environment import save_environment_info


def train_baseline(
    config,
    seed: int = 42,
    profile: str = "fast",
    data_dir: str = "data",
    results_dir: str = "results",
) -> dict:
    set_seed(seed)
    set_deterministic()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_dir = ensure_dir(Path(results_dir) / "baseline" / f"seed_{seed}")
    save_environment_info(exp_dir)

    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    train_loader, val_loader, _ = get_cifar10_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
        seed=seed,
    )

    model_name = config.model
    num_classes = config.num_classes
    model = create_model(model_name, num_classes=num_classes)

    training_config = config.training.to_dict()
    training_config['optimizer'] = config.optimizer.to_dict()
    training_config['scheduler'] = config.scheduler.to_dict()
    training_config['batch_size'] = config.data.batch_size

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        seed=seed,
        exp_dir=exp_dir,
    )

    summary = trainer.train()

    return summary

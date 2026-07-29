from __future__ import annotations

from pathlib import Path

import torch

from .trainer import Trainer
from .checkpoint import load_model_weights
from ..data.cifar10 import get_cifar10_loaders
from ..models.factory import create_model
from ..nonlinearity.controller import NonlinearityController
from ..reproducibility import set_seed, set_deterministic
from ..utils.paths import ensure_dir, get_exp_dir
from ..environment import save_environment_info


def train_fixed_nat(
    config,
    checkpoint_path: str | Path,
    seed: int = 42,
    profile: str = "fast",
    data_dir: str = "data",
    results_dir: str = "results",
) -> dict:
    set_seed(seed)
    set_deterministic()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = ensure_dir(Path(results_dir) / "fixed_nat")
    exp_dir = get_exp_dir(results_dir, "fixed_nat", seed)
    save_environment_info(exp_dir)

    try:
        batch_size = config.data.batch_size
        num_workers = config.data.num_workers
    except AttributeError:
        batch_size = config.get("batch_size", 128)
        num_workers = config.get("num_workers", 4)

    train_loader, val_loader, _ = get_cifar10_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
        seed=seed,
    )

    try:
        model_name = config.model
        num_classes = config.num_classes
    except AttributeError:
        model_cfg = config.get("model", {})
        model_name = model_cfg.get("name", "resnet18_cifar") if hasattr(model_cfg, 'get') else "resnet18_cifar"
        num_classes = model_cfg.get("num_classes", 10) if hasattr(model_cfg, 'get') else 10
    model = create_model(model_name, num_classes=num_classes)

    if checkpoint_path and Path(checkpoint_path).exists():
        load_model_weights(model, checkpoint_path, device=device)

    try:
        nat_cfg = config.nat
        alpha = nat_cfg.alpha
    except AttributeError:
        nat_cfg = config.get("nat", {})
        alpha = nat_cfg.get("alpha", 0.4) if hasattr(nat_cfg, 'get') else 0.4

    try:
        training_config = config.training.to_dict()
    except AttributeError:
        training_cfg = config.get("training", {})
        training_config = training_cfg.to_dict() if hasattr(training_cfg, 'to_dict') else training_cfg
    try:
        training_config['optimizer'] = config.optimizer.to_dict()
        training_config['scheduler'] = config.scheduler.to_dict()
    except AttributeError:
        pass

    controller = NonlinearityController(model)
    controller.set_global_alpha(alpha)
    controller.enable_all()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        seed=seed,
        exp_dir=exp_dir,
    )

    summary = trainer.train(controller=controller)

    return summary

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


def train_random_nat(
    config,
    checkpoint_path: str | Path | None = None,
    seed: int = 42,
    profile: str = "fast",
    data_dir: str = "data",
    results_dir: str = "results",
) -> dict:
    set_seed(seed)
    set_deterministic()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = ensure_dir(Path(results_dir) / "random_nat")
    exp_dir = get_exp_dir(results_dir, "random_nat", seed)
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
        alpha_max = nat_cfg.get("alpha_max", 0.5) if hasattr(nat_cfg, 'get') else 0.5
    except AttributeError:
        nat_cfg = config.get("nat", {})
        alpha_max = nat_cfg.get("alpha_max", 0.5) if hasattr(nat_cfg, 'get') else 0.5

    controller = NonlinearityController(model)
    controller.enable_all()

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

    class RandomNATTrainer(Trainer):
        def train_epoch(self, epoch, controller=None):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            alpha_sum = 0.0
            alpha_count = 0

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                alpha = (torch.rand(1).item() * 2 - 1) * alpha_max
                if controller is not None:
                    controller.set_global_alpha(alpha)
                    controller.enable_all()

                alpha_sum += abs(alpha)
                alpha_count += 1

                self.optimizer.zero_grad()

                if self.scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(inputs)
                        loss = torch.nn.functional.cross_entropy(outputs, targets)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if controller is not None:
                    controller.disable_all()

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss = running_loss / total
            train_acc = 100.0 * correct / total

            return {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "alpha_mean": alpha_sum / max(alpha_count, 1),
            }

    trainer = RandomNATTrainer(
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

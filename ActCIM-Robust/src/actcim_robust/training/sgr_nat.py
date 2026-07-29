from __future__ import annotations

import random
import json
from pathlib import Path

import torch
import torch.nn as nn

from .trainer import Trainer
from .losses import classification_loss, consistency_loss
from .checkpoint import load_model_weights, save_checkpoint
from ..data.cifar10 import get_cifar10_loaders
from ..models.factory import create_model
from ..nonlinearity.controller import NonlinearityController
from ..nonlinearity.scheduler import (
    CurriculumAlphaScheduler,
    SensitivityProbabilityCalculator,
)
from ..reproducibility import set_seed, set_deterministic
from ..utils.logging import setup_logging
from ..utils.paths import ensure_dir, get_exp_dir
from ..utils.serialization import save_json, save_metrics_jsonl
from ..utils.timing import format_duration
from ..environment import save_environment_info


class SGRNATTrainer(Trainer):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device,
        seed,
        exp_dir,
        sensitivity_scores,
        controller,
        alpha_scheduler,
        prob_calculator,
    ):
        super().__init__(model, train_loader, val_loader, config, device, seed, exp_dir)
        self.sensitivity_scores = sensitivity_scores
        self.controller = controller
        self.alpha_scheduler = alpha_scheduler
        self.prob_calculator = prob_calculator

        self.lambda_cons = config.get("lambda_cons", 0.5)
        self.clean_ce_weight = config.get("clean_ce_weight", 0.25)
        self.cons_temperature = config.get("cons_temperature", 2.0)
        self.detach_clean = config.get("detach_clean", True)

        self._batch_stats_log: list[dict] = []
        self._batch_counter = 0

    def _get_clean_logits(self, inputs):
        with torch.no_grad():
            self.model.eval()
            clean_logits = self.model(inputs)
        return clean_logits

    def train_epoch(self, epoch, controller=None):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        alpha_sum = 0.0
        alpha_std_sum = 0.0
        injection_count = 0
        total_batches = 0

        alpha_global = self.alpha_scheduler.get_alpha(epoch)
        self.alpha_scheduler.step()
        probs = self.prob_calculator.get_probabilities()

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            clean_logits = self._get_clean_logits(inputs)

            batch_config = self.controller.sample_batch_configuration(
                probabilities=probs,
                alpha_range=(0.0, alpha_global),
            )

            self.model.train()
            self.controller.disable_all()
            for layer_name, alpha_val in batch_config.items():
                self.controller.set_layer_alpha(layer_name, alpha_val)
                self.controller.enable_layers([layer_name])

            alphas = list(batch_config.values())
            if alphas:
                alpha_sum += sum(abs(a) for a in alphas)
                alpha_std_sum += (sum((a - sum(alphas) / len(alphas)) ** 2 for a in alphas) / len(alphas)) ** 0.5
                injection_count += len(alphas)

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    noisy_logits = self.model(inputs)
                    loss_dict = self._compute_loss(noisy_logits, targets, clean_logits)
                    loss = loss_dict["total"]
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                noisy_logits = self.model(inputs)
                loss_dict = self._compute_loss(noisy_logits, targets, clean_logits)
                loss = loss_dict["total"]
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = noisy_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            inj_rate = len(batch_config) / len(self.controller.get_layer_names())
            self._log_batch_stats(epoch, batch_config, inj_rate, alpha_global, loss_dict)

            self.controller.disable_all()
            total_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
        }
        if total_batches > 0:
            metrics["alpha_mean"] = alpha_sum / max(injection_count, 1)
            metrics["alpha_std"] = alpha_std_sum / max(total_batches, 1)
            metrics["injection_rate"] = injection_count / (total_batches * len(self.controller.get_layer_names()))

        return metrics

    def _compute_loss(self, noisy_logits, targets, clean_logits):
        loss_ce_noisy = classification_loss(noisy_logits, targets, self.label_smoothing)
        loss_ce_clean = classification_loss(clean_logits, targets, self.label_smoothing) if clean_logits is not None else 0.0
        cons = consistency_loss(clean_logits, noisy_logits, self.cons_temperature, self.detach_clean)

        total = loss_ce_noisy + self.clean_ce_weight * loss_ce_clean + self.lambda_cons * cons

        return {
            "ce_noisy": loss_ce_noisy,
            "ce_clean": loss_ce_clean,
            "cons": cons,
            "total": total,
        }

    def _log_batch_stats(self, epoch, batch_config, injection_rate, alpha_global, loss_dict):
        self._batch_counter += 1
        if self._batch_counter % 10 == 0:
            stat = {
                "epoch": epoch,
                "batch": self._batch_counter,
                "injection_rate": injection_rate,
                "alpha_global": alpha_global,
                "n_layers_injected": len(batch_config),
            }
            self._batch_stats_log.append(stat)

    def train(self, epochs=None, controller=None, checkpoint_path=None):
        summary = super().train(epochs=epochs, controller=controller, checkpoint_path=checkpoint_path)
        save_json(self._batch_stats_log, self.exp_dir / "batch_stats.json")
        return summary


def _build_sensitivity_scores(config, model, checkpoint_path, device):
    try:
        sens_cfg = config.sensitivity
        layer_name_list = getattr(config, "layer_names", [])
    except AttributeError:
        sens_cfg = config.get("sensitivity", {})
        layer_name_list = config.get("layer_names", [])

    default_score = sens_cfg.get("default_score", 0.5) if hasattr(sens_cfg, 'get') else 0.5
    score_source = sens_cfg.get("source", "config") if hasattr(sens_cfg, 'get') else "config"

    if score_source == "file":
        scores_path = sens_cfg.get("path", "") if hasattr(sens_cfg, 'get') else ""
        if scores_path and Path(scores_path).exists():
            with open(scores_path, "r") as f:
                loaded = json.load(f)
            return loaded.get("scores", loaded)

    if score_source == "config":
        scores = sens_cfg.get("scores", {}) if hasattr(sens_cfg, 'get') else {}
        if scores:
            return scores

    temp_controller = NonlinearityController(model)
    all_names = temp_controller.get_layer_names()
    return {name: default_score for name in all_names}


def train_sgr_nat(
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

    results_dir = ensure_dir(Path(results_dir) / "sgr_nat")
    exp_dir = get_exp_dir(results_dir, "sgr_nat", seed)
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

    sensitivity_scores = _build_sensitivity_scores(config, model, checkpoint_path, device)

    try:
        nat_cfg = config.nat
    except AttributeError:
        nat_cfg = config.get("nat", {})

    curriculum_cfg = nat_cfg.get("curriculum", {}) if hasattr(nat_cfg, 'get') else {}

    try:
        total_epochs = config.training.epochs
    except AttributeError:
        total_epochs = config.get("training", {}).get("epochs", 100)

    alpha_scheduler = CurriculumAlphaScheduler(
        alpha_start=curriculum_cfg.get("alpha_start", 0.1),
        alpha_end=curriculum_cfg.get("alpha_end", 0.6),
        total_epochs=total_epochs,
        power=curriculum_cfg.get("power", 1.0),
        warmup_epochs=curriculum_cfg.get("warmup_epochs", 0),
    )

    prob_cfg = nat_cfg.get("probability", {}) if hasattr(nat_cfg, 'get') else {}
    prob_calculator = SensitivityProbabilityCalculator(
        sensitivity_scores=sensitivity_scores,
        p_min=prob_cfg.get("p_min", 0.15),
        p_max=prob_cfg.get("p_max", 1.0),
        gamma=prob_cfg.get("gamma", 1.0),
    )

    controller = NonlinearityController(model)

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

    trainer = SGRNATTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        seed=seed,
        exp_dir=exp_dir,
        sensitivity_scores=sensitivity_scores,
        controller=controller,
        alpha_scheduler=alpha_scheduler,
        prob_calculator=prob_calculator,
    )

    summary = trainer.train()

    return summary

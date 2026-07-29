from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from .losses import classification_loss
from .checkpoint import save_checkpoint, load_checkpoint
from ..utils.logging import setup_logging, get_logger
from ..utils.serialization import save_json, save_metrics_jsonl
from ..utils.timing import Timer, format_duration


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: str,
        seed: int,
        exp_dir: str | Path,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.seed = seed
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 15)
        self.label_smoothing = config.get("label_smoothing", 0.0)
        self.grad_clip = config.get("grad_clip", 0.0)
        self.use_amp = config.get("use_amp", torch.cuda.is_available())
        self.metric_attr = config.get("metric_attr", "val_acc")

        self.logger = setup_logging(self.exp_dir)
        self._metrics_log: list[dict] = []

        self.model = self.model.to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer, self.epochs)
        self.scaler = GradScaler("cuda") if self.use_amp else None

        self.best_metric = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self._start_epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        optim_cfg = self.config.get("optimizer", {})
        lr = optim_cfg.get("lr", 0.1)
        momentum = optim_cfg.get("momentum", 0.9)
        weight_decay = optim_cfg.get("weight_decay", 5e-4)
        nesterov = optim_cfg.get("nesterov", True)
        return torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    def _create_scheduler(self, optimizer, epochs):
        sched_cfg = self.config.get("scheduler", {})
        sched_type = sched_cfg.get("type", "cosine")
        warmup_epochs = sched_cfg.get("warmup_epochs", 5)

        if sched_type == "cosine":
            T_max = max(1, epochs - warmup_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max
            )
        elif sched_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif sched_type == "multistep":
            milestones = sched_cfg.get("milestones", [60, 120, 160])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.2
            )
        elif sched_type == "none":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda _: 1.0
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

        if warmup_epochs > 0 and sched_type != "none":
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_epochs],
            )

        return scheduler

    def train_epoch(self, epoch: int, controller=None) -> dict:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = classification_loss(outputs, targets, self.label_smoothing)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = classification_loss(outputs, targets, self.label_smoothing)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if self.scheduler is not None:
            self.scheduler.step()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        return {"train_loss": train_loss, "train_acc": train_acc}

    @torch.no_grad()
    def validate(self, controller=None) -> dict:
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(inputs)
            loss = classification_loss(outputs, targets, self.label_smoothing)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / total
        val_acc = 100.0 * correct / total

        return {"val_loss": val_loss, "val_acc": val_acc}

    def _get_metric_value(self, metrics: dict) -> float:
        value = metrics.get(self.metric_attr, 0.0)
        return float(value)

    def _is_better(self, current: float, best: float) -> bool:
        return current > best

    def _save_metrics(self, metrics: dict, epoch: int) -> None:
        self._metrics_log.append(metrics)

        metrics_path = self.exp_dir / "metrics.jsonl"
        save_metrics_jsonl(metrics, metrics_path)

    def train(
        self,
        epochs: int | None = None,
        controller=None,
        checkpoint_path: str | Path | None = None,
    ) -> dict:
        if epochs is not None:
            self.epochs = epochs

        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists():
                loaded_epoch, loaded_metrics = load_checkpoint(
                    ckpt_path, self.model, self.optimizer, self.scheduler, self.device
                )
                self._start_epoch = loaded_epoch + 1
                self.best_metric = loaded_metrics.get(self.metric_attr, 0.0)
                self.logger.info(f"Resumed from epoch {loaded_epoch}")
                self.logger.info(f"Best {self.metric_attr}: {self.best_metric:.4f}")

        self.logger.info(f"Starting training for {self.epochs} epochs (from {self._start_epoch})")
        total_train_start = time.perf_counter()

        for epoch in range(self._start_epoch, self.epochs):
            epoch_start = time.perf_counter()

            train_metrics = self.train_epoch(epoch, controller=controller)
            val_metrics = self.validate(controller=controller)

            lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.perf_counter() - epoch_start

            metrics = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "learning_rate": lr,
                "epoch_time": epoch_time,
            }

            if torch.cuda.is_available():
                metrics["gpu_memory"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

            self._save_metrics(metrics, epoch)

            current_metric = self._get_metric_value(metrics)

            is_best = self._is_better(current_metric, self.best_metric)
            last_path = self.exp_dir / "last.pt"
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, metrics, last_path, is_best=is_best,
            )

            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            best_flag = "*" if is_best else " "
            self.logger.info(
                f"Epoch {epoch:3d} {best_flag} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_acc']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_acc']:.2f}% | "
                f"LR: {lr:.6f} | "
                f"Time: {format_duration(epoch_time)}"
            )

            if self.patience > 0 and self.epochs_no_improve >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch} (patience={self.patience})")
                break

        total_time = time.perf_counter() - total_train_start

        best_metrics = {"best_epoch": self.best_epoch, "best_val_acc": self.best_metric}

        summary = {
            **best_metrics,
            "total_epochs": self._start_epoch + len(self._metrics_log),
            "total_time": total_time,
            "total_time_str": format_duration(total_time),
            "config": self.config,
            "seed": self.seed,
        }
        save_json(summary, self.exp_dir / "summary.json")

        self.logger.info(f"Training complete in {format_duration(total_time)}")
        self.logger.info(f"Best {self.metric_attr} at epoch {self.best_epoch}: {self.best_metric:.4f}")

        return summary

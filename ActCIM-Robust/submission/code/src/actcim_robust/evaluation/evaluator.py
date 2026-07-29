from __future__ import annotations

import torch
import torch.nn.functional as F

from .classification_metrics import (
    compute_accuracy,
    compute_topk_accuracy,
    compute_per_class_accuracy,
)
from .calibration import (
    compute_ece,
    compute_mce,
    compute_brier_score,
    compute_nll,
    compute_mean_confidence,
)


class Evaluator:
    def __init__(
        self,
        model,
        data_loader,
        device,
        controller=None,
        criterion=None,
    ):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.controller = controller
        self.criterion = criterion or torch.nn.CrossEntropyLoss()

    def evaluate(self, compute_calibration=True):
        self.model.eval()
        self.model.to(self.device)

        all_outputs = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[:2]
                else:
                    inputs = batch
                    targets = torch.zeros(inputs.size(0), dtype=torch.long)

                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                all_outputs.append(outputs)
                all_targets.append(targets)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        results = {
            "accuracy": compute_accuracy(all_outputs, all_targets),
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
        }

        if compute_calibration:
            results["ece"] = compute_ece(all_outputs, all_targets)
            results["mce"] = compute_mce(all_outputs, all_targets)
            results["brier"] = compute_brier_score(all_outputs, all_targets)
            results["nll"] = compute_nll(all_outputs, all_targets)
            results["mean_confidence"] = compute_mean_confidence(all_outputs)
            results["per_class_accuracy"] = compute_per_class_accuracy(all_outputs, all_targets)

        return results

    def evaluate_with_alpha(self, alpha):
        if self.controller is not None:
            self.controller.set_global_alpha(alpha)
            self.controller.enable_all()
        return self.evaluate()

    def evaluate_alpha_sweep(self, alphas):
        results = []
        for alpha in alphas:
            metrics = self.evaluate_with_alpha(alpha)
            metrics["alpha"] = alpha
            results.append(metrics)
        return results

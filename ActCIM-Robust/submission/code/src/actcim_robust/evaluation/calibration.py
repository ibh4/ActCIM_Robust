from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def compute_ece(outputs, targets, n_bins=15):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach()
        targets = targets.detach()
        probs = F.softmax(outputs, dim=1)
        confidences, predictions = probs.max(dim=1)
        correct = predictions.eq(targets).float()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=outputs.device)
        ece = torch.tensor(0.0, device=outputs.device)
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            if i == n_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)
            bin_size = in_bin.float().sum()
            if bin_size > 0:
                bin_acc = correct[in_bin].mean()
                bin_conf = confidences[in_bin].mean()
                ece += (bin_size / targets.size(0)) * torch.abs(bin_acc - bin_conf)
        return ece.item()

    if isinstance(outputs, np.ndarray):
        probs = _softmax_numpy(outputs)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == targets).astype(np.float32)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            if i == n_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)
            bin_size = in_bin.sum()
            if bin_size > 0:
                bin_acc = correct[in_bin].mean()
                bin_conf = confidences[in_bin].mean()
                ece += (bin_size / len(targets)) * abs(bin_acc - bin_conf)
        return float(ece)
    return 0.0


def compute_mce(outputs, targets, n_bins=15):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach()
        targets = targets.detach()
        probs = F.softmax(outputs, dim=1)
        confidences, predictions = probs.max(dim=1)
        correct = predictions.eq(targets).float()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=outputs.device)
        max_gap = torch.tensor(0.0, device=outputs.device)
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            if i == n_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)
            bin_size = in_bin.float().sum()
            if bin_size > 0:
                bin_acc = correct[in_bin].mean()
                bin_conf = confidences[in_bin].mean()
                gap = (bin_acc - bin_conf).abs()
                if gap > max_gap:
                    max_gap = gap
        return max_gap.item()

    if isinstance(outputs, np.ndarray):
        probs = _softmax_numpy(outputs)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == targets).astype(np.float32)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_gap = 0.0
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            if i == n_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)
            else:
                in_bin = (confidences >= lower) & (confidences < upper)
            bin_size = in_bin.sum()
            if bin_size > 0:
                bin_acc = correct[in_bin].mean()
                bin_conf = confidences[in_bin].mean()
                gap = abs(bin_acc - bin_conf)
                if gap > max_gap:
                    max_gap = gap
        return float(max_gap)
    return 0.0


def compute_brier_score(outputs, targets):
    if isinstance(outputs, torch.Tensor):
        targets = targets.detach()
        probs = F.softmax(outputs, dim=1)
        num_classes = outputs.size(1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).float()
        brier = ((probs - targets_one_hot) ** 2).sum(dim=1).mean()
        return brier.item()

    if isinstance(outputs, np.ndarray):
        probs = _softmax_numpy(outputs)
        num_classes = outputs.shape[1]
        targets_one_hot = np.eye(num_classes)[targets.astype(int)]
        brier = ((probs - targets_one_hot) ** 2).sum(axis=1).mean()
        return float(brier)
    return 0.0


def compute_nll(outputs, targets):
    if isinstance(outputs, torch.Tensor):
        log_probs = F.log_softmax(outputs, dim=1)
        nll = F.nll_loss(log_probs, targets.long(), reduction="mean")
        return nll.item()

    if isinstance(outputs, np.ndarray):
        probs = _softmax_numpy(outputs)
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0 - eps)
        targets_int = targets.astype(np.int64)
        nll = -np.mean(np.log(probs[np.arange(len(targets_int)), targets_int]))
        return float(nll)
    return 0.0


def compute_mean_confidence(outputs):
    if isinstance(outputs, torch.Tensor):
        probs = F.softmax(outputs, dim=1)
        return probs.max(dim=1)[0].mean().item()

    if isinstance(outputs, np.ndarray):
        probs = _softmax_numpy(outputs)
        return float(probs.max(axis=1).mean())
    return 0.0


def _softmax_numpy(x):
    shifted = x - x.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)

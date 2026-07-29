from __future__ import annotations

import torch
import numpy as np


def compute_accuracy(outputs, targets):
    if isinstance(outputs, torch.Tensor):
        predictions = outputs.argmax(dim=1)
        correct = (predictions == targets).float().sum().item()
        return correct / targets.size(0)
    if isinstance(outputs, np.ndarray):
        predictions = np.argmax(outputs, axis=1)
        return np.mean(predictions == targets)
    return 0.0


def compute_topk_accuracy(outputs, targets, k=5):
    if isinstance(outputs, torch.Tensor):
        _, topk_indices = outputs.topk(k, dim=1, largest=True, sorted=True)
        correct = topk_indices.eq(targets.view(-1, 1).expand_as(topk_indices))
        return correct.float().sum().item() / targets.size(0)
    if isinstance(outputs, np.ndarray):
        topk_indices = np.argpartition(-outputs, k - 1, axis=1)[:, :k]
        correct = np.any(topk_indices == targets.reshape(-1, 1), axis=1)
        return np.mean(correct)
    return 0.0


def compute_per_class_accuracy(outputs, targets, num_classes=10):
    if isinstance(outputs, torch.Tensor):
        predictions = outputs.argmax(dim=1)
        per_class = {}
        for c in range(num_classes):
            mask = (targets == c)
            if mask.sum() == 0:
                per_class[c] = 0.0
            else:
                per_class[c] = (predictions[mask] == c).float().mean().item()
        return per_class

    if isinstance(outputs, np.ndarray):
        predictions = np.argmax(outputs, axis=1)
        per_class = {}
        for c in range(num_classes):
            mask = (targets == c)
            if mask.sum() == 0:
                per_class[c] = 0.0
            else:
                per_class[c] = np.mean(predictions[mask] == c)
        return per_class

    return {}


def compute_confusion_matrix(outputs, targets, num_classes=10):
    if isinstance(outputs, torch.Tensor):
        predictions = outputs.argmax(dim=1)
        cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for t, p in zip(targets.view(-1), predictions.view(-1)):
            cm[t.long(), p.long()] += 1
        return cm.numpy()

    if isinstance(outputs, np.ndarray):
        predictions = np.argmax(outputs, axis=1)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(targets.flatten(), predictions.flatten()):
            cm[int(t), int(p)] += 1
        return cm

    return np.array([])

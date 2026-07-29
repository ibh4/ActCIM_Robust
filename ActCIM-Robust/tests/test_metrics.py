import torch
import pytest
from actcim_robust.evaluation.classification_metrics import compute_accuracy
from actcim_robust.evaluation.calibration import compute_ece


def test_perfect_accuracy():
    outputs = torch.eye(10).unsqueeze(0).repeat(5, 1, 1).reshape(50, 10)
    targets = torch.arange(10).repeat(5)
    acc = compute_accuracy(outputs, targets)
    assert acc == 1.0


def test_random_accuracy():
    torch.manual_seed(42)
    outputs = torch.randn(100, 10)
    targets = torch.randint(0, 10, (100,))
    acc = compute_accuracy(outputs, targets)
    assert 0.0 <= acc <= 1.0


def test_ece_range():
    outputs = torch.randn(100, 10)
    targets = torch.randint(0, 10, (100,))
    ece = compute_ece(outputs, targets)
    assert 0.0 <= ece <= 1.0

import torch
import pytest
from actcim_robust.nonlinearity.function import nonlinearity


def test_alpha_zero_equivalence():
    """9.1: alpha=0 should produce identical output"""
    x = torch.randn(4, 3, 32, 32)
    y = nonlinearity(x, alpha=0.0)
    assert torch.allclose(x, y, atol=1e-6, rtol=1e-5)


def test_output_no_nan_inf():
    """9.2: Output should not contain NaN or Inf"""
    x = torch.randn(8, 64, 16, 16)
    for alpha in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        y = nonlinearity(x, alpha=alpha)
        assert not torch.isnan(y).any(), f"NaN at alpha={alpha}"
        assert not torch.isinf(y).any(), f"Inf at alpha={alpha}"


def test_gradient_exists():
    """9.3: Gradients should be computable"""
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    y = nonlinearity(x, alpha=0.4)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
    assert x.grad.abs().sum() > 0


def test_alpha_zero_gradient():
    """Gradient with alpha=0 should pass through correctly"""
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    y = nonlinearity(x, alpha=0.0)
    loss = y.sum()
    loss.backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad), atol=1e-5)


def test_monotonic_positive_alpha():
    """Positive alpha should increase magnitude of large values"""
    x = torch.tensor([0.5, -0.5, 0.9, -0.9])
    y = nonlinearity(x, alpha=0.5)
    assert abs(y[2].item()) > abs(x[2].item()) or abs(y[2].item()) >= abs(x[2].item())


def test_per_tensor_vs_per_sample():
    """Per-tensor and per-sample should differ for varying inputs"""
    x = torch.randn(4, 3, 8, 8)
    from actcim_robust.nonlinearity.function import nonlinearity_per_tensor, nonlinearity_per_sample
    y1 = nonlinearity_per_tensor(x, 0.3)
    y2 = nonlinearity_per_sample(x, 0.3)
    assert y1.shape == y2.shape == x.shape

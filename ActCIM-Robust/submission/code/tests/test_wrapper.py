import torch
import torch.nn as nn
import pytest
from actcim_robust.nonlinearity.wrapper import NonlinearInputWrapper


def test_wrapper_disabled_equivalence():
    """9.4: Disabled wrapper should be equivalent to original"""
    conv = nn.Conv2d(3, 16, 3, padding=1)
    wrapper = NonlinearInputWrapper(conv, "test", alpha=0.0, enabled=False)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        wrapper.load_state_dict(conv.state_dict())
    with torch.no_grad():
        y_orig = conv(x)
        y_wrap = wrapper(x)
    assert torch.allclose(y_orig, y_wrap, atol=1e-5)


def test_wrapper_alpha_zero_equivalence():
    """Wrapper with alpha=0 (enabled) should equal original"""
    conv = nn.Conv2d(3, 16, 3, padding=1)
    wrapper = NonlinearInputWrapper(conv, "test", alpha=0.0, enabled=True)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y_orig = conv(x)
        y_wrap = wrapper(x)
    assert torch.allclose(y_orig, y_wrap, atol=1e-5)


def test_wrapper_nonzero_alpha_changes_output():
    """Non-zero alpha should change the output"""
    conv = nn.Conv2d(3, 16, 3, padding=1)
    wrapper = NonlinearInputWrapper(conv, "test", alpha=0.4, enabled=True)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y_orig = conv(x)
        y_wrap = wrapper(x)
    assert not torch.allclose(y_orig, y_wrap, atol=1e-3)


def test_wrapper_preserves_weight():
    """Wrapper should expose weight and bias attributes"""
    conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
    wrapper = NonlinearInputWrapper(conv, "test")
    assert hasattr(wrapper, 'weight')
    assert wrapper.weight.shape == conv.weight.shape


def test_wrapper_enable_disable():
    """Enable/disable should toggle behavior"""
    conv = nn.Conv2d(3, 16, 3, padding=1)
    wrapper = NonlinearInputWrapper(conv, "test", alpha=0.5, enabled=False)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y_disabled = wrapper(x)
        assert torch.allclose(y_disabled, conv(x), atol=1e-5)
        wrapper.enable()
        y_enabled = wrapper(x)
        assert not torch.allclose(y_enabled, conv(x), atol=1e-3)

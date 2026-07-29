"""Comprehensive nonlinearity implementation audit tests."""
from __future__ import annotations

import copy
import sys
import torch
import torch.nn as nn

from actcim_robust.nonlinearity.function import (
    nonlinearity,
    nonlinearity_per_tensor,
    nonlinearity_per_sample,
    nonlinearity_per_channel,
)
from actcim_robust.nonlinearity.wrapper import NonlinearInputWrapper
from actcim_robust.nonlinearity.controller import NonlinearityController
from actcim_robust.nonlinearity.scheduler import SensitivityProbabilityCalculator


# ---------------------------------------------------------------------------
# Test 2.1: alpha=0 identity
# ---------------------------------------------------------------------------
def test_alpha_zero_identity():
    x = torch.randn(4, 3, 32, 32)
    y = nonlinearity(x, alpha=0.0)
    assert torch.allclose(x, y, atol=1e-6), "alpha=0 should be identity"
    print("  PASS: alpha=0 identity")


# ---------------------------------------------------------------------------
# Test 2.2: Zero input no NaN / Inf
# ---------------------------------------------------------------------------
def test_zero_input_no_nan():
    x = torch.zeros(4, 64, 16, 16)
    for alpha in [0.0, 0.1, 0.4, 0.8, 1.0]:
        y = nonlinearity(x, alpha=alpha)
        assert not torch.isnan(y).any(), f"NaN at alpha={alpha}"
        assert not torch.isinf(y).any(), f"Inf at alpha={alpha}"
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6), f"zero input should map to zero, alpha={alpha}"
    print("  PASS: zero input no NaN/Inf")


# ---------------------------------------------------------------------------
# Test 2.3: Injection location (input, not output/weights)
# ---------------------------------------------------------------------------
def test_injection_location():
    conv = nn.Conv2d(3, 16, 3, padding=1)
    conv_weight_copy = conv.weight.data.clone()
    wrapper = NonlinearInputWrapper(conv, "test", alpha=0.4, enabled=True)

    x = torch.randn(2, 3, 32, 32)

    # Verify wrapper applies nonlinearity BEFORE the conv (by checking weight unchanged)
    with torch.no_grad():
        _ = wrapper(x)

    assert torch.allclose(conv.weight.data, conv_weight_copy, atol=1e-8), \
        "Weights should NOT be modified by nonlinearity"

    # Verify nonlinearity is on INPUT by comparing:
    # wrapper.forward(x) = conv(nonlinearity(x))
    # vs conv(x) -> should differ since alpha=0.4
    wrapper2 = NonlinearInputWrapper(conv, "test2", alpha=0.4, enabled=True)
    conv2 = nn.Conv2d(3, 16, 3, padding=1)
    conv2.load_state_dict(conv.state_dict())

    with torch.no_grad():
        y_wrapped = wrapper2(x)
        y_nonlin = nonlinearity(x, alpha=0.4)
        y_expected = conv2(y_nonlin)
        assert torch.allclose(y_wrapped, y_expected, atol=1e-5), \
            "wrapper(x) should equal conv(nonlinearity(x))"

    print("  PASS: nonlinearity applied to INPUT of Conv2d, not weights/output")


# ---------------------------------------------------------------------------
# Test 2.4: Single-layer injection
# ---------------------------------------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


def test_single_layer_injection():
    model = SimpleNet()
    ctrl = NonlinearityController(model)
    ctrl.disable_all()
    ctrl.enable_layers(['conv1'])
    ctrl.set_layer_alpha('conv1', 0.5)

    x = torch.randn(2, 3, 8, 8)

    # Verify conv1 wrapper is active
    assert ctrl.get_layer_names() == ['conv1', 'conv2']
    wrappers = ctrl.get_wrappers()
    assert wrappers['conv1'].enabled is True
    assert wrappers['conv2'].enabled is False
    assert wrappers['conv1'].alpha == 0.5

    # conv2 should receive input that has gone through conv1(nonlinearity(x))
    # Verify conv2's input differs from clean path
    # Capture conv2 input via forward hook
    conv2_inputs = []

    def hook(module, input, _output):
        conv2_inputs.append(input[0].detach().clone())

    handle = model.conv2.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(x)
    handle.remove()

    # Now run without nonlinearity
    ctrl.disable_all()
    conv2_inputs_clean = []

    def hook2(module, input, _output):
        conv2_inputs_clean.append(input[0].detach().clone())

    handle2 = model.conv2.register_forward_hook(hook2)
    with torch.no_grad():
        _ = model(x)
    handle2.remove()

    # Input to conv2 should differ between clean and nonlinearity-injected paths
    assert not torch.allclose(conv2_inputs[0], conv2_inputs_clean[0], atol=1e-3), \
        "conv2 should receive different input when conv1 has nonlinearity injected"
    print("  PASS: single-layer injection (conv1 only)")


# ---------------------------------------------------------------------------
# Test 2.5: All-layer injection (Conv2d, Linear, ConvTranspose2d)
# ---------------------------------------------------------------------------
class AllLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.convt = nn.ConvTranspose2d(8, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(4, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.relu(self.convt(x))
        x = self.pool(x).flatten(1)
        return self.linear(x)


def test_all_layer_injection():
    model = AllLayerNet()
    ctrl = NonlinearityController(model)

    names = ctrl.get_layer_names()
    assert 'conv' in names, "conv should be wrapped"
    assert 'convt' in names, "convt should be wrapped"
    assert 'linear' in names, "linear should be wrapped"
    assert len(names) == 3, f"expected 3 layers, got {len(names)}: {names}"

    # Verify all can be enabled and disabled
    ctrl.enable_all()
    ctrl.set_global_alpha(0.3)
    for name in names:
        assert ctrl.get_wrappers()[name].enabled is True

    ctrl.disable_all()
    for name in names:
        assert ctrl.get_wrappers()[name].enabled is False

    # Verify forward pass works in all states
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        y_disabled = model(x)
        assert y_disabled.shape == (2, 10)

        ctrl.enable_all()
        ctrl.set_global_alpha(0.3)
        y_enabled = model(x)
        assert y_enabled.shape == (2, 10)
        assert not torch.isnan(y_enabled).any()
        assert not torch.isinf(y_enabled).any()

    print("  PASS: all Conv2d/Linear/ConvTranspose2d layers wrapped & controllable")


# ---------------------------------------------------------------------------
# Test 2.6: BatchNorm freeze during eval
# ---------------------------------------------------------------------------
class BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn(self.conv(x))


def test_batchnorm_freeze_during_eval():
    model = BatchNormModel()
    ctrl = NonlinearityController(model)
    ctrl.enable_all()
    ctrl.set_global_alpha(0.4)

    # Put in eval mode
    model.eval()

    # Record running mean/var before forward passes
    running_mean_before = model.bn.running_mean.clone()
    running_var_before = model.bn.running_var.clone()

    # Multiple forward passes
    for _ in range(10):
        with torch.no_grad():
            x = torch.randn(4, 3, 16, 16)
            _ = model(x)

    running_mean_after = model.bn.running_mean.clone()
    running_var_after = model.bn.running_var.clone()

    assert torch.allclose(running_mean_before, running_mean_after, atol=1e-8), \
        "BatchNorm running_mean should not update in eval mode"
    assert torch.allclose(running_var_before, running_var_after, atol=1e-8), \
        "BatchNorm running_var should not update in eval mode"

    # Verify train mode DOES update
    model.train()
    running_mean_train_before = model.bn.running_mean.clone()
    for _ in range(10):
        x = torch.randn(4, 3, 16, 16)
        _ = model(x)

    running_mean_train_after = model.bn.running_mean.clone()
    assert not torch.allclose(running_mean_train_before, running_mean_train_after, atol=1e-8), \
        "BatchNorm running_mean should update in train mode"

    print("  PASS: BatchNorm running stats frozen in eval, updated in train")


# ---------------------------------------------------------------------------
# Test 2.7: Gradient computation through nonlinearity
# ---------------------------------------------------------------------------
def test_gradient_computation():
    x = torch.randn(2, 3, 16, 16, requires_grad=True)

    # Test various alpha values
    for alpha in [0.1, 0.3, 0.5, 0.7, 1.0]:
        # Reset grad
        if x.grad is not None:
            x.grad.zero_()
        x2 = x.detach().clone().requires_grad_(True)
        y = nonlinearity(x2, alpha=alpha)
        loss = y.sum()
        loss.backward()
        assert x2.grad is not None, f"grad should not be None at alpha={alpha}"
        assert not torch.isnan(x2.grad).any(), f"NaN grad at alpha={alpha}"
        assert not torch.isinf(x2.grad).any(), f"Inf grad at alpha={alpha}"
        assert x2.grad.abs().sum() > 0, f"grad should be non-zero at alpha={alpha}"

    # alpha=0 should give gradient of 1.0
    if x.grad is not None:
        x.grad.zero_()
    x3 = torch.randn(2, 3, 16, 16, requires_grad=True)
    y = nonlinearity(x3, alpha=0.0)
    loss = y.sum()
    loss.backward()
    assert torch.allclose(x3.grad, torch.ones_like(x3.grad), atol=1e-5), \
        "gradient with alpha=0 should be all ones"

    print("  PASS: gradient computation through nonlinearity")


# ---------------------------------------------------------------------------
# Test 2.7(cont): Enable/disable toggling
# ---------------------------------------------------------------------------
def test_enable_disable_toggle():
    conv = nn.Conv2d(3, 8, 3, padding=1)
    wrapper = NonlinearInputWrapper(conv, "test", alpha=0.5, enabled=False)
    x = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        y_disabled = wrapper(x)
        assert torch.allclose(y_disabled, conv(x), atol=1e-5), \
            "disabled should equal original"

        wrapper.enable()
        assert wrapper.enabled is True
        y_enabled = wrapper(x)
        assert not torch.allclose(y_enabled, conv(x), atol=1e-3), \
            "enabled with alpha=0.5 should differ from original"

        wrapper.disable()
        assert wrapper.enabled is False
        y_disabled2 = wrapper(x)
        assert torch.allclose(y_disabled2, conv(x), atol=1e-5), \
            "disabled again should equal original"

    print("  PASS: enable/disable properly toggles behavior")


# ---------------------------------------------------------------------------
# Test 2.7(cont): State dict save/load roundtrip
# ---------------------------------------------------------------------------
def test_state_dict_roundtrip():
    # Test wrapper state_dict roundtrip
    conv1 = nn.Conv2d(3, 16, 3, padding=1)
    wrapper1 = NonlinearInputWrapper(conv1, "test", alpha=0.5, enabled=True)

    # Save
    sd = wrapper1.state_dict()
    assert 'weight' in sd
    assert 'bias' in sd

    # Load into a new conv through wrapper
    conv2 = nn.Conv2d(3, 16, 3, padding=1)
    wrapper2 = NonlinearInputWrapper(conv2, "test2", alpha=0.1, enabled=False)
    wrapper2.load_state_dict(sd)

    # Verify weights match
    assert torch.allclose(wrapper1.weight, wrapper2.weight, atol=1e-8), \
        "weights should match after load_state_dict"

    # Verify bias matches
    assert torch.allclose(wrapper1.bias, wrapper2.bias, atol=1e-8), \
        "bias should match after load_state_dict"

    # Verify outputs match for same input
    x = torch.randn(2, 3, 32, 32)
    wrapper1.disable()
    wrapper2.disable()
    with torch.no_grad():
        y1 = wrapper1(x)
        y2 = wrapper2(x)
    assert torch.allclose(y1, y2, atol=1e-5), \
        "outputs should match after state_dict roundtrip"

    print("  PASS: state_dict save/load roundtrip")


# ---------------------------------------------------------------------------
# Test controller export/restore roundtrip
# ---------------------------------------------------------------------------
def test_controller_export_restore():
    model = SimpleNet()
    ctrl = NonlinearityController(model)
    ctrl.enable_layers(['conv1'])
    ctrl.set_layer_alpha('conv1', 0.4)

    state = ctrl.export_state()
    assert state['injection_mode'] == 'single_layer'
    assert state['wrappers']['conv1']['alpha'] == 0.4
    assert state['wrappers']['conv1']['enabled'] is True
    assert state['wrappers']['conv2']['enabled'] is False

    # Modify
    ctrl.disable_all()
    ctrl.set_global_alpha(0.0)

    # Restore
    ctrl.restore_state(state)
    assert ctrl._injection_mode == 'single_layer'
    assert ctrl.get_wrappers()['conv1'].alpha == 0.4
    assert ctrl.get_wrappers()['conv1'].enabled is True
    assert ctrl.get_wrappers()['conv2'].enabled is False

    print("  PASS: controller export/restore state roundtrip")


# ---------------------------------------------------------------------------
# Test 3: SensitivityProbabilityCalculator
# ---------------------------------------------------------------------------
def test_sensitivity_probability_calculator():
    scores = {'layer_a': 0.1, 'layer_b': 0.5, 'layer_c': 0.9, 'layer_d': 0.3}
    calc = SensitivityProbabilityCalculator(scores, p_min=0.15, p_max=1.0, gamma=1.0)

    probs = calc.get_probabilities()
    assert len(probs) == 4

    # All probabilities should be in [p_min, p_max]
    for name, p in probs.items():
        assert 0.15 <= p <= 1.0, f"{name}: {p} not in [0.15, 1.0]"

    # layer_c (highest score) should have max probability
    assert probs['layer_c'] == 1.0, \
        f"highest score layer should have p_max=1.0, got {probs['layer_c']}"

    # layer_a (lowest score) should have p_min
    assert probs['layer_a'] == 0.15, \
        f"lowest score layer should have p_min=0.15, got {probs['layer_a']}"

    # Test with gamma != 1
    calc2 = SensitivityProbabilityCalculator(scores, p_min=0.1, p_max=0.9, gamma=2.0)
    probs2 = calc2.get_probabilities()
    for name, p in probs2.items():
        assert 0.1 <= p <= 0.9, f"{name}: {p} not in [0.1, 0.9] with gamma=2.0"

    # Test uniform scores: all normalized=1.0 -> p = p_min + (p_max-p_min)*1 = p_max
    calc3 = SensitivityProbabilityCalculator({'a': 0.5, 'b': 0.5}, p_min=0.2, p_max=0.8)
    probs3 = calc3.get_probabilities()
    for p in probs3.values():
        assert abs(p - 0.8) < 0.01, f"uniform scores should map to p_max=0.8, got {p}"

    # Test empty input
    calc4 = SensitivityProbabilityCalculator({}, p_min=0.0, p_max=1.0)
    assert calc4.get_probabilities() == {}

    # Test get_top_layers
    top = calc.get_top_layers(fraction=0.5)
    assert len(top) == 2
    assert 'layer_c' in top  # highest score
    assert 'layer_a' not in top  # lowest score

    print("  PASS: SensitivityProbabilityCalculator normalization and bounds")


# ---------------------------------------------------------------------------
# Additional: Verify nonlinearity_per_tensor vs per_sample vs per_channel shapes
# ---------------------------------------------------------------------------
def test_nonlinearity_variants_shapes():
    x = torch.randn(4, 3, 32, 32)

    y_tensor = nonlinearity_per_tensor(x, alpha=0.3)
    y_sample = nonlinearity_per_sample(x, alpha=0.3)
    y_channel = nonlinearity_per_channel(x, alpha=0.3)

    assert y_tensor.shape == x.shape
    assert y_sample.shape == x.shape
    assert y_channel.shape == x.shape

    # Per-tensor and per-sample should produce DIFFERENT results
    # (different normalization per sample vs per whole tensor)
    assert not torch.allclose(y_tensor, y_sample, atol=1e-3), \
        "per_tensor and per_sample should produce different results"

    print("  PASS: nonlinearity variant shapes")


# ---------------------------------------------------------------------------
# Additional: Verify scaling invariance (cubic function should scale)
# ---------------------------------------------------------------------------
def test_scaling_invariance():
    x = torch.randn(4, 64, 16, 16)
    alpha = 0.3
    y1 = nonlinearity(x, alpha=alpha)

    # Scaling input by constant should scale output by same constant
    scale = 2.5
    y2 = nonlinearity(x * scale, alpha=alpha)
    assert torch.allclose(y2, y1 * scale, atol=1e-4), \
        "nonlinearity should be scale-invariant (homogeneous of degree 1)"

    print("  PASS: scaling invariance")


# ---------------------------------------------------------------------------
# Additional: Clamp_min(eps) prevents division by zero
# ---------------------------------------------------------------------------
def test_eps_prevents_div_by_zero():
    x = torch.zeros(4, 64, 16, 16)
    y = nonlinearity(x, alpha=0.5, eps=1e-8)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    assert y.abs().sum() == 0.0, "zeros should map to zeros"

    print("  PASS: eps clamp prevents division by zero")


# ---------------------------------------------------------------------------
# Additional: Verify wrapper with Linear layer works
# ---------------------------------------------------------------------------
def test_wrapper_linear():
    linear = nn.Linear(10, 5)
    wrapper = NonlinearInputWrapper(linear, "fc", alpha=0.4, enabled=True)
    x = torch.randn(3, 10)

    with torch.no_grad():
        y_wrap = wrapper(x)
        y_expected = linear(nonlinearity(x, alpha=0.4))
        assert torch.allclose(y_wrap, y_expected, atol=1e-5)

    print("  PASS: wrapper with Linear layer")


# ---------------------------------------------------------------------------
# Additional: Verify wrapper with ConvTranspose2d
# ---------------------------------------------------------------------------
def test_wrapper_convt():
    convt = nn.ConvTranspose2d(4, 8, 3, padding=1)
    wrapper = NonlinearInputWrapper(convt, "convt", alpha=0.4, enabled=True)
    x = torch.randn(2, 4, 8, 8)

    with torch.no_grad():
        y_wrap = wrapper(x)
        assert y_wrap.shape[1] == 8
        assert not torch.isnan(y_wrap).any()

    print("  PASS: wrapper with ConvTranspose2d")


# ---------------------------------------------------------------------------
# Additional: Wrong module type should raise TypeError
# ---------------------------------------------------------------------------
def test_wrapper_type_check():
    try:
        NonlinearInputWrapper(nn.ReLU(), "test")
        assert False, "should have raised TypeError"
    except TypeError:
        pass

    try:
        NonlinearInputWrapper(nn.BatchNorm2d(8), "test")
        assert False, "should have raised TypeError"
    except TypeError:
        pass

    print("  PASS: wrapper type checking rejects non-injectable modules")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ActCIM-Robust Nonlinearity Implementation Audit")
    print("=" * 60)

    all_passed = True
    tests = [
        ("Test 2.1: alpha=0 identity", test_alpha_zero_identity),
        ("Test 2.2: zero input no NaN", test_zero_input_no_nan),
        ("Test 2.3: injection location", test_injection_location),
        ("Test 2.4: single-layer injection", test_single_layer_injection),
        ("Test 2.5: all-layer injection", test_all_layer_injection),
        ("Test 2.6: BatchNorm freeze during eval", test_batchnorm_freeze_during_eval),
        ("Test 2.7a: gradient computation", test_gradient_computation),
        ("Test 2.7b: enable/disable toggling", test_enable_disable_toggle),
        ("Test 2.7c: state_dict save/load roundtrip", test_state_dict_roundtrip),
        ("Test 2.7d: controller export/restore", test_controller_export_restore),
        ("Test 3: sensitivity probability", test_sensitivity_probability_calculator),
        ("Extra: nonlinearity variant shapes", test_nonlinearity_variants_shapes),
        ("Extra: scaling invariance", test_scaling_invariance),
        ("Extra: eps prevents div by zero", test_eps_prevents_div_by_zero),
        ("Extra: wrapper with Linear", test_wrapper_linear),
        ("Extra: wrapper with ConvTranspose2d", test_wrapper_convt),
        ("Extra: wrapper type checking", test_wrapper_type_check),
    ]

    for name, test_fn in tests:
        try:
            print(f"\n[{name}]")
            test_fn()
        except Exception as e:
            print(f"  FAIL: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL AUDIT TESTS PASSED")
    else:
        print("SOME AUDIT TESTS FAILED")
    print("=" * 60)

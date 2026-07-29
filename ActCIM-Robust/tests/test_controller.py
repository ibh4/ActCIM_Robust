import torch
import torch.nn as nn
import pytest
from actcim_robust.nonlinearity.controller import NonlinearityController


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


def test_controller_wraps_all_layers():
    model = SimpleNet()
    ctrl = NonlinearityController(model)
    names = ctrl.get_layer_names()
    assert 'conv1' in names
    assert 'conv2' in names


def test_controller_enable_disable_all():
    model = SimpleNet()
    ctrl = NonlinearityController(model)
    ctrl.enable_all()
    ctrl.set_global_alpha(0.0)
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        y_ctrl = model(x)
    ctrl.disable_all()
    with torch.no_grad():
        y_orig = model(x)
    assert torch.allclose(y_ctrl, y_orig, atol=1e-5)


def test_controller_single_layer():
    """9.5: Single layer injection"""
    model = SimpleNet()
    ctrl = NonlinearityController(model)
    ctrl.disable_all()
    ctrl.enable_layers(['conv1'])
    ctrl.set_layer_alpha('conv1', 0.4)
    state = ctrl.export_state()
    assert state['wrappers']['conv1']['enabled'] is True
    assert state['wrappers']['conv2']['enabled'] is False

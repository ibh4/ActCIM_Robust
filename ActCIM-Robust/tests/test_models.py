import torch
import torch.nn as nn
import pytest
from actcim_robust.models import create_model, TinyCNN


def test_tinycnn_forward():
    """9.6: TinyCNN forward/backward works"""
    model = TinyCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    assert y.shape == (4, 10)
    loss = y.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_resnet18_cifar_forward():
    model = create_model("resnet18_cifar", num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_resnet18_cifar_first_layer():
    model = create_model("resnet18_cifar")
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)
    assert model.fc.out_features == 10


def test_model_training_step():
    """Full training step: forward, loss, backward, optimizer"""
    model = TinyCNN()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 10, (4,))
    y = model(x)
    loss = nn.CrossEntropyLoss()(y, t)
    loss.backward()
    opt.step()
    opt.zero_grad()

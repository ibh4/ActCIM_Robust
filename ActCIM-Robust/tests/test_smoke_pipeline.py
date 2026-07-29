import torch
import pytest
from actcim_robust.models import TinyCNN
from actcim_robust.data.cifar10 import get_cifar10_loaders
from actcim_robust.nonlinearity.controller import NonlinearityController


@pytest.mark.smoke
def test_smoke_pipeline():
    """9.8: Full smoke test pipeline"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN().to(device)
    ctrl = NonlinearityController(model)

    x = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 10)

    ctrl.enable_all()
    ctrl.set_global_alpha(0.4)
    with torch.no_grad():
        y2 = model(x)
    assert y2.shape == (2, 10)

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(8, 3, 32, 32).to(device)
    t = torch.randint(0, 10, (8,)).to(device)
    ctrl.set_global_alpha(0.3)
    y = model(x)
    loss = torch.nn.functional.cross_entropy(y, t)
    loss.backward()
    opt.step()

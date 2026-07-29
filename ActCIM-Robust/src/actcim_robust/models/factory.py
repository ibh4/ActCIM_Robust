from .resnet_cifar import resnet18_cifar, resnet20_cifar
from .tinycnn import TinyCNN


_MODEL_REGISTRY = {
    "resnet18_cifar": resnet18_cifar,
    "resnet20_cifar": resnet20_cifar,
    "tinycnn": TinyCNN,
}


def create_model(model_name, num_classes=10, **kwargs):
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_name](num_classes=num_classes, **kwargs)

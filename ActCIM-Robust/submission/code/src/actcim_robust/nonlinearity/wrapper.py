from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .function import nonlinearity_per_tensor, nonlinearity_per_sample, nonlinearity_per_channel
from .registry import NonlinearityRegistry

_SCOPE_MAP = {
    "per_tensor": nonlinearity_per_tensor,
    "per_sample": nonlinearity_per_sample,
    "per_channel": nonlinearity_per_channel,
}


class NonlinearInputWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        layer_name: str,
        alpha: float = 0.0,
        enabled: bool = False,
        normalization_scope: str = "per_tensor",
    ) -> None:
        super().__init__()
        if not isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            raise TypeError(
                f"Expected Conv2d, Linear, or ConvTranspose2d, got {type(module).__name__}"
            )

        self.module = module
        self.layer_name = layer_name
        self._alpha = alpha
        self._enabled = enabled
        self._normalization_scope = normalization_scope
        self._nonlinearity_fn = _SCOPE_MAP.get(normalization_scope, nonlinearity_per_tensor)

        NonlinearityRegistry.register(layer_name, self)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def normalization_scope(self) -> str:
        return self._normalization_scope

    @normalization_scope.setter
    def normalization_scope(self, value: str) -> None:
        if value not in _SCOPE_MAP:
            raise ValueError(f"Unknown normalization_scope: {value}. Choose from {list(_SCOPE_MAP.keys())}")
        self._normalization_scope = value
        self._nonlinearity_fn = _SCOPE_MAP[value]

    @property
    def weight(self) -> torch.Tensor | None:
        return self.module.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.module.bias

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._enabled and self._alpha != 0.0:
            x = self._nonlinearity_fn(x, alpha=self._alpha)
        return self.module(x)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict=strict)

    def __repr__(self) -> str:
        return (
            f"NonlinearInputWrapper("
            f"module={self.module.__class__.__name__}, "
            f"layer={self.layer_name}, "
            f"alpha={self._alpha}, "
            f"enabled={self._enabled}, "
            f"scope={self._normalization_scope})"
        )

from __future__ import annotations

from typing import Callable

import random
import torch.nn as nn

from .wrapper import NonlinearInputWrapper

_INJECTABLE_TYPES = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)


class NonlinearityController:
    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._wrappers: dict[str, NonlinearInputWrapper] = {}
        self._layer_names: list[str] = []
        self._injection_mode = "disabled"
        self._alpha_mode = "fixed"
        self._wrap_all_layers(model)

    def _wrap_all_layers(self, model: nn.Module) -> None:
        for name, module in list(model.named_children()):
            if isinstance(module, _INJECTABLE_TYPES):
                wrapper = NonlinearInputWrapper(module, layer_name=name)
                setattr(model, name, wrapper)
                self._wrappers[name] = wrapper
                self._layer_names.append(name)
            elif len(list(module.children())) > 0:
                self._wrap_all_layers(module)

    def enable_all(self) -> None:
        for wrapper in self._wrappers.values():
            wrapper.enable()
        self._injection_mode = "all_layers"

    def disable_all(self) -> None:
        for wrapper in self._wrappers.values():
            wrapper.disable()
        self._injection_mode = "disabled"

    def enable_layers(self, layer_names: list[str]) -> None:
        for name in layer_names:
            if name in self._wrappers:
                self._wrappers[name].enable()
        if len(layer_names) == 1:
            self._injection_mode = "single_layer"
        else:
            self._injection_mode = "selected_layers"

    def set_global_alpha(self, alpha: float) -> None:
        for wrapper in self._wrappers.values():
            wrapper.alpha = alpha
        self._alpha_mode = "fixed"

    def set_layer_alpha(self, layer_name: str, alpha: float) -> None:
        if layer_name in self._wrappers:
            self._wrappers[layer_name].alpha = alpha

    def set_layer_probability(self, layer_name: str, prob: float) -> None:
        if layer_name in self._wrappers:
            self._wrappers[layer_name].probability = prob

    def sample_batch_configuration(
        self,
        probabilities: dict[str, float] | None = None,
        alpha_range: tuple[float, float] | None = None,
    ) -> dict[str, float]:
        config: dict[str, float] = {}
        for name, wrapper in self._wrappers.items():
            prob = (probabilities or {}).get(name, 1.0)
            if random.random() < prob:
                if alpha_range is not None:
                    alpha = random.uniform(*alpha_range)
                else:
                    alpha = getattr(wrapper, "probability", wrapper.alpha)
                config[name] = alpha
        return config

    def get_wrappers(self) -> dict[str, NonlinearInputWrapper]:
        return dict(self._wrappers)

    def get_layer_names(self) -> list[str]:
        return list(self._layer_names)

    def export_state(self) -> dict:
        state: dict = {}
        for name, wrapper in self._wrappers.items():
            state[name] = {
                "alpha": wrapper.alpha,
                "enabled": wrapper.enabled,
                "normalization_scope": wrapper.normalization_scope,
            }
        return {
            "wrappers": state,
            "injection_mode": self._injection_mode,
            "alpha_mode": self._alpha_mode,
        }

    def restore_state(self, state: dict) -> None:
        wrapper_state = state.get("wrappers", {})
        for name, wrapper in self._wrappers.items():
            if name in wrapper_state:
                s = wrapper_state[name]
                wrapper.alpha = s.get("alpha", wrapper.alpha)
                if s.get("enabled", False):
                    wrapper.enable()
                else:
                    wrapper.disable()
                wrapper.normalization_scope = s.get("normalization_scope", wrapper.normalization_scope)
        self._injection_mode = state.get("injection_mode", self._injection_mode)
        self._alpha_mode = state.get("alpha_mode", self._alpha_mode)

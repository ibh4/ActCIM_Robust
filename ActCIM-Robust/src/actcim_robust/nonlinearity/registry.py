from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wrapper import NonlinearInputWrapper


class NonlinearityRegistry:
    _wrappers: dict[str, "NonlinearInputWrapper"] = {}

    @classmethod
    def register(cls, name: str, wrapper: "NonlinearInputWrapper") -> None:
        cls._wrappers[name] = wrapper

    @classmethod
    def get(cls, name: str) -> "NonlinearInputWrapper | None":
        return cls._wrappers.get(name)

    @classmethod
    def clear(cls) -> None:
        cls._wrappers.clear()

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._wrappers.keys())

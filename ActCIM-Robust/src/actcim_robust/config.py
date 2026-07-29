from __future__ import annotations

from pathlib import Path


class Config:
    def __init__(self, config_path=None, data=None):
        if config_path is not None:
            data = _load_yaml_file(config_path)
        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise TypeError(f"Config data must be a dict, got {type(data).__name__}")

        self._data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = Config(data=value)
            elif isinstance(value, list):
                self._data[key] = [
                    Config(data=v) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                self._data[key] = value

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"Config has no attribute '{key}'")

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def to_dict(self):
        result = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, Config) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def __repr__(self):
        return f"Config({self._data})"

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


def load_config(config_path):
    return Config(config_path=config_path)


def _load_yaml_file(config_path):
    from .utils.serialization import load_yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_yaml(path)

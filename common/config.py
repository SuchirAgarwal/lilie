"""
YAML configs -> typed dataclasses
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any

import yaml


def _coerce(value: str, target_type: type) -> Any:
    """Parse one command-line override to the field's type."""
    if target_type is bool:
        if value.lower() in ("1", "true", "yes"):
            return True
        if value.lower() in ("0", "false", "no"):
            return False
        raise ValueError(f"expected a boolean, got {value!r}")
    return target_type(value)


def load_config(config_cls, path: str, overrides: list[str] | None = None):
    """Instantiate ``config_cls`` from a YAML file plus ``key=value`` overrides."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    field_types = {f.name: f.type for f in dataclasses.fields(config_cls)}
    unknown = set(raw) - set(field_types)
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    for item in overrides or []:
        key, _, value = item.partition("=")
        if key not in field_types:
            raise ValueError(f"unknown config key {key!r}")
        current = raw.get(key, getattr(config_cls(), key))
        raw[key] = _coerce(value, type(current) if current is not None else str)
    cfg = config_cls(**raw)
    if hasattr(cfg, "validate"):
        cfg.validate()
    return cfg


def parse_cli(description: str) -> tuple[str, list[str]]:
    """``--config path [key=value ...]``; returns the path and the override list."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="key=value overrides of config fields")
    args = parser.parse_args()
    return args.config, args.overrides


def pretty(cfg) -> str:
    return json.dumps(dataclasses.asdict(cfg), indent=2, sort_keys=True)

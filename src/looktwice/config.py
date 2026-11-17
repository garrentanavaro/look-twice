from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _coerce(v: Any) -> Any:
    if isinstance(v, str) and _FLOAT_RE.match(v.strip()):
        return float(v)
    if isinstance(v, dict):
        return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    return v


def deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (other or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = _coerce(value)


def parse_override(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"override must look like key.sub=value, got: {s}")
    k, v = s.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def load_config(paths: list[str | Path], overrides: list[str] | dict | None = None) -> dict:
    cfg: dict = {}
    for p in paths:
        cfg = deep_merge(cfg, yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {})
    items = overrides.items() if isinstance(overrides, dict) else map(parse_override, overrides or [])
    for k, v in items:
        set_dotted(cfg, k, v)
    return _coerce(cfg)

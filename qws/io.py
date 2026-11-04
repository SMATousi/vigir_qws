"""Result serialization (doc 04 §7): JSON with numpy support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def save_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, default=_default, indent=2))
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())

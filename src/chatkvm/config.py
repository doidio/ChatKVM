"""Load deployment-wide defaults from TOML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomlkit


def read_config(path: Path) -> dict[str, Any]:
    return tomlkit.parse(path.read_text(encoding='utf-8')).unwrap()

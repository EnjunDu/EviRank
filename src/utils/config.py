"""Safe configuration loading with environment-variable expansion."""

from __future__ import annotations

import os
import re
from typing import Any, Dict

import yaml


_UNRESOLVED_ENV = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML without ever requiring credentials to be stored in YAML."""
    with open(config_path, "r", encoding="utf-8") as file:
        raw = file.read()
    expanded = os.path.expandvars(raw)
    config = yaml.safe_load(expanded) or {}
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping.")
    return config


def reject_unresolved_secret_placeholder(value: Any, field_name: str) -> str:
    """Reject an unresolved ``${VAR}`` before it reaches an HTTP request."""
    text = str(value or "").strip()
    if _UNRESOLVED_ENV.search(text):
        raise ValueError(
            f"{field_name} references an environment variable that is not set."
        )
    return text

"""Privacy helpers for external-model boundaries and persisted artifacts."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any, Dict, Iterable, Optional


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "token",
    "secret",
    "password",
    "authorization",
    "email",
    "user_name",
    "username",
)
_PATH_KEYS = {
    "path",
    "image_path",
    "reference_path",
    "target_path",
    "file",
    "file_name",
    "filename",
    "directory",
    "data_root",
}
_LABEL_KEYS = {
    "class_id",
    "super_class_id",
    "label",
    "target_id",
    "gt_paths",
    "ground_truth",
}
_PATH_PATTERN = re.compile(
    r"(?<!\w)(?:[A-Za-z]:[\\/]|/)(?:[^ \t\r\n\"'<>]+[\\/])*[^ \t\r\n\"'<>]*"
)
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+\b"
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*"
    r"[\"']?[^\s,\"']{6,}[\"']?"
)


def make_opaque_id(
    value: Any,
    namespace: str = "item",
    salt: Optional[str] = None,
) -> str:
    """Create a stable, non-reversible identifier without exposing ``value``.

    Set ``EVIRANK_ID_SALT`` to a private random value when artifacts must be
    unlinkable across deployments. The salt itself is never persisted.
    """
    raw = str(value or "")
    key = (salt if salt is not None else os.getenv("EVIRANK_ID_SALT", "")).encode()
    payload = f"{namespace}\0{raw}".encode("utf-8", errors="replace")
    digest = (
        hmac.new(key, payload, hashlib.sha256).hexdigest()
        if key
        else hashlib.sha256(payload).hexdigest()
    )
    safe_namespace = re.sub(r"[^A-Za-z0-9_-]", "_", namespace) or "item"
    return f"{safe_namespace}_{digest[:16]}"


def sanitize_metadata(
    metadata: Optional[Dict[str, Any]],
    allowed_keys: Iterable[str] = (),
) -> Dict[str, Any]:
    """Return an allowlisted metadata copy with sensitive fields removed."""
    if not metadata:
        return {}
    allowed = set(allowed_keys)
    sanitized: Dict[str, Any] = {}
    for key in allowed:
        if key not in metadata or is_sensitive_key(key):
            continue
        value = metadata[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = redact_paths(str(value)) if isinstance(value, str) else value
    return sanitized


def is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return (
        normalized in _PATH_KEYS
        or normalized in _LABEL_KEYS
        or any(part in normalized for part in _SENSITIVE_KEY_PARTS)
    )


def redact_paths(text: str, replacement: str = "<redacted-path>") -> str:
    """Remove Unix/Windows path-like substrings from text sent or logged."""
    return _PATH_PATTERN.sub(replacement, str(text))


def redact_external_text(text: Any) -> str:
    """Redact common PII, credential, and local-path patterns."""
    sanitized = redact_paths(str(text or ""))
    sanitized = _EMAIL_PATTERN.sub("<redacted-email>", sanitized)
    sanitized = _BEARER_PATTERN.sub("Bearer <redacted-token>", sanitized)
    sanitized = _CREDENTIAL_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}=<redacted-secret>",
        sanitized,
    )
    return sanitized


def safe_error_name(error: BaseException) -> str:
    """Return only the exception type, never a path-bearing message."""
    return type(error).__name__

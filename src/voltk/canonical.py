"""Deterministic serialisation.

Replay guarantees rest on this: two runs over the same snapshot must produce
byte-identical output, so ordering and float formatting have to be fixed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any


def _encode(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.astimezone(tz=obj.tzinfo).isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Decimal):
        return format(obj, "f")
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, bytes):
        return obj.hex()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"No canonical form for {type(obj).__name__}")


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_encode,
    ).encode("utf-8")


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

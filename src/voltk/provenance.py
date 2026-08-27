"""Where a payload came from and when."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from voltk.canonical import sha256_bytes


@dataclass(frozen=True, slots=True)
class Provenance:
    source: str
    endpoint: str
    retrieved_at: datetime
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RawResponse:
    """An untouched response body plus its provenance.

    The body is kept verbatim so a snapshot can be re-parsed years later by a
    parser that did not exist when it was captured.
    """

    provenance: Provenance
    body: bytes

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.body)

    def result(self) -> Any:
        return json.loads(self.body)["result"]

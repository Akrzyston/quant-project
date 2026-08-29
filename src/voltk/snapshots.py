"""Snapshot storage and replay.

Raw response bodies are stored verbatim and hashed. Replay re-parses those bytes
through the same code the live path uses, so a reloaded snapshot reproduces the
downstream dossier exactly. Nothing derived is persisted.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import voltk
from voltk.canonical import canonical_json, sha256_bytes
from voltk.capture import SCHEMA_VERSION, Capture, Quality, build_forward_curve
from voltk.instruments import Kind
from voltk.marketdata import parse
from voltk.marketdata.base import MarketDataError
from voltk.provenance import Provenance, RawResponse
from voltk.universe import Universe, UniverseSpec

DEFAULT_DB = Path("data/snapshots.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    library_version TEXT NOT NULL,
    source_label    TEXT NOT NULL,
    spec_json       TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    completed_at    TEXT NOT NULL,
    as_of           TEXT NOT NULL,
    window_ms       REAL NOT NULL,
    drift_json      TEXT NOT NULL,
    quality_json    TEXT NOT NULL,
    degraded        INTEGER NOT NULL,
    note            TEXT NOT NULL DEFAULT '',
    content_hash    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payloads (
    snapshot_id   TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    component     TEXT NOT NULL,
    endpoint      TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    source        TEXT NOT NULL,
    retrieved_at  TEXT NOT NULL,
    body          BLOB NOT NULL,
    body_sha256   TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, component)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_as_of ON snapshots(as_of DESC);
"""


class SnapshotError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SnapshotMeta:
    snapshot_id: str
    schema_version: int
    library_version: str
    source_label: str
    spec: UniverseSpec
    started_at: datetime
    completed_at: datetime
    as_of: datetime
    quality: Quality
    note: str
    content_hash: str

    @property
    def degraded(self) -> bool:
        return not self.quality.ok

    def display_name(self) -> str:
        flag = "" if self.quality.ok else "  [degraded]"
        currencies = "/".join(self.spec.currencies)
        suffix = f"  {self.note}" if self.note else ""
        return f"{self.as_of:%Y-%m-%d %H:%M:%S}Z  {currencies}{flag}{suffix}"


@dataclass(frozen=True, slots=True)
class Snapshot:
    meta: SnapshotMeta
    responses: dict[str, RawResponse]


class SnapshotStore:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            _migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, capture: Capture) -> str:
        snapshot_id = _new_id(capture)
        content_hash = sha256_bytes(
            b"".join(sorted(response.body for _, response in capture.responses))
        )
        quality_blob = canonical_json(
            {
                "window_ms": capture.quality.window_ms,
                "max_window_ms": capture.quality.max_window_ms,
                "max_index_drift_bps": capture.quality.max_index_drift_bps,
                "reasons": list(capture.quality.reasons),
            }
        ).decode()

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    snapshot_id,
                    capture.schema_version,
                    voltk.__version__,
                    capture.source_label,
                    canonical_json(
                        {
                            "currencies": list(capture.spec.currencies),
                            "kinds": sorted(str(k) for k in capture.spec.kinds),
                            "include_inactive": capture.spec.include_inactive,
                        }
                    ).decode(),
                    capture.started_at.isoformat(),
                    capture.completed_at.isoformat(),
                    capture.as_of.isoformat(),
                    capture.quality.window_ms,
                    canonical_json(capture.quality.index_drift_bps).decode(),
                    quality_blob,
                    int(not capture.quality.ok),
                    capture.note,
                    content_hash,
                ),
            )
            conn.executemany(
                "INSERT INTO payloads VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        snapshot_id,
                        component,
                        response.provenance.endpoint,
                        canonical_json(response.provenance.params).decode(),
                        response.provenance.source,
                        response.provenance.retrieved_at.isoformat(),
                        response.body,
                        response.sha256,
                    )
                    for component, response in capture.responses
                ],
            )
        return snapshot_id

    def list(self, limit: int = 100) -> list[SnapshotMeta]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM snapshots ORDER BY as_of DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_meta_from_row(row) for row in rows]

    def load(self, snapshot_id: str) -> Snapshot:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
            if row is None:
                raise SnapshotError(f"No snapshot {snapshot_id!r} in {self.path}.")
            if row["library_version"] != voltk.__version__:
                raise SnapshotError(
                    f"Snapshot {snapshot_id} was captured with voltk "
                    f"{row['library_version']!r}, but this is voltk {voltk.__version__!r}. "
                    "Reload with the matching library version to replay it."
                )
            payloads = conn.execute(
                "SELECT * FROM payloads WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchall()

        responses: dict[str, RawResponse] = {}
        for payload in payloads:
            body = bytes(payload["body"])
            stored_hash = payload["body_sha256"]
            if sha256_bytes(body) != stored_hash:
                raise SnapshotError(
                    f"Payload {payload['component']} in {snapshot_id} failed its hash check. "
                    "The store is corrupt; do not trust results derived from it."
                )
            responses[payload["component"]] = RawResponse(
                provenance=Provenance(
                    source=payload["source"],
                    endpoint=payload["endpoint"],
                    retrieved_at=datetime.fromisoformat(payload["retrieved_at"]),
                    params=json.loads(payload["params_json"]),
                ),
                body=body,
            )
        return Snapshot(meta=_meta_from_row(row), responses=responses)

    def delete(self, snapshot_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM payloads WHERE snapshot_id = ?", (snapshot_id,))
            conn.execute("DELETE FROM snapshots WHERE snapshot_id = ?", (snapshot_id,))


def _migrate(conn: sqlite3.Connection) -> None:
    """Backfill columns added to the schema after a store already existed on disk.

    CREATE TABLE IF NOT EXISTS leaves an older table's columns untouched, so a
    store opened before library_version was added needs it added explicitly. The
    default marks those rows as being from before version tracking existed,
    which load() treats as a mismatch like any other.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(snapshots)")}
    if "library_version" not in columns:
        conn.execute(
            "ALTER TABLE snapshots ADD COLUMN library_version TEXT NOT NULL DEFAULT 'unknown'"
        )


def _new_id(capture: Capture) -> str:
    stamp = capture.as_of.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{capture.spec.key().replace(':', '_')}-{uuid.uuid4().hex[:6]}"


def _meta_from_row(row: sqlite3.Row) -> SnapshotMeta:
    spec_blob = json.loads(row["spec_json"])
    quality_blob = json.loads(row["quality_json"])
    return SnapshotMeta(
        snapshot_id=row["snapshot_id"],
        schema_version=row["schema_version"],
        library_version=row["library_version"],
        source_label=row["source_label"],
        spec=UniverseSpec(
            currencies=tuple(spec_blob["currencies"]),
            kinds=tuple(Kind(k) for k in spec_blob["kinds"]),
            include_inactive=bool(spec_blob.get("include_inactive", False)),
        ),
        started_at=datetime.fromisoformat(row["started_at"]),
        completed_at=datetime.fromisoformat(row["completed_at"]),
        as_of=datetime.fromisoformat(row["as_of"]),
        quality=Quality(
            window_ms=quality_blob["window_ms"],
            index_drift_bps=json.loads(row["drift_json"]),
            max_window_ms=quality_blob["max_window_ms"],
            max_index_drift_bps=quality_blob["max_index_drift_bps"],
            reasons=tuple(quality_blob["reasons"]),
        ),
        note=row["note"],
        content_hash=row["content_hash"],
    )


class ReplaySource:
    """A MarketDataSource backed by stored payloads.

    Satisfies the same protocol as the live client, so anything that reads market
    data works unchanged against a snapshot.
    """

    def __init__(self, snapshot: Snapshot) -> None:
        self._snapshot = snapshot
        self._index_reads: dict[str, int] = {}

    @property
    def label(self) -> str:
        return f"{self._snapshot.meta.source_label} (replay {self._snapshot.meta.snapshot_id})"

    @property
    def is_live(self) -> bool:
        return False

    def now(self) -> datetime:
        return self._snapshot.meta.as_of

    def _get(self, component: str) -> RawResponse:
        try:
            return self._snapshot.responses[component]
        except KeyError:
            raise MarketDataError(
                f"Snapshot {self._snapshot.meta.snapshot_id} has no component {component!r}. "
                "It was captured for a different universe."
            ) from None

    def fetch_instruments(self, currency: str, kind: Kind) -> RawResponse:
        return self._get(f"instruments:{currency}:{kind}")

    def fetch_index(self, index_name: str) -> RawResponse:
        """Return the bracket readings in the order they were captured."""
        seen = self._index_reads.get(index_name, 0)
        self._index_reads[index_name] = seen + 1
        order = ("index_after", "index_before") if seen else ("index_before", "index_after")
        for prefix in order:
            for component, response in self._snapshot.responses.items():
                if component.startswith(prefix) and response.provenance.params.get(
                    "index_name"
                ) == index_name:
                    return response
        raise MarketDataError(f"Snapshot holds no index reading for {index_name!r}.")

    def fetch_book_summary(self, currency: str, kind: Kind) -> RawResponse:
        return self._get(f"summary:{currency}:{kind}")


def rebuild_universe(snapshot: Snapshot) -> Universe:
    """Reconstruct the universe from stored bytes.

    as_of comes from the snapshot, never the clock, which is what makes repeated
    reloads produce identical downstream values.
    """
    meta = snapshot.meta
    instruments = []
    for currency in meta.spec.currencies:
        for kind in meta.spec.kinds:
            component = f"instruments:{currency}:{kind}"
            if component in snapshot.responses:
                instruments.extend(parse.instruments(snapshot.responses[component]))

    marks: dict[str, float] = {}
    for component, response in snapshot.responses.items():
        if component.startswith("summary:"):
            marks.update(parse.mark_prices(response))

    index_prices: dict[str, float] = {}
    for currency in meta.spec.currencies:
        readings = [
            parse.index_price(snapshot.responses[key])
            for key in (f"index_before:{currency}", f"index_after:{currency}")
            if key in snapshot.responses
        ]
        if readings:
            index_prices[currency] = sum(readings) / len(readings)

    return Universe(
        spec=meta.spec,
        as_of=meta.as_of,
        instruments=tuple(sorted(instruments, key=lambda i: i.name)),
        index_prices=index_prices,
        forward_curve=build_forward_curve(instruments, marks, meta.as_of),
    )


def verify(store: SnapshotStore, snapshot_id: str) -> dict[str, Any]:
    """Reload twice and confirm the derived dossier is identical."""
    from voltk.capture import dossier

    first = dossier(rebuild_universe(store.load(snapshot_id)))
    second = dossier(rebuild_universe(store.load(snapshot_id)))
    return {
        "snapshot_id": snapshot_id,
        "digest": first["digest"],
        "reproducible": first["digest"] == second["digest"],
    }


def payload_hashes(snapshot: Snapshot) -> dict[str, str]:
    return {component: response.sha256 for component, response in snapshot.responses.items()}


def components(responses: Iterable[tuple[str, RawResponse]]) -> Sequence[str]:
    return [component for component, _ in responses]


__all__ = [
    "DEFAULT_DB",
    "ReplaySource",
    "SCHEMA_VERSION",
    "Snapshot",
    "SnapshotError",
    "SnapshotMeta",
    "SnapshotStore",
    "payload_hashes",
    "rebuild_universe",
    "verify",
]


def now_utc() -> datetime:
    return datetime.now(UTC)

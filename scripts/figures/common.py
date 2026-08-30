"""Shared setup for the report figure scripts.

Every figure is rendered from a stored snapshot, not a fresh live pull, so
`make all` reproduces the same figures on repeated runs. If no snapshot
exists yet (a fresh checkout), one is captured once and reused by every
later script in the same `make all` run -- currency is discovered from the
venue, never hardcoded. Each script only renders results voltk already
computed; none of them price or fit anything themselves.
"""

from __future__ import annotations

from pathlib import Path

from voltk.capture import capture as run_capture
from voltk.instruments import Kind
from voltk.marketdata import ALL_CURRENCIES, DeribitClient
from voltk.marketdata import parse
from voltk.snapshots import Snapshot, SnapshotError, SnapshotStore, rebuild_universe
from voltk.universe import Universe, UniverseSpec, currencies_with

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"


def _discover_currency(client: DeribitClient) -> str:
    instruments = parse.instruments(client.fetch_instruments(ALL_CURRENCIES, Kind.OPTION))
    currencies = currencies_with(instruments, Kind.OPTION)
    if not currencies:
        raise RuntimeError("No option currencies discovered from the venue.")
    return currencies[0]


def _latest_snapshot(store: SnapshotStore) -> Snapshot | None:
    existing = store.list(limit=1)
    if not existing:
        return None
    try:
        return store.load(existing[0].snapshot_id)
    except SnapshotError:
        return None  # stored under a different library version; recapture below


def load_universe_and_marks() -> tuple[Universe, dict[str, float]]:
    store = SnapshotStore()
    snapshot = _latest_snapshot(store)

    if snapshot is None:
        client = DeribitClient()
        currency = _discover_currency(client)
        captured = run_capture(client, UniverseSpec.of((currency,)), note="scripts/figures bootstrap")
        store.save(captured)
        snapshot = _latest_snapshot(store)

    universe = rebuild_universe(snapshot)
    currency = universe.spec.currencies[0]
    component = f"summary:{currency}:{Kind.OPTION}"
    marks = dict(parse.mark_prices(snapshot.responses[component])) if component in snapshot.responses else {}
    return universe, marks


def savefig(figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    return path

"""Synchronous capture. REST cannot give a true instant, so the capture is
bracketed: index read before and after everything else, with drift and
elapsed window recorded and gated -- a breach marks the capture degraded
rather than discarding it. Order matters: slowest-moving data (definitions)
goes first, quotes go last, closest to the second index read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from voltk.canonical import digest
from voltk.instruments import Instrument, Kind
from voltk.marketdata import parse
from voltk.marketdata.base import MarketDataError, MarketDataSource
from voltk.provenance import RawResponse
from voltk.universe import ForwardPoint, Universe, UniverseSpec

DEFAULT_MAX_WINDOW_MS = 6000.0
DEFAULT_MAX_INDEX_DRIFT_BPS = 15.0

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Quality:
    window_ms: float
    index_drift_bps: dict[str, float]
    max_window_ms: float
    max_index_drift_bps: float
    reasons: tuple[str, ...] = ()
    replayed: bool = False

    @property
    def ok(self) -> bool:
        return not self.reasons

    @property
    def worst_drift_bps(self) -> float:
        return max(self.index_drift_bps.values(), default=0.0)

    def summary(self) -> str:
        if self.replayed:
            return f"replayed: {self.worst_drift_bps:.1f}bps drift at capture"
        state = "ok" if self.ok else "degraded"
        return f"{state}: {self.window_ms:.0f}ms window, {self.worst_drift_bps:.1f}bps drift"


@dataclass(frozen=True, slots=True)
class Capture:
    """Everything observed in one window, plus the raw bodies behind it."""

    spec: UniverseSpec
    source_label: str
    started_at: datetime
    completed_at: datetime
    quality: Quality
    responses: tuple[tuple[str, RawResponse], ...]
    universe: Universe
    schema_version: int = SCHEMA_VERSION
    note: str = ""

    @property
    def as_of(self) -> datetime:
        return self.universe.as_of

    def component(self, key: str) -> RawResponse:
        for name, response in self.responses:
            if name == key:
                return response
        raise KeyError(f"No captured component {key!r}.")


def _component_key(kind: str, currency: str, suffix: str = "") -> str:
    return f"{kind}:{currency}{':' + suffix if suffix else ''}"


def _drift_bps(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return abs(after - before) / before * 10_000


def capture(
    source: MarketDataSource,
    spec: UniverseSpec,
    *,
    max_window_ms: float = DEFAULT_MAX_WINDOW_MS,
    max_index_drift_bps: float = DEFAULT_MAX_INDEX_DRIFT_BPS,
    note: str = "",
) -> Capture:
    """Read the whole universe in one bracketed window."""
    started_at = datetime.now(UTC)
    responses: list[tuple[str, RawResponse]] = []

    definitions: dict[tuple[str, Kind], list[Instrument]] = {}
    for currency in spec.currencies:
        for kind in spec.kinds:
            response = source.fetch_instruments(currency, kind)
            responses.append((_component_key("instruments", currency, str(kind)), response))
            definitions[(currency, kind)] = parse.instruments(response)

    all_instruments = [inst for group in definitions.values() for inst in group]
    if not all_instruments:
        raise MarketDataError(
            f"No instruments returned for {spec.key()}. The universe cannot be captured."
        )

    index_map = parse.index_names(all_instruments)
    missing = [c for c in spec.currencies if c not in index_map]
    if missing:
        raise MarketDataError(
            f"No price index in metadata for {', '.join(missing)}. "
            "Index names are read from instruments, never assumed."
        )

    index_before: dict[str, float] = {}
    for currency, index_name in index_map.items():
        response = source.fetch_index(index_name)
        responses.append((_component_key("index_before", currency), response))
        index_before[currency] = parse.index_price(response)

    marks: dict[str, float] = {}
    for currency in spec.currencies:
        for kind in spec.kinds:
            response = source.fetch_book_summary(currency, kind)
            responses.append((_component_key("summary", currency, str(kind)), response))
            marks.update(parse.mark_prices(response))

    index_after: dict[str, float] = {}
    for currency, index_name in index_map.items():
        response = source.fetch_index(index_name)
        responses.append((_component_key("index_after", currency), response))
        index_after[currency] = parse.index_price(response)

    completed_at = datetime.now(UTC)

    # A replayed source has no capture window -- reading the clock here would
    # make the same snapshot produce a different tau on every reload.
    live = source.is_live
    window_ms = (completed_at - started_at).total_seconds() * 1000 if live else 0.0

    drift = {
        currency: _drift_bps(index_before[currency], index_after[currency])
        for currency in index_before
    }

    reasons: list[str] = []
    if live and window_ms > max_window_ms:
        reasons.append(f"capture took {window_ms:.0f}ms, limit {max_window_ms:.0f}ms")
    for currency, moved in sorted(drift.items()):
        if moved > max_index_drift_bps:
            reasons.append(
                f"{currency} index moved {moved:.1f}bps during capture, "
                f"limit {max_index_drift_bps:.1f}bps"
            )

    quality = Quality(
        window_ms=window_ms,
        index_drift_bps=drift,
        max_window_ms=max_window_ms,
        max_index_drift_bps=max_index_drift_bps,
        reasons=tuple(reasons),
        replayed=not live,
    )

    # Midpoint of the bracket, so tau is centred on the observation rather than
    # biased to whichever end the clock was read at.
    as_of = started_at + (completed_at - started_at) / 2 if live else source.now()
    mid_index = {c: (index_before[c] + index_after[c]) / 2 for c in index_before}

    universe = Universe(
        spec=spec,
        as_of=as_of,
        instruments=tuple(sorted(all_instruments, key=lambda i: i.name)),
        index_prices=mid_index,
        forward_curve=build_forward_curve(all_instruments, marks, as_of),
    )

    return Capture(
        spec=spec,
        source_label=source.label,
        started_at=started_at,
        completed_at=completed_at,
        quality=quality,
        responses=tuple(responses),
        universe=universe,
        note=note,
    )


def build_forward_curve(
    instruments: Sequence[Instrument],
    marks: Mapping[str, float],
    as_of: datetime,
) -> tuple[ForwardPoint, ...]:
    """Forward points from dated futures marks, nearest expiry first."""
    points: list[ForwardPoint] = []
    for inst in instruments:
        if inst.kind is not Kind.FUTURE or inst.expiry is None:
            continue
        if inst.is_expired(as_of):
            continue
        mark = marks.get(inst.name)
        if mark is None or mark <= 0:
            continue
        points.append(
            ForwardPoint(
                currency=inst.base_currency,
                expiry=inst.expiry,
                forward=float(mark),
                instrument_name=inst.name,
            )
        )
    points.sort(key=lambda p: (p.currency, p.expiry))
    return tuple(points)


def dossier(universe: Universe) -> dict[str, Any]:
    """Deterministic summary of a captured universe.

    Derived only from the universe and its as_of, never from the wall clock, so
    the same snapshot always produces the same dossier.
    """
    per_currency: dict[str, Any] = {}
    for currency in universe.spec.currencies:
        options = universe.options(currency)
        futures = universe.futures(currency)
        expiries = universe.expiries(currency)
        inverse = sum(1 for i in options if i.is_inverse)
        contract_sizes = sorted({i.contract_size for i in options})
        tick_sizes = sorted({i.tick_size for i in options})
        settlement_currencies = sorted({i.settlement_currency for i in options})

        per_currency[currency] = {
            "option_count": len(options),
            "future_count": len(futures),
            "expiry_count": len(expiries),
            "expiries": [e.isoformat() for e in expiries],
            "strike_count": len({i.strike for i in options if i.strike is not None}),
            "inverse_count": inverse,
            "linear_count": len(options) - inverse,
            "contract_sizes": contract_sizes,
            "tick_sizes": tick_sizes,
            "settlement_currencies": settlement_currencies,
            "index_price": universe.index_prices.get(currency),
            "forward_points": [
                {
                    "expiry": p.expiry.isoformat(),
                    "forward": p.forward,
                    "instrument": p.instrument_name,
                }
                for p in universe.forwards(currency)
            ],
            "tau_front": (
                min(t for t in (i.tau(universe.as_of) for i in options) if t is not None)
                if options
                else None
            ),
        }

    body = {
        "as_of": universe.as_of.isoformat(),
        "spec": {
            "currencies": list(universe.spec.currencies),
            "kinds": sorted(str(k) for k in universe.spec.kinds),
        },
        "instrument_count": len(universe.instruments),
        "currencies": per_currency,
    }
    return body | {"digest": digest(body)}

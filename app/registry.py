"""Panel registry: a milestone adds a panel by adding one file under
app/panels/ and decorating its render function; app/main.py never changes.
slot/order are explicit, never inferred from filenames.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Iterable


class Slot(StrEnum):
    GLOBAL = "global"
    FEATURED = "featured"
    LEFT_RAIL = "left_rail"
    MAIN = "main"
    QUOTER_RAIL = "quoter_rail"
    QUOTER_MAIN = "quoter_main"


RenderFn = Callable[[], None]


@dataclass(frozen=True, slots=True)
class PanelSpec:
    key: str
    title: str
    slot: Slot
    order: int
    milestone: str
    render: RenderFn
    caption: str = ""


_REGISTRY: dict[str, PanelSpec] = {}


def panel(
    *,
    key: str,
    title: str,
    slot: Slot,
    order: int,
    milestone: str,
    caption: str = "",
) -> Callable[[RenderFn], RenderFn]:
    def decorator(fn: RenderFn) -> RenderFn:
        if key in _REGISTRY:
            raise ValueError(f"Panel key {key!r} registered twice.")
        _REGISTRY[key] = PanelSpec(key, title, slot, order, milestone, fn, caption)
        return fn

    return decorator


def discover(package: str = "app.panels") -> None:
    pkg = importlib.import_module(package)
    for module in pkgutil.iter_modules(pkg.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{package}.{module.name}")


def panels_for(slot: Slot) -> list[PanelSpec]:
    return sorted(
        (spec for spec in _REGISTRY.values() if spec.slot is slot),
        key=lambda spec: (spec.order, spec.key),
    )


def manifest() -> list[tuple[str, str, int]]:
    return sorted((spec.key, str(spec.slot), spec.order) for spec in _REGISTRY.values())


def all_panels() -> Iterable[PanelSpec]:
    return list(_REGISTRY.values())

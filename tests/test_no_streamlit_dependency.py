"""The library must not depend on the view layer.

Two checks, because either alone can be worked around: a static scan catches
imports on paths that never execute, and a runtime import with the view packages
poisoned catches dynamic imports the scan would miss.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE = SRC / "voltk"

FORBIDDEN = {"streamlit", "plotly", "app"}


def _module_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_module_does_not_import_view_layer(path: Path) -> None:
    offenders = _imported_roots(path) & FORBIDDEN
    assert not offenders, f"{path.relative_to(SRC)} imports {sorted(offenders)}."


class _Poison:
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname.split(".")[0] in FORBIDDEN:
            raise AssertionError(f"voltk imported {fullname!r} at import time.")
        return None


def test_package_imports_with_view_layer_unavailable() -> None:
    poison = _Poison()
    saved = {name: mod for name, mod in sys.modules.items() if name.startswith("voltk")}
    for name in saved:
        del sys.modules[name]

    sys.meta_path.insert(0, poison)
    try:
        for module in pkgutil.walk_packages([str(PACKAGE)], prefix="voltk."):
            importlib.import_module(module.name)
    finally:
        sys.meta_path.remove(poison)
        sys.modules.update(saved)

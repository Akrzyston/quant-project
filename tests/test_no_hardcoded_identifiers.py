"""No instrument identifier may appear as a literal in the library.

Currencies, instrument names and index names are discovered from the venue at
runtime. A literal is how a toolkit quietly becomes single-asset: the first
hardcoded ticker works, and every later one is a special case.

Docstrings are exempt because documentation is not a code path. Anything that
can be compared against or assigned is checked.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE = SRC / "voltk"

TICKER = re.compile(
    r"\b(BTC|ETH|SOL|XRP|USDC|USDT|EURR|PAXG|DOGE|ADA|LTC|BCH|LINK|AVAX|DOT|MATIC|BNB|TRX)\b",
    re.IGNORECASE,
)
INSTRUMENT_NAME = re.compile(r"[A-Z]{2,6}[-_]\d{1,2}[A-Z]{3}\d{2}")
INDEX_NAME = re.compile(r"\b[a-z]{2,6}_(usd|usdc|usdt)\b")
PERPETUAL = re.compile(r"\b[A-Z]{2,6}-PERPETUAL\b")

PATTERNS = {
    "currency ticker": TICKER,
    "instrument name": INSTRUMENT_NAME,
    "index name": INDEX_NAME,
    "perpetual name": PERPETUAL,
}


def _python_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Constant nodes that are docstrings, identified by position."""
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return docstrings


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_instrument_literals(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = _docstring_nodes(tree)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        for label, pattern in PATTERNS.items():
            match = pattern.search(node.value)
            if match:
                offenders.append(f"line {node.lineno}: {label} {match.group(0)!r}")

    assert not offenders, (
        f"{path.relative_to(SRC)} contains instrument literals: {offenders}. "
        "Discover these from the venue or read them from a spec file."
    )


def test_spec_files_are_not_currency_keyed() -> None:
    """Spec data describes the venue, not individual assets."""
    import json

    for spec_path in (PACKAGE / "specs").glob("*.json"):
        text = spec_path.read_text()
        json.loads(text)
        match = TICKER.search(text)
        assert match is None, (
            f"{spec_path.name} mentions {match.group(0)!r}. Venue conventions apply "
            "across assets; anything per-asset belongs in instrument metadata."
        )

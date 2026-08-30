"""M2: implied forward from put-call parity against the traded future --
the basis Deribit's own chain display misses by pricing off the index.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from common import load_universe_and_marks, savefig
from voltk.forward import implied_forward_curve


def main() -> None:
    universe, marks = load_universe_and_marks()
    currency = universe.spec.currencies[0]
    curve = sorted(implied_forward_curve(universe, marks, currency=currency), key=lambda f: f.expiry)
    if not curve:
        print("No implied forward curve available; skipping.")
        return

    expiries = [f.expiry for f in curve]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[2, 1])

    ax1.plot(expiries, [f.forward for f in curve], "o-", label="Implied forward (parity)")
    ax1.plot(expiries, [f.traded_future for f in curve], "s--", label="Traded future")
    ax1.set_ylabel(f"{currency} forward")
    ax1.set_title(f"{currency}: implied vs traded forward, {universe.as_of:%Y-%m-%d %H:%M UTC}")
    ax1.legend()

    ax2.bar(expiries, [f.basis_bps for f in curve], width=2.0, color="tab:gray")
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylabel("Basis (bps)")
    fig.autofmt_xdate()

    savefig(fig, "forward_curve")


if __name__ == "__main__":
    main()

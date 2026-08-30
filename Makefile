.PHONY: all

PYTHON := PYTHONPATH=src uv run python

all:
	$(PYTHON) scripts/figures/forward_curve.py
	$(PYTHON) scripts/figures/smile.py
	$(PYTHON) scripts/figures/skew_delta.py
	$(PYTHON) scripts/figures/pnl_attribution.py
	$(PYTHON) scripts/figures/vol_dynamics.py
	$(PYTHON) scripts/figures/market_making.py
	$(PYTHON) scripts/figures/strategy.py

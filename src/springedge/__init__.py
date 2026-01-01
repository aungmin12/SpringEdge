"""
SpringEdge: feature generation for an "Edge system".

Primary entrypoint:
- `springedge.features.compute_edge_features`
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "compute_edge_features",
    "EdgeEvaluation",
    "IndicatorSpec",
    "apply_indicators_to_candidates",
    "build_candidate_list",
    "default_edge_evaluation",
    "fetch_ohlcv_daily",
    "run_edge",
    "StockUniverse",
    "fetch_sp500_baseline",
    "fetch_market_regime_last_n_days",
    "fetch_regime_daily",
    "market_regime_counts_and_trend",
    "quarterly_regime_profile",
    "fetch_score_name_groups",
    "fetch_actionable_score_names",
    "connect_db",
    "db_connection",
]

# NOTE:
# We intentionally *avoid* importing heavy submodules at package import time.
# This keeps `python -m springedge.edge` from emitting a noisy runpy warning
# ("found in sys.modules ... prior to execution") and generally speeds up imports.
_EXPORTS: dict[str, tuple[str, str]] = {
    # features
    "compute_edge_features": ("springedge.features", "compute_edge_features"),
    # edge
    "EdgeEvaluation": ("springedge.edge", "EdgeEvaluation"),
    "IndicatorSpec": ("springedge.edge", "IndicatorSpec"),
    "apply_indicators_to_candidates": (
        "springedge.edge",
        "apply_indicators_to_candidates",
    ),
    "build_candidate_list": ("springedge.edge", "build_candidate_list"),
    "default_edge_evaluation": ("springedge.edge", "default_edge_evaluation"),
    "fetch_ohlcv_daily": ("springedge.edge", "fetch_ohlcv_daily"),
    "run_edge": ("springedge.edge", "run_edge"),
    # layers
    "StockUniverse": ("springedge.layers", "StockUniverse"),
    "fetch_sp500_baseline": ("springedge.layers", "fetch_sp500_baseline"),
    # db
    "connect_db": ("springedge.db", "connect_db"),
    "db_connection": ("springedge.db", "db_connection"),
    # regime
    "fetch_market_regime_last_n_days": (
        "springedge.regime",
        "fetch_market_regime_last_n_days",
    ),
    "fetch_regime_daily": ("springedge.regime", "fetch_regime_daily"),
    "market_regime_counts_and_trend": (
        "springedge.regime",
        "market_regime_counts_and_trend",
    ),
    "quarterly_regime_profile": ("springedge.regime", "quarterly_regime_profile"),
    # score_performance
    "fetch_score_name_groups": (
        "springedge.score_performance",
        "fetch_score_name_groups",
    ),
    "fetch_actionable_score_names": (
        "springedge.score_performance",
        "fetch_actionable_score_names",
    ),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    mod = import_module(mod_name)
    return getattr(mod, attr_name)


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(globals().keys()) | set(__all__))

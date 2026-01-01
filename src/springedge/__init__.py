"""
SpringEdge: feature generation for an "Edge system".

Primary entrypoint:
- `springedge.features.compute_edge_features`
"""

from .features import compute_edge_features
from .edge import (
    EdgeEvaluation,
    IndicatorSpec,
    apply_indicators_to_candidates,
    build_candidate_list,
    default_edge_evaluation,
    fetch_ohlcv_daily,
    run_edge,
)
from .layers import StockUniverse, fetch_sp500_baseline
from .db import connect_db, db_connection
from .regime import (
    fetch_market_regime_last_n_days,
    fetch_regime_daily,
    market_regime_counts_and_trend,
    quarterly_regime_profile,
)

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
    "connect_db",
    "db_connection",
]

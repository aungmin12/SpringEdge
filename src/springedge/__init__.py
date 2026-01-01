"""
SpringEdge: feature generation for an "Edge system".

Primary entrypoint:
- `springedge.features.compute_edge_features`
"""

from .features import compute_edge_features
from .layers import StockUniverse, fetch_sp500_baseline
from .db import connect_db, db_connection
from .regime import fetch_regime_daily, quarterly_regime_profile

__all__ = [
    "compute_edge_features",
    "StockUniverse",
    "fetch_sp500_baseline",
    "fetch_regime_daily",
    "quarterly_regime_profile",
    "connect_db",
    "db_connection",
]

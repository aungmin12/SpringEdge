"""
SpringEdge: feature generation for an "Edge system".

Primary entrypoint:
- `springedge.features.compute_edge_features`
"""

from .features import compute_edge_features
from .layers import StockUniverse
from .db import connect_db, db_connection

__all__ = ["compute_edge_features", "StockUniverse", "connect_db", "db_connection"]

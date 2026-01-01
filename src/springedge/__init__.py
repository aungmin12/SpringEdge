"""
SpringEdge: feature generation for an "Edge system".

Primary entrypoint:
- `springedge.features.compute_edge_features`
"""

from .features import compute_edge_features
from .layers import StockUniverse

__all__ = ["compute_edge_features", "StockUniverse"]

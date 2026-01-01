"""
Compatibility package for external orchestrators.

Some schedulers reference modules by dotted import path. This project’s core
library is `springedge`, but we also expose a minimal `intelligence.*` surface
area to keep those import paths stable.
"""

from __future__ import annotations

__all__ = ["sp500_tickers"]


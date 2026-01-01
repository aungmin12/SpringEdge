"""
Compatibility shim for running the repo version via:

  python3 -m score_performance ...

Some environments also have an unrelated `score_performance` module installed
in site-packages; putting this shim at the repo root ensures `-m score_performance`
uses *this* project’s CLI instead.
"""

from __future__ import annotations

from springedge.score_performance import main


if __name__ == "__main__":
    raise SystemExit(main())


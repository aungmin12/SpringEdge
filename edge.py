"""
Convenience wrapper so you can run from repo root:

  python3 edge.py --demo

This simply forwards to `springedge.edge.main` while ensuring `src/` is on sys.path.
Recommended invocation (after install) remains:

  python3 -m springedge.edge
"""

from __future__ import annotations

import os
import sys


def _ensure_src_on_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(repo_root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def main() -> int:
    _ensure_src_on_path()
    from springedge.edge import main as _main

    return int(_main())


if __name__ == "__main__":
    raise SystemExit(main())


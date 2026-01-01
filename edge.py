"""
Convenience wrapper so you can run from repo root:

  python3 edge.py --demo
  python3 edge.py --score-performance --demo

This simply forwards to `springedge.edge.main` while ensuring `src/` is on sys.path.
Recommended invocation (after install) remains:

  python3 -m springedge.edge

Score performance helper:

  python3 edge.py --score-performance --demo
  python3 -m springedge.score_performance --demo
"""

from __future__ import annotations

import os
import sys


def _ensure_src_on_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(repo_root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _pop_flag(argv: list[str], flag: str) -> bool:
    try:
        i = argv.index(flag)
    except ValueError:
        return False
    argv.pop(i)
    return True


def main() -> int:
    _ensure_src_on_path()
    argv = sys.argv[1:]
    if _pop_flag(argv, "--score-performance"):
        from springedge.score_performance import main as _sp_main

        return int(_sp_main(argv))

    from springedge.edge import main as _main

    return int(_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())


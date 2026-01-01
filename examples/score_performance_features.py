from __future__ import annotations

import json

from springedge import db_connection, fetch_score_name_groups


def main() -> None:
    """
    Print distinct score_name values grouped by horizon_days and regime_label.

    Usage:
      export SPRINGEDGE_DB_URL="postgresql://USER:PASSWORD@HOST:5432/DBNAME"
      python3 examples/score_performance_features.py
    """
    with db_connection() as conn:
        df = fetch_score_name_groups(conn)

    # Pretty JSON for copy/paste into docs or tickets.
    payload = [
        {
            "horizon_days": int(row.horizon_days),
            "regime_label": str(row.regime_label),
            "n_scores": int(row.n_scores),
            "score_names": list(row.score_names),
        }
        for row in df.itertuples(index=False)
    ]
    print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()

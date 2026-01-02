from springedge.edge import main


def test_edge_cli_accepts_selection_flags_in_demo_mode() -> None:
    # Regression: these flags should be accepted by the CLI (argparse),
    # even when run via the repo-root `edge.py` wrapper.
    rc = main(
        [
            "--demo",
            "--topdown",
            "--persist-topdown",
            "--select-qualified",
            "--select-as-of",
            "2025-12-31",
            "--top",
            "0",
        ]
    )
    assert int(rc) == 0


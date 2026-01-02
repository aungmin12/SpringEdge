from springedge.edge import main


def test_edge_cli_accepts_select_qualified_flag_in_demo_mode() -> None:
    # Should not error even though demo DB doesn't include intelligence tables.
    rc = int(main(["--demo", "--select-qualified"]))
    assert rc == 0


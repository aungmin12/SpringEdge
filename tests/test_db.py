import os

import pytest

from springedge.db import connect_db, db_connection, get_db_url


def test_get_db_url_requires_env_or_param(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPRINGEDGE_DB_URL", raising=False)
    with pytest.raises(ValueError):
        get_db_url()

    assert get_db_url("sqlite:///:memory:") == "sqlite:///:memory:"


def test_db_connection_sqlite_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPRINGEDGE_DB_URL", "sqlite:///:memory:")
    with db_connection() as conn:
        # sqlite3 uses .execute directly
        got = conn.execute("select 1").fetchone()
    assert got[0] == 1


def test_connect_db_postgres_requires_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    # If a postgres driver is installed in this environment, we can't reliably
    # connect in CI; skip instead.
    try:
        import psycopg  # noqa: F401

        pytest.skip("psycopg installed; skipping driver-missing assertion")
    except ModuleNotFoundError:
        pass
    try:
        import psycopg2  # noqa: F401

        pytest.skip("psycopg2 installed; skipping driver-missing assertion")
    except ModuleNotFoundError:
        pass

    monkeypatch.setenv("SPRINGEDGE_DB_URL", "postgresql://user:pass@localhost:5432/dbname")
    with pytest.raises(ModuleNotFoundError):
        connect_db()


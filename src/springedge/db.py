from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

DEFAULT_DB_URL_ENV = "SPRINGEDGE_DB_URL"
FALLBACK_DB_URL_ENV = "DATABASE_URL"


def get_db_url(db_url: str | None = None, *, env_var: str = DEFAULT_DB_URL_ENV) -> str:
    """
    Resolve a database URL from an explicit value or an environment variable.

    Users typically export:
      SPRINGEDGE_DB_URL="postgresql://user:pass@host:5432/dbname"
    """
    if db_url and str(db_url).strip():
        return str(db_url).strip()
    primary = os.getenv(env_var, "").strip()
    if primary:
        return primary

    # Common convention in many apps/tooling.
    fallback = os.getenv(FALLBACK_DB_URL_ENV, "").strip()
    if fallback:
        return fallback

    raise ValueError(
        "Missing database URL. Provide db_url=... or set env var "
        f"{env_var}=... (or {FALLBACK_DB_URL_ENV}=...)."
    )


def connect_db(db_url: str | None = None, *, env_var: str = DEFAULT_DB_URL_ENV, **kwargs: Any) -> Any:
    """
    Create a DB-API connection based on a URL.

    Supported schemes:
    - sqlite:///:memory:
    - sqlite:////absolute/path/to.db
    - postgresql://... (requires 'psycopg' or 'psycopg2' installed)

    Returns a DB-API connection object (type depends on the driver).
    """
    url = get_db_url(db_url, env_var=env_var)
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()

    if scheme in {"sqlite", "sqlite3"}:
        # Examples:
        # - sqlite:///:memory:
        # - sqlite:////tmp/foo.db
        # - sqlite:///relative.db (treated as relative path)
        path = unquote(parsed.path or "")
        if parsed.netloc and not path:
            # Rare form: sqlite://localhost/path.db -> treat netloc as path
            path = f"/{parsed.netloc}"

        if path in {"", "/"}:
            raise ValueError(f"Invalid sqlite URL (missing path): {url}")

        if path == "/:memory:":
            sqlite_path = ":memory:"
        else:
            sqlite_path = path[1:] if path.startswith("/") else path

        return sqlite3.connect(sqlite_path, **kwargs)

    if scheme in {"postgres", "postgresql"}:
        # Prefer psycopg (v3), fall back to psycopg2.
        try:
            import psycopg  # type: ignore

            return psycopg.connect(url, **kwargs)
        except ModuleNotFoundError:
            pass

        try:
            import psycopg2  # type: ignore

            return psycopg2.connect(url, **kwargs)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Postgres driver not installed. Install one of:\n"
                "  - pip install 'psycopg[binary]'\n"
                "  - pip install psycopg2-binary\n"
                f"Then set {env_var} to a postgresql:// URL."
            ) from e

    raise ValueError(
        f"Unsupported DB scheme '{scheme}' in URL: {url}. "
        "Supported: sqlite://, postgresql://"
    )


@contextmanager
def db_connection(
    db_url: str | None = None, *, env_var: str = DEFAULT_DB_URL_ENV, **kwargs: Any
) -> Iterator[Any]:
    """
    Context manager for a database connection.

    Example:
        from springedge.db import db_connection

        with db_connection() as conn:
            rows = conn.execute("select 1").fetchall()
    """
    conn = connect_db(db_url, env_var=env_var, **kwargs)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            # Some drivers may not have a standard close() or may already be closed.
            pass


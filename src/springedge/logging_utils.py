from __future__ import annotations

import logging
from typing import Iterable


class TableLogFormatter(logging.Formatter):
    """
    Render logs in a stable, readable, "table-like" console format.

    Example:
      2026-01-01 12:34:56 | INFO    | springedge.edge           | run_edge: candidates=123
                          |         |                          | <continuation line>
    """

    def __init__(
        self,
        *,
        datefmt: str = "%Y-%m-%d %H:%M:%S",
        ts_width: int = 19,
        level_width: int = 7,
        name_width: int = 26,
        ellipsis: str = "...",
    ) -> None:
        super().__init__(datefmt=datefmt)
        self._ts_width = int(ts_width)
        self._level_width = int(level_width)
        self._name_width = int(name_width)
        self._ellipsis = str(ellipsis)

    def _crop(self, s: str, width: int, *, mode: str = "right") -> str:
        s = "" if s is None else str(s)
        if width <= 0:
            return s
        if len(s) <= width:
            return s
        ell = self._ellipsis
        if not ell:
            return s[:width]
        if width <= len(ell):
            return s[:width]
        if mode == "left":
            return ell + s[-(width - len(ell)) :]
        return s[: width - len(ell)] + ell

    def _format_row(self, ts: str, level: str, name: str, msg: str) -> str:
        ts_cell = self._crop(ts, self._ts_width).ljust(self._ts_width)
        level_cell = self._crop(level, self._level_width).ljust(self._level_width)
        # Keep the *rightmost* part of long logger names (usually most specific),
        # e.g. "...score_performance" rather than "springedge.score_per..."
        name_cell = self._crop(name, self._name_width, mode="left").ljust(
            self._name_width
        )
        return f"{ts_cell} | {level_cell} | {name_cell} | {msg}"

    def _display_logger_name(self, name: str) -> str:
        """
        Produce a compact logger name for console logs.

        We intentionally hide the top-level package prefix for readability:
          "springedge.edge" -> "edge"
          "springedge.score_performance" -> "score_performance"
        """
        n = "" if name is None else str(name)
        prefix = "springedge."
        if n.startswith(prefix):
            return n[len(prefix) :]
        return n

    def format(self, record: logging.LogRecord) -> str:
        # Keep logging's semantics for %-formatting, lazy args, etc.
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        # Split multi-line messages into aligned continuation rows.
        lines = message.splitlines() or [""]

        ts = self.formatTime(record, self.datefmt)
        level = record.levelname
        name = self._display_logger_name(record.name)

        first = self._format_row(ts, level, name, lines[0])
        out: list[str] = [first]
        if len(lines) > 1:
            # Make continuation lines readable by avoiding repeated empty "table cells".
            # Align under the message column and prefix with a subtle marker.
            msg_indent = " " * (self._ts_width + 3 + self._level_width + 3 + self._name_width + 3)
            for line in lines[1:]:
                out.append(f"{msg_indent}↳ {line}")
        return "\n".join(out)


def configure_logging(
    *,
    level: int = logging.INFO,
    handlers: Iterable[logging.Handler] | None = None,
) -> None:
    """
    Configure root logging with a table-like formatter.

    Intended for CLIs (safe to call multiple times).
    """
    if handlers is None:
        h = logging.StreamHandler()
        h.setFormatter(TableLogFormatter())
        handlers = [h]

    # Ensure our formatter is actually used, even if another library has already
    # configured the root logger (common in notebooks/REPLs and some CLIs).
    #
    # `force=True` (Python >=3.8) removes any existing root handlers first.
    logging.basicConfig(level=level, handlers=list(handlers), force=True)


__all__ = ["TableLogFormatter", "configure_logging"]

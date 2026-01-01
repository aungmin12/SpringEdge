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
    ) -> None:
        super().__init__(datefmt=datefmt)
        self._ts_width = int(ts_width)
        self._level_width = int(level_width)
        self._name_width = int(name_width)

    def _crop(self, s: str, width: int) -> str:
        s = "" if s is None else str(s)
        if width <= 0:
            return s
        if len(s) <= width:
            return s
        if width <= 1:
            return s[:width]
        return s[: width - 1] + "…"

    def _format_row(self, ts: str, level: str, name: str, msg: str) -> str:
        ts_cell = self._crop(ts, self._ts_width).ljust(self._ts_width)
        level_cell = self._crop(level, self._level_width).ljust(self._level_width)
        name_cell = self._crop(name, self._name_width).ljust(self._name_width)
        return f"{ts_cell} | {level_cell} | {name_cell} | {msg}"

    def format(self, record: logging.LogRecord) -> str:
        # Keep logging's semantics for %-formatting, lazy args, etc.
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        # Split multi-line messages into aligned continuation rows.
        lines = message.splitlines() or [""]

        ts = self.formatTime(record, self.datefmt)
        level = record.levelname
        name = record.name

        out: list[str] = [self._format_row(ts, level, name, lines[0])]
        if len(lines) > 1:
            blank_ts = ""
            blank_level = ""
            blank_name = ""
            for line in lines[1:]:
                out.append(self._format_row(blank_ts, blank_level, blank_name, line))
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

    # Avoid duplicate handlers if the CLI calls this more than once.
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    logging.basicConfig(level=level, handlers=list(handlers))


__all__ = ["TableLogFormatter", "configure_logging"]


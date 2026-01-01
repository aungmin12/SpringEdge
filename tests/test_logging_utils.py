import logging

from springedge.logging_utils import TableLogFormatter


def test_table_log_formatter_crops_logger_name_from_left_to_preserve_suffix() -> None:
    fmt = TableLogFormatter(name_width=12, ellipsis="...")
    record = logging.LogRecord(
        name="springedge.score_performance",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    out = fmt.format(record)
    assert "...rformance" in out
    assert "springedge" not in out


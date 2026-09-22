"""Structured logging setup for PairScout (UTC timestamps)."""

import logging
import sys
import time


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fmt.converter = time.gmtime  # type: ignore[attr-defined]
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

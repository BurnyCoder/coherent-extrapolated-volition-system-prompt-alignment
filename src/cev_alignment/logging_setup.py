"""Timestamped logging to the terminal and to `logs/run_<UTC timestamp>.log`, in real time.

Local context: configures the root logger once, so messages from this package and from libraries
(inspect_ai, darkbench, humanebench, httpx) all land in the same file and on stderr.

Global context: phase 2 (`setup_logging`) of `cev_alignment.main`. Inspect additionally writes its own
per-eval `.eval` logs under `logs/inspect/<condition>/` (viewable with `uv run inspect view`).
Pattern adapted from the sibling project llms-take-autism-test (utils.configure_logging).
Docs: https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime  # timezone-aware UTC timestamps for file names
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"  # asctime = local wall-clock time


def utc_stamp() -> str:
    """Compact UTC timestamp for file names, e.g. 20261016T213000Z."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def setup_logging(log_dir: Path, name: str = "run") -> Path:
    """Send INFO+ records to stderr and to `log_dir/<name>_<UTC>.log`; return the log file path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{utc_stamp()}.log"
    formatter = logging.Formatter(LOG_FORMAT)
    stream_handler = logging.StreamHandler()  # stderr, flushed after every record
    file_handler = logging.FileHandler(log_path, encoding="utf-8")  # FileHandler flushes after every record
    for handler in (stream_handler, file_handler):
        handler.setFormatter(formatter)
    # force=True replaces handlers left by earlier calls (e.g. in tests) instead of duplicating output.
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per HTTP request is too noisy
    logging.getLogger(__name__).info("Logging to %s", log_path)
    return log_path

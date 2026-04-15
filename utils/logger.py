"""
Structured logging setup.
One file handler (JSON-like lines) + console handler.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON for easy parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload)


def setup_logger(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("rag")
    logger.setLevel(level)

    if logger.handlers:          # avoid duplicate handlers on reload
        return logger

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)

    # Console handler (human-friendly)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    return logger


def get_logger(name: str = "rag") -> logging.Logger:
    return logging.getLogger(name)

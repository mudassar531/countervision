"""Tiny logging helper shared by every pipeline entrypoint."""

from __future__ import annotations

import logging
import os
import sys


def configure_logging(level: str | int | None = None) -> logging.Logger:
    """Configure root logging once and return the package logger.

    The level can be overridden with the ``COUNTERVISION_LOG_LEVEL`` env var
    (e.g. ``DEBUG``); the explicit ``level`` argument wins if provided.
    """
    if level is None:
        level = os.environ.get("COUNTERVISION_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = level.upper()

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)

    root.setLevel(level)
    return logging.getLogger("countervision")

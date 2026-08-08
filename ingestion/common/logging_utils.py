from __future__ import annotations

import logging
import os

from mlops.structured_logging import configure_logging


_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured

    if not _configured:
        configure_logging(
            os.getenv("LOG_LEVEL", "INFO")
        )
        _configured = True

    return logging.getLogger(name)
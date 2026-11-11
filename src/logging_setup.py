"""Central logging. One file, one format, every layer.

Enabled everywhere (CLI, Streamlit, tests) so a failure in the UI leaves the
same trace as a failure in a script. Streamlit swallows stdout from callbacks,
which is precisely why UI problems need a file to look at.

    LOG_LEVEL=DEBUG  .venv/bin/streamlit run app.py
    tail -f logs/ai-lawyer.log
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "ai-lawyer.log"
_CONFIGURED = False


def setup(level: str | None = None) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("ailawyer")
    if _CONFIGURED:
        return root

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lvl = getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    root.setLevel(lvl)
    root.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)-18s %(message)s", datefmt="%H:%M:%S"
    )
    fh = RotatingFileHandler(LOG_FILE, maxBytes=4_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(lvl)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    sh.setLevel(lvl)
    root.addHandler(sh)

    _CONFIGURED = True
    root.debug("logging configured -> %s (level=%s)", LOG_FILE, logging.getLevelName(lvl))
    return root


def get(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(f"ailawyer.{name}")

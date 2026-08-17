#!/usr/bin/env python
"""Root packaging entry — real metadata lives in ``setup/``."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "setup" / "setup.py"), run_name="__main__")

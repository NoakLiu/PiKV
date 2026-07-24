#!/usr/bin/env python3
"""
Legacy entry point — delegates to repo-root ``data.download_data``.

Preferred:
  python -m data.download_data
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.download_data import download


def generate_data():
    print("=== Canonical data/ (repo root) ===")
    download()
    # Optional: also refresh legacy path for old scripts that hard-code relative data/
    legacy = os.path.join(os.path.dirname(__file__), "data")
    print("\n=== Legacy next_tok_pred/data (symlink-friendly copy of prompts) ===")
    os.makedirs(legacy, exist_ok=True)
    download(out_dir=legacy)


if __name__ == "__main__":
    generate_data()

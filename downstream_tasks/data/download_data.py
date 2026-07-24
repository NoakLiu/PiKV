#!/usr/bin/env python3
"""Shim — use ``python -m data.download_data`` instead."""

from data.download_data import download, main

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reproduce Table 5 matching coverage across the full ATen input inventory."""
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    runner = Path(__file__).resolve().parents[1] / 'reproduce.py'
    return subprocess.call([sys.executable, str(runner), 'aten_table5',
                            *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    sys.exit(main())

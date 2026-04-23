"""Backward-compatible wrapper for the environment inspection CLI."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.inspect_env import main


if __name__ == "__main__":
    main()

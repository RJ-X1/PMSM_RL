"""Backward-compatible wrapper for checkpoint sweep evaluation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.batch_eval_checkpoints import main


if __name__ == "__main__":
    main()

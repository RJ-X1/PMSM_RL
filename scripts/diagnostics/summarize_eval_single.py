"""Print tracking metrics for an exported evaluation CSV."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from utils.metrics import summarize_eval_csv



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize PMSM evaluation CSV metrics")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--as-json", action="store_true")
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    summary = summarize_eval_csv(args.csv_path)
    if args.as_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

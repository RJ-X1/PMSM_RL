"""Helpers for canonical experiment run directory layout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


@dataclass(slots=True)
class RunLayout:
    """Canonical filesystem layout for one experiment run."""

    run_name: str
    run_dir: Path
    checkpoints_dir: Path
    eval_dir: Path
    summaries_dir: Path
    figures_dir: Path
    configs_dir: Path
    train_log_csv: Path
    metadata_json: Path


def make_run_layout(run_name: str, output_dir: str | Path = "outputs") -> RunLayout:
    output_root = Path(output_dir)
    run_dir = output_root / "runs" / run_name
    return RunLayout(
        run_name=run_name,
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        eval_dir=run_dir / "eval",
        summaries_dir=run_dir / "summaries",
        figures_dir=run_dir / "figures",
        configs_dir=run_dir / "configs",
        train_log_csv=run_dir / "train_log.csv",
        metadata_json=run_dir / "metadata.json",
    )


def ensure_run_layout(layout: RunLayout) -> RunLayout:
    for path in (
        layout.run_dir,
        layout.checkpoints_dir,
        layout.eval_dir,
        layout.summaries_dir,
        layout.figures_dir,
        layout.configs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def snapshot_config_file(config_path: str | Path, destination_dir: str | Path) -> Path:
    src = Path(config_path)
    dst_dir = Path(destination_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def snapshot_run_configs(
    layout: RunLayout,
    *,
    env_config_path: str | Path,
    train_config_path: str | Path,
    eval_config_path: str | Path,
) -> dict[str, str]:
    """Copy canonical config files into a run-local snapshot directory."""
    ensure_run_layout(layout)
    env_snapshot = snapshot_config_file(env_config_path, layout.configs_dir)
    train_snapshot = snapshot_config_file(train_config_path, layout.configs_dir)
    eval_snapshot = snapshot_config_file(eval_config_path, layout.configs_dir)
    return {
        "env": str(env_snapshot),
        "train": str(train_snapshot),
        "eval": str(eval_snapshot),
    }


def write_run_metadata(
    layout: RunLayout,
    *,
    agent_name: str,
    task_name: str,
    env_id: str,
    canonical_config_paths: dict[str, str],
    snapshot_config_paths: dict[str, str],
    creation_time: str | None = None,
) -> Path:
    """Write or refresh run metadata while preserving creation time when possible."""
    ensure_run_layout(layout)
    existing: dict[str, Any] = {}
    if layout.metadata_json.exists():
        try:
            existing = json.loads(layout.metadata_json.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    created_at = creation_time or str(existing.get("creation_time") or datetime.now(timezone.utc).isoformat())
    payload = {
        "run_name": layout.run_name,
        "agent_name": agent_name,
        "task_name": task_name,
        "env_id": env_id,
        "config_paths": canonical_config_paths,
        "snapshot_config_paths": snapshot_config_paths,
        "creation_time": created_at,
    }
    layout.metadata_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return layout.metadata_json


def infer_run_dir_from_path(path: str | Path) -> Path | None:
    """Best-effort run dir inference for paths under outputs/runs/<run_name>/..."""
    p = Path(path)
    parts = p.parts
    if "runs" not in parts:
        return None
    idx = parts.index("runs")
    if idx + 1 >= len(parts):
        return None
    return Path(*parts[: idx + 2])

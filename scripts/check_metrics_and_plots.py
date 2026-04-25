"""Check the canonical metric schema and plotting contract for PMSM experiments."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
from typing import Any

from scripts.analysis.summarize_eval_batch import summarize_one
from utils.config import load_yaml
from utils.metrics import summarize_eval_csv


DEFAULT_SCHEMA_PATH = ROOT / "configs" / "eval" / "metric_plot_schema.yaml"
PIPELINE_PATHS = {
    "single_run_trace_export": ROOT / "scripts" / "core" / "evaluate.py",
    "single_run_summary": ROOT / "scripts" / "diagnostics" / "summarize_eval_single.py",
    "batch_checkpoint_helper": ROOT / "scripts" / "analysis" / "batch_eval_checkpoints.py",
    "batch_paper_summary": ROOT / "scripts" / "analysis" / "summarize_eval_batch.py",
    "episode_trace_plot": ROOT / "scripts" / "plotting" / "plot_episode_trace.py",
    "training_curve_plot": ROOT / "scripts" / "plotting" / "plot_training_curves.py",
    "checkpoint_comparison_plot": ROOT / "scripts" / "plotting" / "plot_checkpoint_sweep.py",
    "paper_figure_toolkit": ROOT / "scripts" / "plotting" / "paper_figures.py",
    "failure_case_helper": ROOT / "scripts" / "plotting" / "plot_failure_case.py",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the PMSM metric and plotting schema")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    return parser


def _resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def _make_synthetic_eval_csv(tmp_dir: Path) -> Path:
    csv_path = tmp_dir / "_metric_plot_synthetic_eval.csv"
    fieldnames = [
        "controller",
        "env_id",
        "layout",
        "scenario",
        "step",
        "time_s",
        "cum_reward",
        "terminated",
        "truncated",
        "done",
        "done_reason",
        "i_d_phys",
        "i_q_phys",
        "ref_i_d_phys",
        "ref_i_q_phys",
        "act_u_d",
        "act_u_q",
        "u_d",
        "u_q",
        "active_Umax",
        "action_saturated",
        "speed_rpm",
        "load_torque_nm",
    ]
    rows = [
        [ "pi", "Custom-PMSM-Current-v0", "custom_dq", "synthetic_nominal", 0, 0.0, 0.0, 0, 0, 0, "", 0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.0, 0.0, 100.0, 0, 1000.0, 1.0 ],
        [ "pi", "Custom-PMSM-Current-v0", "custom_dq", "synthetic_nominal", 1, 0.1, 0.9, 0, 0, 0, "", -0.8, 5.5, -3.0, 10.0, -0.40, 0.80, -40.0, 80.0, 100.0, 0, 1000.0, 1.0 ],
        [ "pi", "Custom-PMSM-Current-v0", "custom_dq", "synthetic_nominal", 2, 0.2, 1.8, 0, 0, 0, "", -2.4, 8.9, -3.0, 10.0, -0.55, 0.92, -55.0, 92.0, 100.0, 0, 1000.0, 1.0 ],
        [ "pi", "Custom-PMSM-Current-v0", "custom_dq", "synthetic_nominal", 3, 0.3, 2.7, 0, 0, 0, "", -3.25, 10.6, -3.0, 10.0, 1.00, 1.00, 70.710678, 70.710678, 100.0, 1, 1000.0, 1.0 ],
        [ "pi", "Custom-PMSM-Current-v0", "custom_dq", "synthetic_nominal", 4, 0.4, 3.6, 0, 0, 0, "", -3.02, 10.1, -3.0, 10.0, -0.05, 0.10, -5.0, 10.0, 100.0, 0, 1000.0, 1.0 ],
        [ "pi", "Custom-PMSM-Current-v0", "custom_dq", "synthetic_nominal", 5, 0.5, 4.5, 0, 1, 1, "episode_limit", -3.0, 10.0, -3.0, 10.0, 0.00, 0.00, 0.0, 0.0, 100.0, 0, 1000.0, 1.0 ],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    return csv_path


def _print_pipeline_map() -> None:
    print("pipeline_code_paths:")
    for name, path in PIPELINE_PATHS.items():
        print(f"- {name}: {path}")
    print("notes:")
    print(f"- episode_trace_debug_helper: {PIPELINE_PATHS['failure_case_helper']}")
    print("- paper_figures.py is the canonical paper-style figure toolkit; plot_failure_case.py remains a small wrapper helper.")


def _verify_generators(figures: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for figure_type, config in figures.items():
        generator = str(config.get("generator", "")).strip()
        if not generator:
            failures.append(f"{figure_type}: missing generator")
            continue
        path_text, _, command = generator.partition("::")
        generator_path = _resolve_project_path(path_text)
        if not generator_path.exists():
            failures.append(f"{figure_type}: generator path missing -> {generator_path}")
            continue
        if command:
            file_text = generator_path.read_text(encoding="utf-8")
            if command not in file_text:
                failures.append(f"{figure_type}: generator command '{command}' not found in {generator_path}")
    return failures


def main() -> None:
    args = build_arg_parser().parse_args()
    schema = load_yaml(args.schema)
    metadata = dict(schema.get("canonical_metadata", {}))
    metrics = dict(schema.get("canonical_metrics", {}))
    figures = dict(schema.get("standard_figures", {}))
    compatibility_aliases = dict(schema.get("compatibility_aliases", {}))
    contract_notes = dict(schema.get("contract_notes", {}))

    if not metadata:
        raise ValueError(f"No canonical_metadata found in {args.schema}")
    if not metrics:
        raise ValueError(f"No canonical_metrics found in {args.schema}")
    if not figures:
        raise ValueError(f"No standard_figures found in {args.schema}")
    if "scenario" not in metadata:
        raise ValueError("canonical_metadata must include 'scenario'")
    if "paper_figure_data_rule" not in contract_notes:
        raise ValueError("contract_notes must include 'paper_figure_data_rule'")
    export_pipeline_text = str(contract_notes.get("canonical_export_pipeline", "")).strip()
    if export_pipeline_text != "scripts/core/evaluate.py":
        raise ValueError("canonical_export_pipeline must point to scripts/core/evaluate.py")

    script_failures = [str(path) for path in PIPELINE_PATHS.values() if not path.exists()]
    figure_failures = _verify_generators(figures)

    temp_root = ROOT / "outputs" / "diagnostics"
    temp_root.mkdir(parents=True, exist_ok=True)
    synthetic_csv = _make_synthetic_eval_csv(temp_root)
    try:
        single_summary = summarize_eval_csv(synthetic_csv)
        batch_summary = summarize_one(str(synthetic_csv), scenario_override="synthetic_nominal")
    finally:
        try:
            synthetic_csv.unlink(missing_ok=True)
        except OSError:
            pass

    canonical_field_names = list(metadata.keys()) + list(metrics.keys())
    single_missing = [name for name in canonical_field_names if name not in single_summary]
    batch_missing = [name for name in canonical_field_names if name not in batch_summary]
    alias_targets_missing = sorted({
        target_name
        for target_name in compatibility_aliases.values()
        if target_name not in canonical_field_names and target_name != "done"
    })
    paper_rule_ok = (
        "scripts/core/evaluate.py" in str(contract_notes.get("paper_figure_data_rule", ""))
        and "legacy csv" in str(contract_notes.get("paper_figure_data_rule", "")).lower()
    )

    print(f"schema_path: {args.schema}")
    _print_pipeline_map()
    print("contract_notes:")
    for key, value in contract_notes.items():
        print(f"- {key}: {value}")
    print("canonical_metadata:")
    for field_name, config in metadata.items():
        print(
            f"- {field_name}: "
            f"{config.get('meaning', '')} "
            f"(objective={config.get('objective', 'unspecified')})"
        )
    print("canonical_metrics:")
    for metric_name, config in metrics.items():
        print(
            f"- {metric_name}: "
            f"{config.get('meaning', '')} "
            f"(objective={config.get('objective', 'unspecified')})"
        )
    print("compatibility_aliases:")
    for alias_name, canonical_name in compatibility_aliases.items():
        print(f"- {alias_name} -> {canonical_name}")
    print("standard_figures:")
    for figure_type, config in figures.items():
        print(f"- {figure_type}:")
        print(f"  generator: {config.get('generator', '')}")
        print(f"  input_schema: {json.dumps(config.get('input_schema', {}), ensure_ascii=False)}")
        print(f"  default_output: {config.get('default_output', '')}")
    print("metric_output_checks:")
    print(f"- synthetic_single_run_canonical_fields_present: {len(single_missing) == 0}")
    if single_missing:
        print(f"  missing: {single_missing}")
    print(f"- synthetic_batch_summary_canonical_fields_present: {len(batch_missing) == 0}")
    if batch_missing:
        print(f"  missing: {batch_missing}")
    print(f"- compatibility_alias_targets_defined: {len(alias_targets_missing) == 0}")
    if alias_targets_missing:
        print(f"  missing: {alias_targets_missing}")
    print(f"- paper_figure_data_rule_present: {paper_rule_ok}")
    print("script_checks:")
    print(f"- required_pipeline_scripts_exist: {len(script_failures) == 0}")
    if script_failures:
        print(f"  missing: {script_failures}")
    print(f"- figure_generators_resolved: {len(figure_failures) == 0}")
    if figure_failures:
        print(f"  missing: {figure_failures}")
    print("sample_metric_keys:")
    print(f"- single_run_summary_keys: {sorted(single_summary.keys())}")
    print(f"- batch_summary_keys: {sorted(batch_summary.keys())}")
    all_checks_passed = not (
        single_missing
        or batch_missing
        or alias_targets_missing
        or script_failures
        or figure_failures
        or not paper_rule_ok
    )
    print(f"metric_plot_checks_passed: {all_checks_passed}")


if __name__ == "__main__":
    main()

"""Plot one evaluation episode trace with layout-aware PMSM support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from envs.obs_parser import required_eval_trace_columns
from utils.run_layout import infer_run_dir_from_path


SUPPORTED_TASKS = ("pmsm_cc",)
CUSTOM_TRACE_COLUMNS = ("i_d", "ref_i_d", "i_q", "ref_i_q", "act_u_d", "act_u_q")
LEGACY_TRACE_COLUMNS = required_eval_trace_columns()
OPTIONAL_REWARD_COLUMNS = (
    "reward",
    "cum_reward",
    "reward_limit_penalty",
    "reward_action_smoothness",
    "reward_delta_action",
)


@dataclass(slots=True)
class LoadedTrace:
    """Numerical episode trace plus the metadata needed for plotting."""

    layout: str
    x: np.ndarray
    data: dict[str, np.ndarray]
    fieldnames: list[str]
    done_step: int | None
    done_reason: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one evaluation episode trace from CSV")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--task", default="pmsm_cc", choices=SUPPORTED_TASKS)
    parser.add_argument("--output", type=Path, default=None, help="Optional output image path")
    return parser


def _default_output_path(csv_path: Path) -> Path:
    run_dir = infer_run_dir_from_path(csv_path)
    if run_dir is not None:
        return run_dir / "figures" / f"{csv_path.stem}_trace.png"
    return csv_path.with_name(f"{csv_path.stem}_trace.png")


def _read_numeric_column(rows: list[dict[str, str]], column_name: str) -> np.ndarray:
    values: list[float] = []
    for row_idx, row in enumerate(rows):
        raw_value = row.get(column_name, "")
        if raw_value in ("", None):
            raise ValueError(f"Column '{column_name}' is empty at row {row_idx}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Column '{column_name}' contains a non-numeric value at row {row_idx}: {raw_value!r}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Column '{column_name}' contains a non-finite value at row {row_idx}")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _first_nonempty_value(rows: list[dict[str, str]], column_name: str) -> str:
    for row in rows:
        value = str(row.get(column_name, "")).strip()
        if value:
            return value
    return ""


def _detect_trace_layout(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    available = set(fieldnames)
    explicit_layout = _first_nonempty_value(rows, "layout").lower()
    has_custom_columns = all(name in available for name in CUSTOM_TRACE_COLUMNS)
    has_legacy_columns = all(name in available for name in LEGACY_TRACE_COLUMNS)

    if explicit_layout == "custom_dq":
        missing = [name for name in CUSTOM_TRACE_COLUMNS if name not in available]
        if missing:
            raise KeyError(
                "CSV declares layout=custom_dq but is missing required dq trace columns: "
                f"{missing}"
            )
        return "custom_dq"

    if has_custom_columns:
        return "custom_dq"

    if has_legacy_columns:
        return "legacy"

    raise KeyError(
        "Could not infer episode-trace layout from the CSV. "
        "Expected either custom dq columns "
        f"{list(CUSTOM_TRACE_COLUMNS)} or legacy columns {list(LEGACY_TRACE_COLUMNS)}. "
        f"Available columns: {fieldnames}"
    )


def _load_eval_trace(csv_path: Path) -> LoadedTrace:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV contains no data rows: {csv_path}")

    layout = _detect_trace_layout(fieldnames, rows)
    required_columns = CUSTOM_TRACE_COLUMNS if layout == "custom_dq" else LEGACY_TRACE_COLUMNS
    missing = [name for name in required_columns if name not in fieldnames]
    if missing:
        raise KeyError(f"CSV is missing required trace columns for layout '{layout}': {missing}")

    if "step" in fieldnames:
        x = _read_numeric_column(rows, "step")
    else:
        x = np.arange(len(rows), dtype=np.float32)

    data = {name: _read_numeric_column(rows, name) for name in required_columns}
    for name in OPTIONAL_REWARD_COLUMNS:
        if name in fieldnames:
            data[name] = _read_numeric_column(rows, name)

    done_step: int | None = None
    done_reason = ""
    for row in rows:
        if row.get("done", "") == "1":
            step_value = row.get("step", "")
            done_step = int(float(step_value)) if step_value not in ("", None) else None
            done_reason = row.get("done_reason") or row.get("termination_reason") or ""
            break

    return LoadedTrace(
        layout=layout,
        x=x,
        data=data,
        fieldnames=fieldnames,
        done_step=done_step,
        done_reason=done_reason,
    )


def _add_done_marker(axes: list[plt.Axes], done_step: int | None) -> None:
    if done_step is None:
        return
    for ax in axes:
        ax.axvline(done_step, color="tab:red", linestyle=":", linewidth=1.2, alpha=0.8)


def _plot_custom_dq_trace(fig: plt.Figure, axes: np.ndarray, trace: LoadedTrace) -> list[plt.Axes]:
    x = trace.x
    data = trace.data

    axes[0].plot(x, data["i_d"], label="i_d", color="tab:blue", linewidth=1.8)
    axes[0].plot(x, data["ref_i_d"], label="ref_i_d", color="tab:orange", linestyle="--", linewidth=1.6)
    axes[0].set_ylabel("d current\n(norm.)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, data["i_q"], label="i_q", color="tab:green", linewidth=1.8)
    axes[1].plot(x, data["ref_i_q"], label="ref_i_q", color="tab:red", linestyle="--", linewidth=1.6)
    axes[1].set_ylabel("q current\n(norm.)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, data["act_u_d"], label="act_u_d", color="tab:purple", linewidth=1.6)
    axes[2].plot(x, data["act_u_q"], label="act_u_q", color="tab:brown", linewidth=1.6)
    axes[2].set_ylabel("dq action\n(norm.)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    reward_ax = axes[3]
    reward_ax.plot(x, data.get("reward", np.zeros_like(x)), label="reward", color="tab:blue", linewidth=1.7)
    reward_ax.set_ylabel("reward")
    reward_ax.set_xlabel("step")
    reward_ax.grid(True, alpha=0.3)

    if "reward_limit_penalty" in data and np.any(np.abs(data["reward_limit_penalty"]) > 1e-8):
        reward_ax.plot(
            x,
            data["reward_limit_penalty"],
            label="limit_penalty",
            color="tab:red",
            linestyle="--",
            linewidth=1.3,
        )

    for candidate, label, color in (
        ("reward_action_smoothness", "smoothness_penalty", "tab:purple"),
        ("reward_delta_action", "delta_action_penalty", "tab:gray"),
    ):
        if candidate in data and np.any(np.abs(data[candidate]) > 1e-8):
            reward_ax.plot(
                x,
                data[candidate],
                label=label,
                color=color,
                linestyle=":",
                linewidth=1.2,
            )
            break

    extra_axes: list[plt.Axes] = []
    if "cum_reward" in data:
        cum_ax = reward_ax.twinx()
        cum_ax.plot(x, data["cum_reward"], label="cum_reward", color="tab:green", linewidth=1.6)
        cum_ax.set_ylabel("cum_reward")
        extra_axes.append(cum_ax)

        left_handles, left_labels = reward_ax.get_legend_handles_labels()
        right_handles, right_labels = cum_ax.get_legend_handles_labels()
        reward_ax.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
    else:
        reward_ax.legend(loc="best")

    return [*list(axes), *extra_axes]


def _plot_legacy_trace(_fig: plt.Figure, axes: np.ndarray, trace: LoadedTrace) -> list[plt.Axes]:
    x = trace.x
    data = trace.data

    axes[0].plot(x, data["i_d"], label="i_d", color="tab:blue", linewidth=1.8)
    axes[0].plot(x, data["ref_i_d"], label="ref_i_d", color="tab:orange", linestyle="--", linewidth=1.6)
    axes[0].set_ylabel("d current")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, data["i_q"], label="i_q", color="tab:green", linewidth=1.8)
    axes[1].plot(x, data["ref_i_q"], label="ref_i_q", color="tab:red", linestyle="--", linewidth=1.6)
    axes[1].set_ylabel("q current")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, data["act_u_a"], label="act_u_a", linewidth=1.5)
    axes[2].plot(x, data["act_u_b"], label="act_u_b", linewidth=1.5)
    axes[2].plot(x, data["act_u_c"], label="act_u_c", linewidth=1.5)
    axes[2].set_ylabel("abc action")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(x, data["epsilon"], label="epsilon", color="tab:purple", linewidth=1.8)
    axes[3].set_ylabel("epsilon")
    axes[3].set_xlabel("step")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)
    return list(axes)


def create_episode_trace_plot(*, csv_path: Path, output_path: Path, task: str = "pmsm_cc", title_suffix: str = "") -> Path:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")

    trace = _load_eval_trace(csv_path)
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    if trace.layout == "custom_dq":
        plotted_axes = _plot_custom_dq_trace(fig, axes, trace)
    else:
        plotted_axes = _plot_legacy_trace(fig, axes, trace)

    _add_done_marker(plotted_axes, trace.done_step)

    title = f"{csv_path.name} | {trace.layout}"
    if trace.done_step is not None:
        if trace.done_reason:
            title = f"{title} | done_reason={trace.done_reason}"
        else:
            title = f"{title} | done_step={trace.done_step}"
    if title_suffix:
        title = f"{title} | {title_suffix}"
    axes[0].set_title(title)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = args.output or _default_output_path(args.csv_path)
    saved_path = create_episode_trace_plot(
        csv_path=args.csv_path,
        output_path=output_path,
        task=str(args.task),
    )
    print(f"saved plot: {saved_path}")


if __name__ == "__main__":
    main()

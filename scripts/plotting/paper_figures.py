from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLORS = {
    "PI": "#4C78A8",
    "RL": "#F58518",
    "DDPG": "#F58518",
    "TD3": "#F58518",
    "Ours": "#54A24B",
}
METHOD_MARKERS = {
    "PI": "o",
    "RL": "s",
    "DDPG": "s",
    "TD3": "s",
    "Ours": "D",
}
COLUMN_ALIASES = {
    "step": ["global_steps", "global_step", "step", "steps", "episode_steps"],
    "time": ["time", "time_s", "t"],
    "method": ["method", "controller", "agent", "algo", "algorithm", "variant"],
    "reward": ["reward", "episode_return", "return", "ep_return", "episode_reward"],
    "eval_return": ["latest_eval_return", "eval_return"],
    "actor_loss": ["avg_actor_loss", "actor_loss"],
    "critic_loss": ["avg_critic_loss", "critic_loss"],
    "scenario": ["scenario", "test_case", "case", "condition", "done_reason"],
    "speed": ["speed_marker", "speed_rpm", "omega_rpm", "omega_m", "omega"],
    "load": ["load_marker", "load_nm", "load_torque", "T_L", "torque"],
    "rmse": ["rmse", "rmse_all", "rmse_iq"],
    "control_energy": ["control_energy", "u_energy", "voltage_energy"],
    "action_smoothness": ["action_smoothness", "delta_action_energy", "action_rate_penalty"],
    "i_d": ["i_d", "id", "id_actual", "i_d_meas"],
    "i_q": ["i_q", "iq", "iq_actual", "i_q_meas"],
    "i_d_ref": ["ref_i_d", "i_d_ref", "id_ref", "i_d_star", "id_star"],
    "i_q_ref": ["ref_i_q", "i_q_ref", "iq_ref", "i_q_star", "iq_star"],
    "u_d": ["act_u_d", "u_d", "ud", "v_d", "u_d_cmd", "prev_u_d"],
    "u_q": ["act_u_q", "u_q", "uq", "v_q", "u_q_cmd", "prev_u_q"],
}


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "figure.constrained_layout.use": True,
        }
    )


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def savefig(fig: plt.Figure, path: str | Path) -> None:
    fig.savefig(ensure_parent(path), bbox_inches="tight")


def color_for_method(method: str) -> str:
    return METHOD_COLORS.get(str(method), "#333333")


def marker_for_method(method: str) -> str:
    return METHOD_MARKERS.get(str(method), "o")


def annotate_panels(axes) -> None:
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.text(-0.08, 1.02, f"({chr(97 + idx)})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def infer_method_from_path(path: str | Path) -> str:
    joined = "/".join(part.lower() for part in Path(path).parts)
    if "eval_pi" in joined or "/pi/" in joined:
        return "PI"
    if "smooth" in joined or "cdr" in joined or "ours" in joined or "full" in joined:
        return "Ours"
    if "td3" in joined:
        return "TD3"
    if "ddpg" in joined:
        return "DDPG"
    if "eval_rl" in joined or "/rl/" in joined:
        return "RL"
    return Path(path).parent.name or "Unknown"


def infer_seed_from_path(path: str | Path) -> int | None:
    match = re.search(r"seed[_-]?(\d+)", str(path).lower())
    return int(match.group(1)) if match else None


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    lower_map = {str(col).lower(): str(col) for col in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def resolve_column(df: pd.DataFrame, canonical_name: str) -> str:
    col = first_existing(df, COLUMN_ALIASES.get(canonical_name, [canonical_name]))
    if col is None:
        raise KeyError(f"Could not resolve '{canonical_name}'. Available columns: {list(df.columns)}")
    return col


def maybe_resolve_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    return first_existing(df, COLUMN_ALIASES.get(canonical_name, [canonical_name]))


def add_metadata(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    out = df.copy()
    method_col = maybe_resolve_column(out, "method")
    if method_col is None:
        out["method"] = infer_method_from_path(path)
    else:
        value = str(out[method_col].iloc[0]).strip().lower()
        if value == "pi":
            out["method"] = "PI"
        elif value == "rl":
            out["method"] = infer_method_from_path(path)
        else:
            out["method"] = out[method_col]
    if maybe_resolve_column(out, "seed") is None:
        seed = infer_seed_from_path(path)
        out["seed"] = -1 if seed is None else seed
    return out


def load_with_metadata(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = add_metadata(pd.read_csv(path), path)
        df["_source"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("No CSV files loaded.")
    return pd.concat(frames, ignore_index=True)


def resolve_x_axis(df: pd.DataFrame) -> tuple[pd.Series, str]:
    time_col = maybe_resolve_column(df, "time")
    if time_col is not None:
        return df[time_col], "Time [s]"
    step_col = maybe_resolve_column(df, "step")
    if step_col is not None:
        return df[step_col], "Training step" if step_col.startswith("global") else "Step"
    return pd.Series(np.arange(len(df))), "Index"


def aggregate_metric(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    return (
        df.groupby(["method", x_col])[y_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": f"{y_col}_mean", "std": f"{y_col}_std"})
    )


def maybe_smooth(series: pd.Series, rolling: int) -> pd.Series:
    if rolling <= 1:
        return series
    return series.rolling(window=rolling, min_periods=1, center=True).mean()


def plot_training_dashboard(paths: list[str], output: str, rolling: int) -> None:
    df = load_with_metadata(paths)
    x_col = resolve_column(df, "step")
    return_col = resolve_column(df, "reward")
    eval_col = maybe_resolve_column(df, "eval_return")
    actor_col = maybe_resolve_column(df, "actor_loss")
    critic_col = maybe_resolve_column(df, "critic_loss")
    panels = [(return_col, "Episode return")]
    if eval_col is not None:
        panels.append((eval_col, "Periodic eval return"))
    if actor_col is not None:
        panels.append((actor_col, "Actor loss"))
    if critic_col is not None:
        panels.append((critic_col, "Critic loss"))

    fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 3.1 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (metric_col, ylabel) in zip(axes, panels):
        agg = aggregate_metric(df, x_col, metric_col)
        for method, sub in agg.groupby("method"):
            sub = sub.sort_values(x_col)
            x = sub[x_col]
            mean = maybe_smooth(sub[f"{metric_col}_mean"], rolling)
            std = maybe_smooth(sub[f"{metric_col}_std"].fillna(0.0), rolling)
            color = color_for_method(str(method))
            ax.plot(x, mean, label=str(method), color=color)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, ncol=min(3, max(1, agg["method"].nunique())))
    axes[-1].set_xlabel("Training step")
    fig.suptitle("Training convergence with uncertainty bands", y=1.01, fontsize=12)
    annotate_panels(axes)
    savefig(fig, output)
    plt.close(fig)


def plot_nominal_waveforms(csv_path: str, output: str, title: str) -> None:
    df = pd.read_csv(csv_path)
    x, x_label = resolve_x_axis(df)
    i_d = df[resolve_column(df, "i_d")].to_numpy(dtype=float)
    i_q = df[resolve_column(df, "i_q")].to_numpy(dtype=float)
    i_d_ref = df[resolve_column(df, "i_d_ref")].to_numpy(dtype=float)
    i_q_ref = df[resolve_column(df, "i_q_ref")].to_numpy(dtype=float)
    u_d = df[resolve_column(df, "u_d")].to_numpy(dtype=float)
    u_q = df[resolve_column(df, "u_q")].to_numpy(dtype=float)
    e_d = i_d_ref - i_d
    e_q = i_q_ref - i_q
    du_norm = np.sqrt(np.diff(u_d, prepend=u_d[0]) ** 2 + np.diff(u_q, prepend=u_q[0]) ** 2)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    axes[0].plot(x, i_d_ref, label=r"$i_d^*$", linestyle="--")
    axes[0].plot(x, i_d, label=r"$i_d$")
    axes[0].plot(x, i_q_ref, label=r"$i_q^*$", linestyle="--")
    axes[0].plot(x, i_q, label=r"$i_q$")
    axes[0].set_ylabel("Current")
    axes[0].legend(frameon=False, ncol=4, loc="upper right")

    axes[1].plot(x, e_d, label=r"$e_d$")
    axes[1].plot(x, e_q, label=r"$e_q$")
    axes[1].fill_between(x, 0.0, e_q, alpha=0.12)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_ylabel("Error")
    axes[1].legend(frameon=False)

    axes[2].plot(x, u_d, label=r"$u_d$")
    axes[2].plot(x, u_q, label=r"$u_q$")
    ax2 = axes[2].twinx()
    ax2.plot(x, du_norm, label=r"$\|\Delta u\|$", linestyle=":", alpha=0.9)
    axes[2].set_ylabel("Action")
    ax2.set_ylabel(r"$\|\Delta u\|$")
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=3, loc="upper right")
    axes[2].set_xlabel(x_label)

    fig.suptitle(title, y=1.01, fontsize=12)
    annotate_panels(axes)
    savefig(fig, output)
    plt.close(fig)


def add_event_markers(ax: plt.Axes, events: list[float], labels: list[str]) -> None:
    ymin, ymax = ax.get_ylim()
    for idx, event in enumerate(events):
        ax.axvline(event, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
        if idx < len(labels):
            ax.text(event, ymax, labels[idx], rotation=90, va="top", ha="right", fontsize=8)


def plot_event_response(csv_path: str, output: str, title: str, events: list[float], labels: list[str]) -> None:
    df = pd.read_csv(csv_path)
    x, x_label = resolve_x_axis(df)
    i_q = df[resolve_column(df, "i_q")]
    i_q_ref = df[resolve_column(df, "i_q_ref")]
    u_q = df[resolve_column(df, "u_q")]
    reward_col = maybe_resolve_column(df, "reward")
    speed_col = maybe_resolve_column(df, "speed")
    load_col = maybe_resolve_column(df, "load")

    fig, axes = plt.subplots(4, 1, figsize=(9, 8.2), sharex=True)
    axes[0].plot(x, i_q_ref, linestyle="--", label=r"$i_q^*$")
    axes[0].plot(x, i_q, label=r"$i_q$")
    axes[0].set_ylabel("Current")
    axes[0].legend(frameon=False)

    error = i_q_ref - i_q
    axes[1].plot(x, error, label=r"$e_q$")
    axes[1].fill_between(x, 0.0, error, alpha=0.15)
    axes[1].set_ylabel("Error")
    axes[1].legend(frameon=False)

    if speed_col is not None:
        axes[2].plot(x, df[speed_col], label=str(speed_col))
    if load_col is not None:
        axes[2].plot(x, df[load_col], label=str(load_col))
    if speed_col is None and load_col is None:
        axes[2].plot(x, df[resolve_column(df, "i_d")], label="i_d")
        axes[2].plot(x, df[resolve_column(df, "i_d_ref")], label="ref_i_d", linestyle="--")
    axes[2].set_ylabel("Disturbance / state")
    axes[2].legend(frameon=False)

    axes[3].plot(x, u_q, label=r"$u_q$")
    if reward_col is not None:
        ax2 = axes[3].twinx()
        ax2.plot(x, df[reward_col], label="reward", linestyle=":", alpha=0.8)
        lines1, labels1 = axes[3].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[3].legend(lines1 + lines2, labels1 + labels2, frameon=False)
    else:
        axes[3].legend(frameon=False)
    axes[3].set_ylabel("Action")
    axes[3].set_xlabel(x_label)

    for ax in axes:
        add_event_markers(ax, events, labels)
    annotate_panels(axes)
    fig.suptitle(title, y=1.01, fontsize=12)
    savefig(fig, output)
    plt.close(fig)


def plot_heatmap(csv_path: str, output: str, metric: str, x_name: str, y_name: str, methods: list[str] | None, title: str) -> None:
    df = pd.read_csv(csv_path)
    method_col = resolve_column(df, "method")
    x_col = resolve_column(df, x_name)
    y_col = resolve_column(df, y_name)
    metric_col = resolve_column(df, metric) if metric in COLUMN_ALIASES else (metric if metric in df.columns else resolve_column(df, metric))
    ordered_methods = methods or list(dict.fromkeys(df[method_col].tolist()))
    fig, axes = plt.subplots(1, len(ordered_methods), figsize=(4.2 * len(ordered_methods), 4.2), sharex=True, sharey=True)
    if len(ordered_methods) == 1:
        axes = [axes]
    vmin = float(df[metric_col].min())
    vmax = float(df[metric_col].max())
    last_im = None
    for ax, method in zip(axes, ordered_methods):
        sub = df[df[method_col] == method].copy()
        pivot = sub.pivot_table(index=y_col, columns=x_col, values=metric_col, aggfunc="mean")
        pivot = pivot.sort_index().sort_index(axis=1)
        xv = pivot.columns.to_numpy(dtype=float)
        yv = pivot.index.to_numpy(dtype=float)
        z = pivot.to_numpy(dtype=float)
        last_im = ax.imshow(z, origin="lower", aspect="auto", extent=[xv.min(), xv.max(), yv.min(), yv.max()], vmin=vmin, vmax=vmax, interpolation="bicubic")
        contour = ax.contour(xv, yv, z, levels=5, colors="white", linewidths=0.8, alpha=0.75)
        ax.clabel(contour, inline=True, fontsize=7, fmt="%.2f")
        ax.set_title(str(method))
        ax.set_xlabel(x_col)
    axes[0].set_ylabel(y_col)
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(metric_col)
    annotate_panels(axes)
    fig.suptitle(title, y=1.04, fontsize=12)
    savefig(fig, output)
    plt.close(fig)


def plot_pareto(csv_path: str, output: str, x_name: str, y_name: str, annotate: bool, title: str) -> None:
    df = pd.read_csv(csv_path)
    method_col = resolve_column(df, "method")
    x_col = resolve_column(df, x_name) if x_name in COLUMN_ALIASES else (x_name if x_name in df.columns else resolve_column(df, x_name))
    y_col = resolve_column(df, y_name) if y_name in COLUMN_ALIASES else (y_name if y_name in df.columns else resolve_column(df, y_name))
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    centers = []
    for method, sub in df.groupby(method_col):
        color = color_for_method(str(method))
        marker = marker_for_method(str(method))
        ax.scatter(sub[x_col], sub[y_col], label=str(method), alpha=0.42, color=color, marker=marker, s=36, edgecolor="white", linewidth=0.5)
        cx = float(sub[x_col].mean())
        cy = float(sub[y_col].mean())
        centers.append((str(method), cx, cy))
        ax.scatter(cx, cy, color=color, marker=marker, s=140, edgecolor="black", linewidth=0.8)
        if annotate:
            ax.annotate(str(method), (cx, cy), textcoords="offset points", xytext=(8, 6), fontsize=9)
    if centers:
        best_x = min(center[1] for center in centers)
        best_y = min(center[2] for center in centers)
        ax.axvspan(ax.get_xlim()[0], best_x, alpha=0.04, color="#54A24B")
        ax.axhspan(ax.get_ylim()[0], best_y, alpha=0.04, color="#54A24B")
        ax.text(0.02, 0.98, "Preferred region", transform=ax.transAxes, va="top", ha="left", fontsize=9)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(frameon=False)
    ax.set_title(title)
    savefig(fig, output)
    plt.close(fig)


def plot_robustness(csv_path: str, output: str, metric: str, scenario: str | None, title: str) -> None:
    df = pd.read_csv(csv_path)
    method_col = resolve_column(df, "method")
    metric_col = resolve_column(df, metric) if metric in COLUMN_ALIASES else (metric if metric in df.columns else resolve_column(df, metric))
    if scenario is not None:
        scenario_col = resolve_column(df, "scenario")
        df = df[df[scenario_col] == scenario].copy()
    methods = list(dict.fromkeys(df[method_col].tolist()))
    data = [df[df[method_col] == method][metric_col].dropna().to_numpy() for method in methods]
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    positions = np.arange(1, len(methods) + 1)
    violins = ax.violinplot(data, positions=positions, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
    for body, method in zip(violins["bodies"], methods):
        body.set_facecolor(color_for_method(str(method)))
        body.set_edgecolor("none")
        body.set_alpha(0.22)
    box = ax.boxplot(data, positions=positions, widths=0.33, patch_artist=True, medianprops={"color": "black"})
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(color_for_method(str(method)))
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")
    rng = np.random.default_rng(42)
    for x, method, values in zip(positions, methods, data):
        jitter = rng.normal(0.0, 0.04, size=len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=18, alpha=0.55, color=color_for_method(str(method)), edgecolor="white", linewidth=0.35)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(method) for method in methods], rotation=10)
    ax.set_ylabel(metric_col)
    suffix = f" | {scenario}" if scenario else ""
    ax.set_title(f"{title}{suffix}")
    savefig(fig, output)
    plt.close(fig)


def plot_ablation(csv_path: str, output: str, lower_better: bool, title: str) -> None:
    df = pd.read_csv(csv_path)
    if not {"stage", "metric"}.issubset(df.columns):
        raise KeyError("Ablation CSV must contain columns: stage, metric")
    stages = df["stage"].astype(str).tolist()
    values = df["metric"].astype(float).tolist()
    deltas = [values[0]] + [values[i] - values[i - 1] for i in range(1, len(values))]
    if lower_better:
        colors = ["#9E9E9E"] + ["#54A24B" if delta <= 0 else "#E45756" for delta in deltas[1:]]
    else:
        colors = ["#9E9E9E"] + ["#54A24B" if delta >= 0 else "#E45756" for delta in deltas[1:]]
    starts = [0.0] + values[:-1]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for idx, (stage, start, delta, final_value, color) in enumerate(zip(stages, starts, deltas, values, colors)):
        bottom = 0 if idx == 0 else min(start, final_value)
        height = abs(delta) if idx != 0 else final_value
        ax.bar(stage, height, bottom=bottom, color=color, edgecolor="black", linewidth=0.6)
        y_text = final_value + 0.02 * (max(values) - min(values) + 1e-6)
        ax.text(idx, y_text, f"{final_value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.plot(np.arange(len(values)), values, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylabel("Metric")
    ax.set_title(title)
    savefig(fig, output)
    plt.close(fig)


def main() -> None:
    apply_style()
    parser = argparse.ArgumentParser(description="Paper-style plotting toolkit for PMSM RL experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("training_dashboard", help="Multi-seed training curves with uncertainty bands")
    p_train.add_argument("inputs", nargs="+", help="One or more train_log.csv files")
    p_train.add_argument("--output", required=True)
    p_train.add_argument("--rolling", type=int, default=1)

    p_nom = subparsers.add_parser("nominal_waveforms", help="Three-layer nominal waveform figure")
    p_nom.add_argument("--csv", required=True)
    p_nom.add_argument("--output", required=True)
    p_nom.add_argument("--title", default="Nominal tracking waveform")

    p_event = subparsers.add_parser("event_response", help="Event-aligned disturbance response figure")
    p_event.add_argument("--csv", required=True)
    p_event.add_argument("--output", required=True)
    p_event.add_argument("--title", default="Event-aligned disturbance response")
    p_event.add_argument("--events", nargs="*", type=float, default=[])
    p_event.add_argument("--labels", nargs="*", default=[])

    p_heat = subparsers.add_parser("performance_heatmap", help="Speed-load performance heatmaps")
    p_heat.add_argument("--csv", required=True)
    p_heat.add_argument("--metric", default="rmse")
    p_heat.add_argument("--x", default="speed")
    p_heat.add_argument("--y", default="load")
    p_heat.add_argument("--output", required=True)
    p_heat.add_argument("--methods", nargs="*", default=None)
    p_heat.add_argument("--title", default="Performance map across operating conditions")

    p_pareto = subparsers.add_parser("pareto_front", help="Precision-cost Pareto view")
    p_pareto.add_argument("--csv", required=True)
    p_pareto.add_argument("--x", default="rmse")
    p_pareto.add_argument("--y", default="control_energy")
    p_pareto.add_argument("--annotate", action="store_true")
    p_pareto.add_argument("--output", required=True)
    p_pareto.add_argument("--title", default="Precision-cost Pareto view")

    p_robust = subparsers.add_parser("robustness_distribution", help="Violin + box robustness plot")
    p_robust.add_argument("--csv", required=True)
    p_robust.add_argument("--metric", default="rmse")
    p_robust.add_argument("--scenario", default=None)
    p_robust.add_argument("--output", required=True)
    p_robust.add_argument("--title", default="Robustness distribution across methods")

    p_ablate = subparsers.add_parser("ablation_waterfall", help="Incremental ablation waterfall chart")
    p_ablate.add_argument("--csv", required=True)
    p_ablate.add_argument("--output", required=True)
    p_ablate.add_argument("--lower-better", action="store_true")
    p_ablate.add_argument("--title", default="Incremental gains from design modules")

    args = parser.parse_args()
    if args.command == "training_dashboard":
        plot_training_dashboard(args.inputs, args.output, args.rolling)
    elif args.command == "nominal_waveforms":
        plot_nominal_waveforms(args.csv, args.output, args.title)
    elif args.command == "event_response":
        plot_event_response(args.csv, args.output, args.title, args.events, args.labels)
    elif args.command == "performance_heatmap":
        plot_heatmap(args.csv, args.output, args.metric, args.x, args.y, args.methods, args.title)
    elif args.command == "pareto_front":
        plot_pareto(args.csv, args.output, args.x, args.y, args.annotate, args.title)
    elif args.command == "robustness_distribution":
        plot_robustness(args.csv, args.output, args.metric, args.scenario, args.title)
    elif args.command == "ablation_waterfall":
        plot_ablation(args.csv, args.output, args.lower_better, args.title)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(f"saved figure: {args.output}")


if __name__ == "__main__":
    main()

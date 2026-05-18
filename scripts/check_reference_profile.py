"""Lightweight checker for custom PMSM step-reference profiles."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import numpy as np

from envs.make_env import make_eval_env
from utils.config import parse_env_config
from utils.experiment_factory import make_env_build_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check custom PMSM reference profile behavior")
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/env/pmsm_cc_train_external_step.yaml"),
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    return parser


def _feedforward_zero_current_action(env: object, info: dict[str, object]) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    params = getattr(base_env, "motor_params")
    omega_m = float(info["omega_m"])
    u_q = float(params.p) * omega_m * float(params.psi_f)
    u_d = 0.0
    action = np.asarray([u_d, u_q], dtype=np.float64) / max(float(params.Umax), 1e-6)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _assert_close_pair(actual: tuple[float, float], expected: tuple[float, float], *, label: str) -> None:
    if not np.allclose(actual, expected, atol=1e-6, rtol=0.0):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def check_seed(env_config_path: Path, seed: int) -> tuple[float, float]:
    env_cfg = parse_env_config(env_config_path)
    env = make_eval_env(make_env_build_config(env_cfg, seed_override=seed))
    obs, info = env.reset(seed=seed)
    if not np.isfinite(obs).all():
        raise AssertionError(f"seed={seed}: reset observation is not finite")

    reference_profile = str(info["reference_profile"])
    speed_mode = str(info["speed_mode"])
    step_index = int(info["reference_step_step"])
    initial_ref = (float(info["ref_i_d"]), float(info["ref_i_q"]))
    _assert_close_pair(initial_ref, (0.0, 0.0), label=f"seed={seed} reset reference")

    omega_values = [float(info["omega_m"])]
    before_info = info
    action = _feedforward_zero_current_action(env, info)
    for _ in range(max(step_index - 1, 0)):
        obs, _reward, terminated, truncated, before_info = env.step(action)
        if not np.isfinite(obs).all():
            raise AssertionError(f"seed={seed}: non-finite observation before step")
        if bool(terminated or truncated):
            raise AssertionError(f"seed={seed}: episode ended before reference step")
        omega_values.append(float(before_info["omega_m"]))
        action = _feedforward_zero_current_action(env, before_info)

    before_ref = (float(before_info["ref_i_d"]), float(before_info["ref_i_q"]))
    _assert_close_pair(before_ref, (0.0, 0.0), label=f"seed={seed} pre-step reference")

    after_info = before_info
    for _ in range(6):
        obs, _reward, terminated, truncated, after_info = env.step(action)
        if not np.isfinite(obs).all():
            raise AssertionError(f"seed={seed}: non-finite observation after step")
        if bool(terminated or truncated):
            raise AssertionError(f"seed={seed}: episode ended during post-step check")
        omega_values.append(float(after_info["omega_m"]))
        action = _feedforward_zero_current_action(env, after_info)

    final_target = (float(after_info["ref_i_d_final"]), float(after_info["ref_i_q_final"]))
    after_ref = (float(after_info["ref_i_d"]), float(after_info["ref_i_q"]))
    _assert_close_pair(after_ref, final_target, label=f"seed={seed} post-step reference")

    omega_deviation = max(abs(value - omega_values[0]) for value in omega_values)
    if omega_deviation > 1e-9:
        raise AssertionError(f"seed={seed}: omega_m drifted by {omega_deviation}")

    print(
        f"seed={seed} reference_profile={reference_profile} speed_mode={speed_mode} "
        f"step_index={step_index} initial_ref={initial_ref} pre_step_ref={before_ref} "
        f"final_ref=({final_target[0]:.6f}, {final_target[1]:.6f}) "
        f"post_step_ref=({after_ref[0]:.6f}, {after_ref[1]:.6f}) "
        f"omega_m={omega_values[0]:.6f} omega_max_deviation={omega_deviation:.3e}"
    )
    env.close()
    return final_target


def main() -> None:
    args = build_arg_parser().parse_args()
    targets = [check_seed(args.env_config, int(seed)) for seed in args.seeds]
    unique_targets = {tuple(round(value, 6) for value in target) for target in targets}
    if len(unique_targets) <= 1 and len(targets) > 1:
        raise AssertionError("final step targets did not vary across seeds")
    print(f"unique_final_targets={len(unique_targets)}")
    print("reference profile check passed")


if __name__ == "__main__":
    main()

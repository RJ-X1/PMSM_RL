"""Check the configured training and test condition sets for PMSM experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import math
from typing import Any

from utils.config import (
    apply_train_domain_randomization_override,
    load_yaml,
    parse_env_config,
    parse_eval_config,
    parse_train_config,
)
from utils.experiment_factory import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_EVAL_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
)


RPM_TO_RAD_PER_SEC = 2.0 * math.pi / 60.0
RAD_PER_SEC_TO_RPM = 60.0 / (2.0 * math.pi)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check PMSM training/test condition definitions")
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG_PATH)
    return parser


def _ensure_finite_scalar(value: Any, *, label: str) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric, got {type(value)}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite, got {value}")


def _ensure_finite_pair(pair: tuple[float, float], *, label: str) -> None:
    _ensure_finite_scalar(pair[0], label=f"{label}[0]")
    _ensure_finite_scalar(pair[1], label=f"{label}[1]")
    if float(pair[0]) > float(pair[1]):
        raise ValueError(f"{label} must be ordered low<=high, got {pair}")


def _validate_numeric_tree(value: Any, *, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_numeric_tree(child, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_numeric_tree(child, label=f"{label}[{index}]")
        return
    if isinstance(value, (int, float)):
        _ensure_finite_scalar(value, label=label)


def _resolve_test_conditions_path(path_value: str, *, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return root / path


def _validate_training_conditions(env_cfg: Any) -> dict[str, Any]:
    ref_cfg = env_cfg.reference
    env_settings = env_cfg.environment
    dr_cfg = env_cfg.domain_randomization

    _ensure_finite_pair(ref_cfg.ref_i_d_range, label="training.ref_i_d_range")
    _ensure_finite_pair(ref_cfg.ref_i_q_range, label="training.ref_i_q_range")
    _ensure_finite_pair(env_settings.init_omega_m_range, label="training.init_omega_m_range")
    _ensure_finite_pair(env_settings.load_torque_range, label="training.load_torque_range")

    if not tuple(ref_cfg.profile_types):
        raise ValueError("training.reference.profile_types must not be empty")

    dr_mapping = asdict(dr_cfg)
    _validate_numeric_tree(dr_mapping, label="training.domain_randomization")

    return {
        "ref_i_d_range_a": [float(ref_cfg.ref_i_d_range[0]), float(ref_cfg.ref_i_d_range[1])],
        "ref_i_q_range_a": [float(ref_cfg.ref_i_q_range[0]), float(ref_cfg.ref_i_q_range[1])],
        "reference_profile_types": list(ref_cfg.profile_types),
        "speed_range_rpm": [
            round(float(env_settings.init_omega_m_range[0]) * RAD_PER_SEC_TO_RPM, 6),
            round(float(env_settings.init_omega_m_range[1]) * RAD_PER_SEC_TO_RPM, 6),
        ],
        "speed_range_rad_per_sec": [
            float(env_settings.init_omega_m_range[0]),
            float(env_settings.init_omega_m_range[1]),
        ],
        "load_torque_range_nm": [
            float(env_settings.load_torque_range[0]),
            float(env_settings.load_torque_range[1]),
        ],
        "domain_randomization": dr_mapping,
    }


def _validate_test_scenarios(raw: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = raw.get("test_scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("test_scenarios must be a non-empty list")

    seen_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise TypeError(f"test_scenarios[{index}] must be a mapping")

        scenario_id = str(scenario.get("id", "")).strip()
        scenario_name = str(scenario.get("name", "")).strip()
        description = str(scenario.get("description", "")).strip()
        reference_profile = str(scenario.get("reference_profile", "")).strip()
        if not scenario_id or not scenario_name or not description or not reference_profile:
            raise ValueError(f"Scenario {index} must define id, name, description, and reference_profile")
        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate test scenario id: {scenario_id}")
        seen_ids.add(scenario_id)

        _validate_numeric_tree(scenario, label=f"test_scenarios[{scenario_id}]")

        reference = scenario.get("reference")
        reference_ranges = scenario.get("reference_ranges")
        if reference_profile == "step" and not isinstance(reference, dict):
            raise ValueError(f"{scenario_id} uses a step reference but does not define a reference block")
        if "mixed" in reference_profile and not isinstance(reference_ranges, dict):
            raise ValueError(f"{scenario_id} uses a mixed reference but does not define reference_ranges")
        if reference is not None:
            for key in ("i_d_initial", "i_q_initial", "i_d_final", "i_q_final"):
                if key not in reference:
                    raise ValueError(f"{scenario_id}.reference missing required key '{key}'")
                _ensure_finite_scalar(reference[key], label=f"{scenario_id}.reference.{key}")
        if reference_ranges is not None:
            for key in ("i_d_range", "i_q_range"):
                if key not in reference_ranges:
                    raise ValueError(f"{scenario_id}.reference_ranges missing required key '{key}'")
                pair = tuple(reference_ranges[key])
                if len(pair) != 2:
                    raise ValueError(f"{scenario_id}.reference_ranges.{key} must have length 2")
                _ensure_finite_pair((float(pair[0]), float(pair[1])), label=f"{scenario_id}.reference_ranges.{key}")

        if "omega_m_rpm" not in scenario and "speed_rpm_profile" not in scenario:
            raise ValueError(f"{scenario_id} must define omega_m_rpm or speed_rpm_profile")
        if "load_torque_nm" not in scenario and "load_torque_nm_profile" not in scenario:
            raise ValueError(f"{scenario_id} must define load_torque_nm or load_torque_nm_profile")

        if "speed_rpm_profile" in scenario:
            profile = tuple(float(value) for value in scenario["speed_rpm_profile"])
            if len(profile) != 2:
                raise ValueError(f"{scenario_id}.speed_rpm_profile must have length 2")
        if "load_torque_nm_profile" in scenario:
            profile = tuple(float(value) for value in scenario["load_torque_nm_profile"])
            if len(profile) != 2:
                raise ValueError(f"{scenario_id}.load_torque_nm_profile must have length 2")
        if "parameter_scales" in scenario:
            for key, value in dict(scenario["parameter_scales"]).items():
                _ensure_finite_scalar(value, label=f"{scenario_id}.parameter_scales.{key}")
                if float(value) <= 0.0:
                    raise ValueError(f"{scenario_id}.parameter_scales.{key} must be positive")
        if "action_delay_steps" in scenario and int(scenario["action_delay_steps"]) < 0:
            raise ValueError(f"{scenario_id}.action_delay_steps must be >= 0")

        validated.append(dict(scenario))
    return validated


def main() -> None:
    args = build_arg_parser().parse_args()

    env_cfg = parse_env_config(args.env_config)
    train_cfg = parse_train_config(args.train_config)
    eval_cfg = parse_eval_config(args.eval_config)
    effective_env_cfg = apply_train_domain_randomization_override(env_cfg, train_cfg)

    training_conditions = _validate_training_conditions(effective_env_cfg)
    test_conditions_path = _resolve_test_conditions_path(
        eval_cfg.test_conditions_path,
        root=ROOT,
    )
    test_condition_data = load_yaml(test_conditions_path)
    test_scenarios = _validate_test_scenarios(test_condition_data)

    print(f"env_config: {args.env_config}")
    print(f"train_config: {args.train_config}")
    print(f"eval_config: {args.eval_config}")
    print(f"test_conditions_path: {test_conditions_path}")
    print("training_conditions:")
    print(json.dumps(training_conditions, indent=2, ensure_ascii=False))
    print("test_scenarios:")
    for scenario in test_scenarios:
        print(f"- {scenario['name']} ({scenario['id']}): {scenario['description']}")
        print(f"  reference_profile: {scenario['reference_profile']}")
        if "reference" in scenario:
            print(f"  reference: {json.dumps(scenario['reference'], ensure_ascii=False)}")
        if "reference_ranges" in scenario:
            print(f"  reference_ranges: {json.dumps(scenario['reference_ranges'], ensure_ascii=False)}")
        if "omega_m_rpm" in scenario:
            print(f"  omega_m_rpm: {float(scenario['omega_m_rpm']):.3f}")
        if "speed_rpm_profile" in scenario:
            print(f"  speed_rpm_profile: {json.dumps(scenario['speed_rpm_profile'])}")
        if "load_torque_nm" in scenario:
            print(f"  load_torque_nm: {float(scenario['load_torque_nm']):.3f}")
        if "load_torque_nm_profile" in scenario:
            print(f"  load_torque_nm_profile: {json.dumps(scenario['load_torque_nm_profile'])}")
        if "parameter_scales" in scenario:
            print(f"  parameter_scales: {json.dumps(scenario['parameter_scales'], ensure_ascii=False)}")
        if "observation_noise_std" in scenario:
            print(f"  observation_noise_std: {float(scenario['observation_noise_std']):.6f}")
        if "speed_noise_rpm" in scenario:
            print(f"  speed_noise_rpm: {float(scenario['speed_noise_rpm']):.3f}")
        if "action_delay_steps" in scenario:
            print(f"  action_delay_steps: {int(scenario['action_delay_steps'])}")
    print("condition_checks_passed: True")


if __name__ == "__main__":
    main()

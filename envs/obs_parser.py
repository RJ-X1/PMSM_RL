"""Observation parsing helpers for PMSM current-control experiments.

This module keeps the rest of the project from depending on hard-coded flat
observation indices. GEM observations are commonly a concatenation of state and
reference values. For PMSM environments the paper lists the default state part
as ``[omega, torque, ia, ib, ic, ua, ub, uc, usup, epsilon]`` and then appends
reference values for the tracked quantities. When direct introspection from the
installed GEM environment is possible, this module prefers that information.
Otherwise it falls back to conservative heuristics.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

TRACE_SIGNAL_NAMES = ("i_d", "ref_i_d", "i_q", "ref_i_q", "epsilon")
TRACE_ACTION_NAMES = ("act_u_a", "act_u_b", "act_u_c")
CUSTOM_OBSERVATION_NAMES = (
    "i_d",
    "i_q",
    "omega_m",
    "ref_i_d",
    "ref_i_q",
    "e_d",
    "e_q",
    "prev_u_d",
    "prev_u_q",
    "T_L",
)
CUSTOM_ACTION_NAMES = ("act_u_d", "act_u_q")

ABC_STATE_NAMES = [
    "omega",
    "torque",
    "i_a",
    "i_b",
    "i_c",
    "u_a",
    "u_b",
    "u_c",
    "u_sup",
    "epsilon",
]
DQ_STATE_NAMES = [
    "omega",
    "torque",
    "i_d",
    "i_q",
    "u_d",
    "u_q",
    "u_sup",
    "epsilon",
]


@dataclass(slots=True)
class ObservationSpec:
    """Semantic description of a flattened observation vector."""

    env_id: str
    layout: str
    state_names: list[str]
    reference_names: list[str]
    action_names: list[str]
    extra_names: list[str] = field(default_factory=list)

    @property
    def signal_names(self) -> list[str]:
        return [*self.state_names, *self.reference_names, *self.extra_names]

    @property
    def obs_dim(self) -> int:
        return len(self.signal_names)

    def to_mapping(self, flat_obs: Sequence[float] | np.ndarray) -> "OrderedDict[str, float]":
        """Convert a flat observation array into a named ordered mapping."""
        x = np.asarray(flat_obs, dtype=np.float32).reshape(-1)
        names = self.signal_names
        if x.size != len(names):
            raise ValueError(f"Observation size {x.size} does not match spec size {len(names)}")
        return OrderedDict((name, float(value)) for name, value in zip(names, x))


@dataclass(slots=True)
class EnvIntrospection:
    """Data read from a live environment if available."""

    state_names: list[str] | None = None
    reference_names: list[str] | None = None



def _to_name_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        names = [str(x) for x in value]
        return names or None
    return None



def _sanitize_name(name: str) -> str:
    return (
        str(name)
        .replace("omega_me", "omega")
        .replace("omega", "omega")
        .replace("i_sd", "i_d")
        .replace("i_sq", "i_q")
        .replace("u_sd", "u_d")
        .replace("u_sq", "u_q")
        .replace("i_sa", "i_a")
        .replace("i_sb", "i_b")
        .replace("i_sc", "i_c")
        .replace("u_sa", "u_a")
        .replace("u_sb", "u_b")
        .replace("u_sc", "u_c")
        .replace("usup", "u_sup")
    )



def _dedupe(names: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for raw in names:
        name = _sanitize_name(raw)
        if name not in counts:
            counts[name] = 0
            result.append(name)
            continue
        counts[name] += 1
        result.append(f"{name}_{counts[name]}")
    return result



def introspect_env(env: Any) -> EnvIntrospection:
    """Try to extract state/reference names from a live GEM environment.

    The exact attribute names vary across GEM versions, so this function probes a
    few common locations and gracefully returns ``None`` when unavailable.
    """

    obj = getattr(env, "unwrapped", env)
    physical_system = getattr(obj, "physical_system", None)
    reference_generator = getattr(obj, "reference_generator", None)

    state_names: list[str] | None = None
    for candidate in (
        getattr(physical_system, "state_names", None),
        getattr(physical_system, "_state_names", None),
        getattr(obj, "state_names", None),
        getattr(obj, "_state_names", None),
    ):
        state_names = _to_name_list(candidate)
        if state_names:
            break

    reference_names: list[str] | None = None
    for candidate in (
        getattr(reference_generator, "reference_names", None),
        getattr(reference_generator, "_reference_names", None),
        getattr(obj, "reference_names", None),
        getattr(obj, "_reference_names", None),
    ):
        reference_names = _to_name_list(candidate)
        if reference_names:
            break

    if reference_names is None:
        referenced_states = _to_name_list(getattr(reference_generator, "referenced_states", None))
        if referenced_states:
            reference_names = [f"ref_{_sanitize_name(name)}" for name in referenced_states]
        else:
            mask = getattr(reference_generator, "referenced_states", None)
            if state_names and isinstance(mask, (list, tuple, np.ndarray)):
                try:
                    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
                    if mask_arr.size == len(state_names):
                        reference_names = [
                            f"ref_{_sanitize_name(name)}"
                            for name, flag in zip(state_names, mask_arr)
                            if bool(flag)
                        ]
                except Exception:
                    pass

    if state_names:
        state_names = _dedupe(state_names)
    if reference_names:
        reference_names = _dedupe(
            name if str(name).startswith("ref_") else f"ref_{name}" for name in reference_names
        )
    return EnvIntrospection(state_names=state_names, reference_names=reference_names)


def _is_custom_pmsm_env(*, env_id: str, obs_dim: int, act_dim: int, env: Any | None = None) -> bool:
    obj = getattr(env, "unwrapped", env)
    if bool(getattr(obj, "is_custom_pmsm_env", False)):
        return True
    names = getattr(obj, "observation_names", None)
    if names and list(names) == list(CUSTOM_OBSERVATION_NAMES):
        return True
    env_id_lower = str(env_id).lower()
    return (
        "custom-pmsm" in env_id_lower
        or "pmsm-current" in env_id_lower
        or ("custom" in env_id_lower and obs_dim == len(CUSTOM_OBSERVATION_NAMES) and act_dim == 2)
    )


def _custom_observation_spec(env_id: str) -> ObservationSpec:
    return ObservationSpec(
        env_id=env_id,
        layout="custom_dq",
        state_names=["i_d", "i_q", "omega_m"],
        reference_names=["ref_i_d", "ref_i_q"],
        extra_names=["e_d", "e_q", "prev_u_d", "prev_u_q", "T_L"],
        action_names=list(CUSTOM_ACTION_NAMES),
    )


def _infer_layout(env_id: str, act_dim: int) -> str:
    env_id_lower = env_id.lower()
    if "dq" in env_id_lower or act_dim == 2:
        return "dq"
    if act_dim == 3:
        return "abc"
    return "generic"



def _default_state_names(layout: str) -> list[str]:
    if layout == "dq":
        return list(DQ_STATE_NAMES)
    if layout == "abc":
        return list(ABC_STATE_NAMES)
    return []



def _default_action_names(layout: str, act_dim: int) -> list[str]:
    if layout == "dq":
        base = ["act_u_d", "act_u_q"]
    elif layout == "abc":
        base = ["act_u_a", "act_u_b", "act_u_c"]
    else:
        base = []
    if len(base) >= act_dim:
        return base[:act_dim]
    return [*base, *(f"act_{i}" for i in range(len(base), act_dim))]



def _expand_reference_names(channels: Sequence[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if not channels:
        return [f"ref_{i}" for i in range(count)]
    if count <= len(channels):
        return [f"ref_{name}" for name in channels[:count]]
    if count % len(channels) == 0:
        horizon = count // len(channels)
        names: list[str] = []
        for channel in channels:
            for step_idx in range(horizon):
                suffix = "" if step_idx == 0 else f"_t+{step_idx}"
                names.append(f"ref_{channel}{suffix}")
        return names
    return [f"ref_{i}" for i in range(count)]



def infer_observation_spec(
    *,
    env_id: str,
    obs_dim: int,
    act_dim: int,
    env: Any | None = None,
) -> ObservationSpec:
    """Infer a semantic flat-observation spec.

    Preference order:
    1. Live environment introspection.
    2. Known PMSM defaults from GEM literature.
    3. Generic fallback names.
    """

    if _is_custom_pmsm_env(env_id=env_id, obs_dim=obs_dim, act_dim=act_dim, env=env):
        return _custom_observation_spec(env_id)

    layout = _infer_layout(env_id=env_id, act_dim=act_dim)
    introspection = introspect_env(env) if env is not None else EnvIntrospection()

    state_names = list(introspection.state_names or _default_state_names(layout))
    reference_names = list(introspection.reference_names or [])

    if state_names and len(state_names) > obs_dim:
        state_names = state_names[:obs_dim]
    if reference_names and len(state_names) + len(reference_names) > obs_dim:
        reference_names = reference_names[: max(0, obs_dim - len(state_names))]

    if not state_names:
        if layout == "generic":
            state_names = [f"obs_{i}" for i in range(obs_dim)]
            reference_names = []
        else:
            default_states = _default_state_names(layout)
            if len(default_states) <= obs_dim:
                state_names = default_states
            else:
                state_names = default_states[:obs_dim]

    remaining = obs_dim - len(state_names) - len(reference_names)
    if remaining > 0:
        if layout == "dq":
            ref_channels = ["i_d", "i_q"]
        elif layout == "abc":
            ref_channels = ["i_a", "i_b", "i_c"]
        else:
            ref_channels = []
        reference_names.extend(_expand_reference_names(ref_channels, remaining))

    if len(state_names) + len(reference_names) < obs_dim:
        start = len(state_names) + len(reference_names)
        reference_names.extend(f"obs_{i}" for i in range(start, obs_dim))

    action_names = _default_action_names(layout, act_dim)
    return ObservationSpec(
        env_id=env_id,
        layout=layout,
        state_names=_dedupe(state_names),
        reference_names=_dedupe(reference_names),
        action_names=_dedupe(action_names),
    )


def build_observation_spec(*, env_id: str, env: Any) -> ObservationSpec:
    """Build an observation spec directly from a live environment."""
    obs_shape = getattr(getattr(env, "observation_space", None), "shape", None)
    act_shape = getattr(getattr(env, "action_space", None), "shape", None)
    if obs_shape is None or act_shape is None:
        raise ValueError("Environment observation/action spaces must expose shape information")
    if len(obs_shape) != 1 or len(act_shape) != 1:
        raise ValueError("Only flat 1D observation/action spaces are currently supported")
    return infer_observation_spec(
        env_id=env_id,
        obs_dim=int(obs_shape[0]),
        act_dim=int(act_shape[0]),
        env=env,
    )


def resolve_signal_indices(
    signal_names: Sequence[str],
    required_names: Sequence[str],
) -> "OrderedDict[str, int]":
    """Resolve required signal names to stable indices."""
    signal_to_idx = OrderedDict((str(name), idx) for idx, name in enumerate(signal_names))
    missing = [name for name in required_names if name not in signal_to_idx]
    if missing:
        raise KeyError(f"Missing required observation signals: {missing}")
    return OrderedDict((name, int(signal_to_idx[name])) for name in required_names)


def required_eval_trace_columns(layout: str | None = None) -> tuple[str, ...]:
    """Return the standard trace columns expected in PMSM eval CSVs."""
    if layout == "custom_dq":
        return ("i_d", "ref_i_d", "i_q", "ref_i_q")
    return (*TRACE_SIGNAL_NAMES, *TRACE_ACTION_NAMES)



def parse_flat_observation(
    flat_obs: Sequence[float] | np.ndarray,
    *,
    spec: ObservationSpec,
) -> "OrderedDict[str, float]":
    """Public helper to convert flat observations into named values."""
    return spec.to_mapping(flat_obs)

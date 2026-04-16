"""Simple replay buffer for off-policy RL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TransitionBatch:
    """Container for sampled transition batches."""

    obs: np.ndarray
    acts: np.ndarray
    rews: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Fixed-size circular replay buffer.

    Shapes:
    - obs:      (obs_dim,)
    - acts:     (act_dim,)
    - rews:     scalar
    - next_obs: (obs_dim,)
    - dones:    scalar in {0.0, 1.0}
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if obs_dim <= 0 or act_dim <= 0:
            raise ValueError("obs_dim and act_dim must be > 0")

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.capacity = int(capacity)

        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._acts = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self._rews = np.zeros((self.capacity, 1), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        """Return number of currently stored transitions."""
        return self._size

    def push(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool) -> None:
        """Append one transition to replay storage."""
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        act_arr = np.asarray(act, dtype=np.float32).reshape(-1)
        next_obs_arr = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        if obs_arr.shape[0] != self.obs_dim:
            raise ValueError(f"obs shape mismatch: got {obs_arr.shape[0]}, expected {self.obs_dim}")
        if act_arr.shape[0] != self.act_dim:
            raise ValueError(f"act shape mismatch: got {act_arr.shape[0]}, expected {self.act_dim}")
        if next_obs_arr.shape[0] != self.obs_dim:
            raise ValueError(f"next_obs shape mismatch: got {next_obs_arr.shape[0]}, expected {self.obs_dim}")

        self._obs[self._ptr] = obs_arr
        self._acts[self._ptr] = act_arr
        self._rews[self._ptr, 0] = float(rew)
        self._next_obs[self._ptr] = next_obs_arr
        self._dones[self._ptr, 0] = 1.0 if done else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a random batch of transitions."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self._size < batch_size:
            raise ValueError(f"not enough samples: buffer size {self._size}, requested {batch_size}")

        idx = np.random.randint(0, self._size, size=batch_size)
        return TransitionBatch(
            obs=self._obs[idx],  # (B, obs_dim)
            acts=self._acts[idx],  # (B, act_dim)
            rews=self._rews[idx],  # (B, 1)
            next_obs=self._next_obs[idx],  # (B, obs_dim)
            dones=self._dones[idx],  # (B, 1)
        )

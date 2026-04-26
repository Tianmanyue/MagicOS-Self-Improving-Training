"""Trajectory data structures for the AutoDAgger pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class TrajectorySource(str, Enum):
    SCRIPTED = "scripted"
    RECOVERY = "recovery"
    PERTURBATION = "perturbation"


@dataclass
class TrajectoryMetadata:
    source: TrajectorySource = TrajectorySource.SCRIPTED
    dagger_round: int = 0
    target_stage: Optional[str] = None
    original_trajectory_id: Optional[str] = None
    failure_frame: Optional[int] = None
    scene_id: Optional[str] = None


@dataclass
class TrajectoryData:
    """Container for a single trajectory's data, compatible with Isaac Lab HDF5 format."""
    states: dict[str, np.ndarray] = field(default_factory=dict)
    actions: Optional[np.ndarray] = None
    observations: dict[str, np.ndarray] = field(default_factory=dict)
    initial_state: dict[str, np.ndarray] = field(default_factory=dict)
    success: bool = False
    seed: Optional[int] = None
    metadata: TrajectoryMetadata = field(default_factory=TrajectoryMetadata)

    @property
    def num_steps(self) -> int:
        if self.actions is not None:
            return len(self.actions)
        return 0

    @property
    def state_keys(self) -> list[str]:
        return _collect_leaf_keys(self.states)

    @property
    def obs_keys(self) -> list[str]:
        return _collect_leaf_keys(self.observations)

    def get_state_at(self, frame: int) -> dict:
        """Extract all state arrays at a specific frame index."""
        return _slice_nested(self.states, frame)

    def slice(self, start: int, end: int) -> "TrajectoryData":
        """Return a sub-trajectory from start to end (exclusive)."""
        new_states = _slice_nested_range(self.states, start, end)
        new_obs = _slice_nested_range(self.observations, start, end)
        new_actions = self.actions[start:end] if self.actions is not None else None
        return TrajectoryData(
            states=new_states,
            actions=new_actions,
            observations=new_obs,
            initial_state=_slice_nested(self.states, start),
            success=False,
            seed=self.seed,
            metadata=self.metadata,
        )


def _collect_leaf_keys(d: dict, prefix: str = "") -> list[str]:
    keys = []
    for k, v in d.items():
        full = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            keys.extend(_collect_leaf_keys(v, full))
        else:
            keys.append(full)
    return keys


def _slice_nested(d: dict, idx: int) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _slice_nested(v, idx)
        elif isinstance(v, np.ndarray) and len(v.shape) > 0:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def _slice_nested_range(d: dict, start: int, end: int) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _slice_nested_range(v, start, end)
        elif isinstance(v, np.ndarray) and len(v.shape) > 0:
            out[k] = v[start:end]
        else:
            out[k] = v
    return out

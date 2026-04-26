"""Trajectory state replay in Isaac Sim.

Provides the ability to set the simulator to any recorded frame of a
trajectory and to re-execute the recorded actions for verification.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from autodagger.dataset.trajectory import TrajectoryData
from autodagger.sim.isaac_interface import IsaacInterface

logger = logging.getLogger(__name__)


class StateReplay:
    """Replay recorded trajectory states inside Isaac Sim.

    This is used for two purposes in the AutoDAgger pipeline:

    1. **Setting up recovery demos** -- teleport the simulator to the
       exact frame where the policy failed so the scripted expert can
       take over.
    2. **Verification** -- replay actions from a trajectory and compare
       the resulting states against the recorded ones to ensure
       determinism.
    """

    def __init__(self, isaac_interface: IsaacInterface) -> None:
        self.interface = isaac_interface

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replay_to_frame(
        self,
        trajectory: TrajectoryData,
        frame_idx: int,
        env_ids=None,
    ) -> None:
        """Set the simulator to the state recorded at *frame_idx*.

        Args:
            trajectory: Recorded trajectory containing per-frame states.
            frame_idx: Index into the trajectory's state arrays.
            env_ids: Optional environment indices to reset.  If ``None``,
                resets all environments.

        Raises:
            IndexError: If *frame_idx* is out of range for the trajectory.
        """
        if frame_idx < 0 or frame_idx >= trajectory.num_steps + 1:
            raise IndexError(
                f"frame_idx {frame_idx} out of range for trajectory with "
                f"{trajectory.num_steps} action steps "
                f"({trajectory.num_steps + 1} state frames)"
            )

        state = trajectory.get_state_at(frame_idx)

        # Convert numpy arrays to tensors on the correct device
        state_tensors = _to_device_tensors(state, self.interface.config.device)

        self.interface.set_state(state_tensors, env_ids=env_ids)
        logger.info(
            "Replayed to frame %d / %d of trajectory",
            frame_idx,
            trajectory.num_steps,
        )

    def replay_trajectory(
        self,
        trajectory: TrajectoryData,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> list[dict]:
        """Replay actions from a trajectory and collect resulting states.

        Sets the simulator to *start_frame*, then executes each recorded
        action in sequence, collecting the resulting state after each step.
        This is useful for verifying that the simulator is deterministic or
        for comparing recorded vs. replayed trajectories.

        Args:
            trajectory: Trajectory to replay.
            start_frame: First action frame to execute (inclusive).
            end_frame: Last action frame to execute (exclusive).  If
                ``None``, replays through the end of the trajectory.

        Returns:
            List of state dicts, one per step executed (length =
            ``end_frame - start_frame``).  Each dict has the same nested
            structure as :meth:`IsaacInterface.get_state`.
        """
        if trajectory.actions is None or trajectory.num_steps == 0:
            logger.warning("Cannot replay trajectory with no actions")
            return []

        if end_frame is None:
            end_frame = trajectory.num_steps
        end_frame = min(end_frame, trajectory.num_steps)

        if start_frame >= end_frame:
            logger.warning(
                "start_frame (%d) >= end_frame (%d), nothing to replay",
                start_frame,
                end_frame,
            )
            return []

        # Set simulator to the starting state
        self.replay_to_frame(trajectory, start_frame)

        collected_states: list[dict] = []
        actions = trajectory.actions

        for frame_idx in range(start_frame, end_frame):
            action = actions[frame_idx]
            action_tensor = _action_to_tensor(action, self.interface.config.device)

            _obs, _extras = self.interface.step(action_tensor)

            state = self.interface.get_state()
            collected_states.append(state)

        logger.info(
            "Replayed trajectory frames %d-%d (%d steps)",
            start_frame,
            end_frame,
            len(collected_states),
        )

        return collected_states

    def verify_trajectory(
        self,
        trajectory: TrajectoryData,
        tolerance: float = 1e-3,
    ) -> dict:
        """Replay a trajectory and compare against recorded states.

        Args:
            trajectory: Trajectory to verify.
            tolerance: Maximum allowable L2 deviation per state key before
                flagging a mismatch.

        Returns:
            Dict with keys: deterministic (bool), max_deviation (float),
            deviations_per_frame (list[float]), mismatched_frames (list[int]).
        """
        replayed_states = self.replay_trajectory(trajectory)

        deviations: list[float] = []
        mismatched: list[int] = []

        for frame_idx, replayed_state in enumerate(replayed_states):
            # Recorded state at frame_idx+1 (because replay starts after
            # setting the initial state and stepping produces the next state)
            recorded_frame = frame_idx + 1
            recorded_state = trajectory.get_state_at(recorded_frame)

            dev = _compute_state_deviation(recorded_state, replayed_state)
            deviations.append(dev)
            if dev > tolerance:
                mismatched.append(frame_idx)

        max_dev = max(deviations) if deviations else 0.0
        deterministic = len(mismatched) == 0

        result = {
            "deterministic": deterministic,
            "max_deviation": max_dev,
            "deviations_per_frame": deviations,
            "mismatched_frames": mismatched,
        }

        if deterministic:
            logger.info(
                "Trajectory verified: deterministic (max deviation %.6f)",
                max_dev,
            )
        else:
            logger.warning(
                "Trajectory NOT deterministic: %d mismatched frames, "
                "max deviation %.6f",
                len(mismatched),
                max_dev,
            )

        return result


# ======================================================================
# Internal helpers
# ======================================================================


def _to_device_tensors(state: dict, device: str) -> dict:
    """Convert numpy arrays in a nested state dict to torch tensors."""
    import torch

    out: dict = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = _to_device_tensors(v, device)
        elif isinstance(v, np.ndarray):
            out[k] = torch.tensor(v, device=device, dtype=torch.float32)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _action_to_tensor(action: np.ndarray, device: str):
    """Convert a numpy action array to a torch tensor with batch dim."""
    import torch

    t = torch.tensor(action, device=device, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)  # Add batch dimension
    return t


def _compute_state_deviation(recorded: dict, replayed: dict) -> float:
    """Compute the maximum L2 deviation between two nested state dicts."""
    max_dev = 0.0

    for k in recorded:
        rec_v = recorded[k]
        rep_v = replayed.get(k)
        if rep_v is None:
            continue

        if isinstance(rec_v, dict) and isinstance(rep_v, dict):
            max_dev = max(max_dev, _compute_state_deviation(rec_v, rep_v))
        else:
            rec_np = _as_numpy(rec_v)
            rep_np = _as_numpy(rep_v)
            if rec_np.shape == rep_np.shape:
                dev = float(np.linalg.norm(rec_np.flatten() - rep_np.flatten()))
                max_dev = max(max_dev, dev)

    return max_dev


def _as_numpy(v) -> np.ndarray:
    """Convert a tensor or scalar to a numpy array."""
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    return np.asarray(v)

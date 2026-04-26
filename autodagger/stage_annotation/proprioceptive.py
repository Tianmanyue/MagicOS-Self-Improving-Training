"""Proprioceptive signal analysis for discovering candidate stage boundaries.

Analyzes gripper width, joint velocities, and end-effector velocity to find
frames where the robot's behavior changes significantly -- indicative of
transitions between manipulation stages (approach, grasp, lift, place, etc.).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from autodagger.config import StageAnnotationConfig

logger = logging.getLogger(__name__)

# Signal type labels returned alongside each candidate boundary frame.
SIGNAL_GRIPPER = "gripper_change"
SIGNAL_JOINT_VEL = "joint_velocity_change"
SIGNAL_PAUSE = "ee_pause"

# Default window (in frames) within which nearby boundaries are merged.
_DEFAULT_MERGE_WINDOW: int = 5


class ProprioceptiveAnalyzer:
    """Detect candidate stage boundaries from proprioceptive signals.

    Parameters
    ----------
    gripper_width : np.ndarray, shape (T,)
        Gripper opening width over time.
    joint_velocities : np.ndarray, shape (T, num_joints)
        Joint velocities over time.
    ee_velocity : np.ndarray, shape (T, 3) or (T, 6)
        End-effector linear (and optionally angular) velocity.
    config : StageAnnotationConfig
        Thresholds for boundary detection.
    merge_window : int, optional
        Boundaries within this many frames of each other are merged into one.
        Defaults to ``_DEFAULT_MERGE_WINDOW``.
    """

    def __init__(
        self,
        gripper_width: np.ndarray,
        joint_velocities: np.ndarray,
        ee_velocity: np.ndarray,
        config: StageAnnotationConfig,
        merge_window: int = _DEFAULT_MERGE_WINDOW,
    ) -> None:
        # ---- validate and store inputs ----
        self.gripper_width = np.asarray(gripper_width, dtype=np.float64).ravel()
        self.joint_velocities = np.atleast_2d(
            np.asarray(joint_velocities, dtype=np.float64)
        )
        self.ee_velocity = np.atleast_2d(
            np.asarray(ee_velocity, dtype=np.float64)
        )

        self.T = len(self.gripper_width)
        if self.joint_velocities.shape[0] != self.T:
            raise ValueError(
                f"joint_velocities length {self.joint_velocities.shape[0]} "
                f"does not match gripper_width length {self.T}"
            )
        if self.ee_velocity.shape[0] != self.T:
            raise ValueError(
                f"ee_velocity length {self.ee_velocity.shape[0]} "
                f"does not match gripper_width length {self.T}"
            )

        self.config = config
        self.merge_window = merge_window

    # ------------------------------------------------------------------
    # Individual signal detectors
    # ------------------------------------------------------------------

    def find_gripper_boundaries(self) -> list[tuple[int, str]]:
        """Find frames where the gripper width changes abruptly.

        A boundary is placed at the first frame of each contiguous run where
        ``|gripper_width[t] - gripper_width[t-1]|`` exceeds
        ``config.gripper_change_threshold``.

        Returns
        -------
        list of (frame_idx, signal_type)
        """
        if self.T < 2:
            return []

        diff = np.abs(np.diff(self.gripper_width))  # length T-1
        above = diff > self.config.gripper_change_threshold  # bool mask

        boundaries: list[tuple[int, str]] = []
        in_event = False
        for i, is_change in enumerate(above):
            if is_change and not in_event:
                # Frame i+1 is where the change is first observed (diff is
                # between frame i and i+1), but we mark the onset frame.
                boundaries.append((i + 1, SIGNAL_GRIPPER))
                in_event = True
            elif not is_change:
                in_event = False

        logger.debug(
            "Gripper boundaries: %d found (threshold=%.4f)",
            len(boundaries),
            self.config.gripper_change_threshold,
        )
        return boundaries

    def find_velocity_boundaries(self) -> list[tuple[int, str]]:
        """Find frames where joint velocity magnitude has sudden changes.

        We compute the L2 norm of joint velocities at each timestep, then take
        the absolute gradient.  Frames where the gradient exceeds
        ``config.joint_velocity_threshold`` are returned.

        Returns
        -------
        list of (frame_idx, signal_type)
        """
        if self.T < 3:
            return []

        vel_norm = np.linalg.norm(self.joint_velocities, axis=1)  # (T,)
        grad = np.abs(np.gradient(vel_norm))  # (T,)

        indices = np.where(grad > self.config.joint_velocity_threshold)[0]

        # Group consecutive indices and keep only the first of each cluster.
        boundaries: list[tuple[int, str]] = []
        prev: int = -self.merge_window - 1
        for idx in indices:
            if idx - prev > self.merge_window:
                boundaries.append((int(idx), SIGNAL_JOINT_VEL))
                prev = idx

        logger.debug(
            "Joint-velocity boundaries: %d found (threshold=%.4f)",
            len(boundaries),
            self.config.joint_velocity_threshold,
        )
        return boundaries

    def find_pause_points(self) -> list[tuple[int, str]]:
        """Find frames where the end-effector is nearly stationary.

        A *pause* is a contiguous region where ``||ee_velocity||`` is below
        ``config.ee_velocity_zero_threshold``.  We return the midpoint frame
        of each such region.

        Returns
        -------
        list of (frame_idx, signal_type)
        """
        if self.T < 1:
            return []

        ee_norm = np.linalg.norm(self.ee_velocity, axis=1)  # (T,)
        below = ee_norm < self.config.ee_velocity_zero_threshold

        boundaries: list[tuple[int, str]] = []
        region_start: int | None = None

        for i, is_paused in enumerate(below):
            if is_paused and region_start is None:
                region_start = i
            elif not is_paused and region_start is not None:
                midpoint = (region_start + i - 1) // 2
                boundaries.append((midpoint, SIGNAL_PAUSE))
                region_start = None

        # Handle a pause that extends to the last frame.
        if region_start is not None:
            midpoint = (region_start + self.T - 1) // 2
            boundaries.append((midpoint, SIGNAL_PAUSE))

        logger.debug(
            "Pause points: %d found (threshold=%.4f)",
            len(boundaries),
            self.config.ee_velocity_zero_threshold,
        )
        return boundaries

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def find_all_boundaries(self) -> list[tuple[int, str]]:
        """Merge all candidate boundaries, deduplicate, and return sorted.

        Boundaries from the three detectors are combined.  When two (or more)
        boundaries fall within ``merge_window`` frames of each other they are
        merged into a single entry.  The kept frame is the one from the
        highest-priority signal (gripper > joint_velocity > pause) so that the
        most informative event is preserved.

        Returns
        -------
        list of (frame_idx, signal_type)
            Sorted in ascending frame order.
        """
        all_boundaries: list[tuple[int, str]] = []
        all_boundaries.extend(self.find_gripper_boundaries())
        all_boundaries.extend(self.find_velocity_boundaries())
        all_boundaries.extend(self.find_pause_points())

        if not all_boundaries:
            logger.info("No candidate boundaries found in any signal.")
            return []

        # Sort by frame index.
        all_boundaries.sort(key=lambda x: x[0])

        # Priority: prefer gripper events, then velocity, then pauses.
        _priority = {SIGNAL_GRIPPER: 0, SIGNAL_JOINT_VEL: 1, SIGNAL_PAUSE: 2}

        merged: list[tuple[int, str]] = []
        cluster_frames: list[int] = [all_boundaries[0][0]]
        cluster_signals: list[str] = [all_boundaries[0][1]]

        for frame, signal in all_boundaries[1:]:
            if frame - cluster_frames[0] <= self.merge_window:
                # Still within the current cluster.
                cluster_frames.append(frame)
                cluster_signals.append(signal)
            else:
                # Emit the representative of the finished cluster.
                merged.append(self._pick_representative(cluster_frames, cluster_signals, _priority))
                cluster_frames = [frame]
                cluster_signals = [signal]

        # Don't forget the last cluster.
        merged.append(self._pick_representative(cluster_frames, cluster_signals, _priority))

        logger.info(
            "Total candidate boundaries after merging: %d "
            "(from %d raw detections, merge_window=%d)",
            len(merged),
            len(all_boundaries),
            self.merge_window,
        )
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_representative(
        frames: list[int],
        signals: list[str],
        priority: dict[str, int],
    ) -> tuple[int, str]:
        """From a cluster of nearby boundaries pick the best representative.

        The representative is the entry with the highest-priority signal
        (lowest priority number).  Ties are broken by earliest frame.
        """
        best_idx = 0
        for i in range(1, len(frames)):
            if priority.get(signals[i], 99) < priority.get(signals[best_idx], 99):
                best_idx = i
            elif (
                priority.get(signals[i], 99) == priority.get(signals[best_idx], 99)
                and frames[i] < frames[best_idx]
            ):
                best_idx = i
        return (frames[best_idx], signals[best_idx])

"""Failure detection based on progress score stagnation."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from autodagger.config import ProgressModelConfig
from autodagger.stage_annotation.schema import TrajectoryStageAnnotation

logger = logging.getLogger(__name__)


class FailureDetector:
    """Detects stuck policies by analysing progress score trajectories."""

    def __init__(self, config: ProgressModelConfig) -> None:
        self._window = config.stuck_window
        self._threshold = config.stuck_threshold

    def detect_stuck(
        self,
        progress_scores: list[float],
    ) -> tuple[bool, Optional[int]]:
        """Check whether progress has stalled over the configured window.

        Returns (is_stuck, stuck_start_frame).  If the trajectory is shorter
        than the window or progress is still increasing, returns (False, None).
        """
        if len(progress_scores) < self._window:
            return False, None

        for start in range(len(progress_scores) - self._window + 1):
            end = start + self._window - 1
            window_min = min(progress_scores[start : end + 1])
            window_max = max(progress_scores[start : end + 1])
            if window_max - window_min <= self._threshold:
                logger.debug(
                    "Stuck detected: frames %d-%d, range %.4f",
                    start, end, window_max - window_min,
                )
                return True, start

        return False, None

    def detect_failures_batch(
        self,
        trajectories: list[dict],
    ) -> list[dict]:
        """Analyse a batch of trajectories for stuck behaviour.

        Each input dict must contain:
            trajectory_id, progress_scores, success

        Returns one result dict per trajectory with:
            trajectory_id, is_stuck, stuck_frame, final_progress, success
        """
        results: list[dict] = []
        for traj in trajectories:
            tid = traj["trajectory_id"]
            scores = traj["progress_scores"]
            success = traj["success"]

            is_stuck, stuck_frame = self.detect_stuck(scores)
            final_progress = scores[-1] if scores else 0.0

            results.append({
                "trajectory_id": tid,
                "is_stuck": is_stuck,
                "stuck_frame": stuck_frame,
                "final_progress": final_progress,
                "success": success,
            })

            if is_stuck:
                logger.info(
                    "Trajectory %s: stuck at frame %d (final progress %.3f)",
                    tid, stuck_frame, final_progress,
                )

        return results

    @staticmethod
    def identify_failure_stage(
        failure_info: dict,
        stage_annotation: TrajectoryStageAnnotation,
    ) -> Optional[int]:
        """Map the stuck frame to the stage it falls within.

        Returns the stage id, or None if the trajectory was not stuck or no
        matching stage is found.
        """
        stuck_frame = failure_info.get("stuck_frame")
        if stuck_frame is None:
            return None

        for stage in stage_annotation.stages:
            if stage.start_frame <= stuck_frame <= stage.end_frame:
                return stage.id

        logger.warning(
            "Stuck frame %d does not fall within any annotated stage for %s",
            stuck_frame, stage_annotation.trajectory_id,
        )
        return None

    @classmethod
    def summarize_failures(
        cls,
        failure_results: list[dict],
        annotations: list[TrajectoryStageAnnotation],
    ) -> dict:
        """Aggregate failure statistics across a batch of trajectories.

        Returns a summary dict with:
            total, num_failures, failure_rate,
            per_stage_failure_count, most_common_failure_stage
        """
        annotation_map = {a.trajectory_id: a for a in annotations}

        stage_counter: Counter[int] = Counter()
        num_failures = 0

        for result in failure_results:
            if not result["is_stuck"]:
                continue

            num_failures += 1
            tid = result["trajectory_id"]
            ann = annotation_map.get(tid)
            if ann is None:
                logger.warning("No stage annotation for trajectory %s", tid)
                continue

            stage_id = cls.identify_failure_stage(result, ann)
            if stage_id is not None:
                stage_counter[stage_id] += 1

        total = len(failure_results)
        failure_rate = num_failures / total if total > 0 else 0.0

        most_common = stage_counter.most_common(1)
        most_common_stage = most_common[0][0] if most_common else None

        summary = {
            "total": total,
            "num_failures": num_failures,
            "failure_rate": failure_rate,
            "per_stage_failure_count": dict(stage_counter),
            "most_common_failure_stage": most_common_stage,
        }

        logger.info(
            "Failure summary: %d/%d (%.1f%%) stuck, most common stage: %s",
            num_failures, total, failure_rate * 100, most_common_stage,
        )

        return summary

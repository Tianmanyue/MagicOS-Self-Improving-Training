"""Evaluation metrics computation for the AutoDAgger pipeline."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from autodagger.dataset.trajectory import TrajectoryData
from autodagger.stage_annotation.schema import TrajectoryStageAnnotation

logger = logging.getLogger(__name__)


class EvalMetrics:
    """Computes and aggregates evaluation metrics across trajectories and rounds."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Per-round metrics
    # ------------------------------------------------------------------

    def compute_from_trajectories(self, trajectories: list[TrajectoryData]) -> dict:
        """Compute summary statistics from a list of evaluated trajectories.

        Args:
            trajectories: List of trajectories from a single evaluation round.

        Returns:
            Dict with keys: success_rate, avg_episode_length,
            std_episode_length, num_episodes, num_successes, num_failures.
        """
        if not trajectories:
            logger.warning("compute_from_trajectories called with empty list")
            return {
                "success_rate": 0.0,
                "avg_episode_length": 0.0,
                "std_episode_length": 0.0,
                "num_episodes": 0,
                "num_successes": 0,
                "num_failures": 0,
            }

        num_episodes = len(trajectories)
        successes = sum(1 for t in trajectories if t.success)
        failures = num_episodes - successes

        lengths = [t.num_steps for t in trajectories]
        avg_length = float(np.mean(lengths))
        std_length = float(np.std(lengths))

        metrics = {
            "success_rate": successes / num_episodes,
            "avg_episode_length": avg_length,
            "std_episode_length": std_length,
            "num_episodes": num_episodes,
            "num_successes": successes,
            "num_failures": failures,
        }

        logger.info(
            "Metrics: success_rate=%.3f (%d/%d), avg_len=%.1f +/- %.1f",
            metrics["success_rate"],
            successes,
            num_episodes,
            avg_length,
            std_length,
        )

        return metrics

    # ------------------------------------------------------------------
    # Stage-level failure analysis
    # ------------------------------------------------------------------

    def compute_stage_failure_distribution(
        self,
        trajectories: list[TrajectoryData],
        annotations: list[TrajectoryStageAnnotation],
    ) -> dict:
        """For failed trajectories, compute which stage they fail in.

        Args:
            trajectories: Evaluated trajectories (success and failure).
            annotations: Stage annotations aligned with the trajectories.
                Must be the same length as *trajectories* (or matched by
                trajectory id -- see below).

        Returns:
            Dict with keys: stage_failure_counts ({stage_name: count}),
            most_common_failure_stage, total_failures_analysed.
        """
        # Build annotation lookup by trajectory id when available
        ann_by_id: dict[str, TrajectoryStageAnnotation] = {}
        for ann in annotations:
            ann_by_id[ann.trajectory_id] = ann

        stage_counter: Counter[str] = Counter()
        total_analysed = 0

        for idx, traj in enumerate(trajectories):
            if traj.success:
                continue

            # Try to find matching annotation
            traj_id = getattr(traj.metadata, "original_trajectory_id", None) or str(idx)
            ann = ann_by_id.get(traj_id)

            # Fall back to positional matching
            if ann is None and idx < len(annotations):
                ann = annotations[idx]

            if ann is None:
                logger.debug(
                    "No stage annotation for failed trajectory %s (idx=%d)",
                    traj_id,
                    idx,
                )
                continue

            failure_stage = ann.failure_stage
            if failure_stage is not None:
                stage_counter[failure_stage.name] += 1
                total_analysed += 1
            else:
                # Heuristic: attribute failure to the last stage the policy
                # was executing (the stage containing the final frame).
                last_frame = traj.num_steps - 1 if traj.num_steps > 0 else 0
                for stage in reversed(ann.stages):
                    if stage.start_frame <= last_frame <= stage.end_frame:
                        stage_counter[stage.name] += 1
                        total_analysed += 1
                        break

        most_common = stage_counter.most_common(1)
        most_common_name = most_common[0][0] if most_common else None

        result = {
            "stage_failure_counts": dict(stage_counter),
            "most_common_failure_stage": most_common_name,
            "total_failures_analysed": total_analysed,
        }

        logger.info(
            "Stage failure distribution (%d analysed): %s, worst stage: %s",
            total_analysed,
            dict(stage_counter),
            most_common_name,
        )

        return result

    # ------------------------------------------------------------------
    # Cross-round comparison
    # ------------------------------------------------------------------

    def compare_rounds(
        self,
        round_metrics: list[dict],
        convergence_threshold: float = 0.02,
    ) -> dict:
        """Compare metrics across DAgger rounds to assess improvement.

        Args:
            round_metrics: List of metric dicts (one per round), each
                containing at least ``success_rate``.
            convergence_threshold: If the improvement between the last two
                rounds is below this value, the pipeline is considered
                converged.

        Returns:
            Dict with keys: per_round_success_rate, improvement_per_round,
            total_improvement, converged.
        """
        if not round_metrics:
            return {
                "per_round_success_rate": [],
                "improvement_per_round": [],
                "total_improvement": 0.0,
                "converged": False,
            }

        rates = [m.get("success_rate", 0.0) for m in round_metrics]
        improvements = []
        for i in range(1, len(rates)):
            improvements.append(rates[i] - rates[i - 1])

        total_improvement = rates[-1] - rates[0] if len(rates) > 1 else 0.0
        converged = (
            len(improvements) >= 1
            and abs(improvements[-1]) < convergence_threshold
        )

        result = {
            "per_round_success_rate": rates,
            "improvement_per_round": improvements,
            "total_improvement": total_improvement,
            "converged": converged,
        }

        logger.info(
            "Round comparison: rates=%s, improvements=%s, converged=%s",
            [f"{r:.3f}" for r in rates],
            [f"{i:+.3f}" for i in improvements],
            converged,
        )

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_report(metrics: dict, path: str) -> None:
        """Save a metrics dict to a JSON file.

        Args:
            metrics: Arbitrary metrics dict (must be JSON-serialisable).
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python for JSON serialisation
        clean = _make_json_serializable(metrics)

        with open(path, "w") as f:
            json.dump(clean, f, indent=2)

        logger.info("Saved metrics report to %s", path)


# ======================================================================
# Internal helpers
# ======================================================================


def _make_json_serializable(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

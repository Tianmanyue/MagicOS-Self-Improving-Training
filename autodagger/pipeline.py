"""AutoDAgger pipeline: fully automated DAgger in simulation.

Orchestrates the complete closed-loop:
  Round 0:  scripted expert → initial demos → train π₁
  Round N:  πₙ rollout → failure detection → VLM stage annotation →
            scripted expert recovery → merge dataset → train πₙ₊₁
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from autodagger.config import PipelineConfig
from autodagger.dataset.hdf5_manager import HDF5Manager
from autodagger.dataset.versioning import DatasetVersionManager
from autodagger.dataset.trajectory import TrajectoryData, TrajectoryMetadata, TrajectorySource
from autodagger.eval.evaluator import PolicyEvaluator
from autodagger.eval.metrics import EvalMetrics
from autodagger.progress.top_reward import TOPRewardEstimator
from autodagger.progress.failure_detector import FailureDetector
from autodagger.stage_annotation.vlm_annotator import VLMStageAnnotator
from autodagger.stage_annotation.schema import StageLabel, TrajectoryStageAnnotation
from autodagger.training.robomimic_trainer import RobomimicTrainer
from autodagger.training.ra_bc import RABCWeighter

logger = logging.getLogger(__name__)


class AutoDAggerPipeline:
    """Main orchestrator for the automated DAgger loop."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.current_round = 0
        self.round_metrics: list[dict] = []
        self.best_checkpoint: str | None = None

        # task → stage definition cache: annotate once, reuse every round
        self._task_stage_cache: dict[str, list[StageLabel]] = {}

        self._work_dir = Path(config.work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)

        self._hdf5 = HDF5Manager(config.dataset)
        self._versions = DatasetVersionManager(
            config.dataset.base_dir, config.dataset.filename_prefix
        )
        self._evaluator = PolicyEvaluator(config.eval, config.isaac_sim)
        self._trainer = RobomimicTrainer(config.training)
        self._metrics = EvalMetrics()
        self._failure_detector = FailureDetector(config.progress)

        config.save(str(self._work_dir / "pipeline_config.json"))
        logger.info("AutoDAgger pipeline initialized at %s", self._work_dir)

    # ==================================================================
    # Main entry point
    # ==================================================================

    def run(
        self,
        obs_keys: list[str],
        action_dim: int,
        task_description: str,
        initial_demos_path: str | None = None,
        num_initial_demos: int = 200,
    ) -> dict:
        """Run the full AutoDAgger loop.

        Parameters
        ----------
        obs_keys : list[str]
            Observation key names for training config.
        action_dim : int
            Action space dimensionality.
        task_description : str
            Human-readable task name (for VLM prompts).
        initial_demos_path : str, optional
            Path to pre-existing initial demos HDF5. If None, collects
            new demos using the scripted expert.
        num_initial_demos : int
            Number of initial demos to collect if no path given.

        Returns
        -------
        dict
            Final summary with per-round metrics, best checkpoint, etc.
        """
        logger.info(
            "=== AutoDAgger starting: target=%.0f%%, max_rounds=%d ===",
            self.config.dagger.target_success_rate * 100,
            self.config.dagger.max_rounds,
        )

        # ---- Round 0: initial training ----
        if initial_demos_path and os.path.exists(initial_demos_path):
            round0_path = initial_demos_path
            logger.info("Using pre-existing initial demos: %s", round0_path)
        else:
            round0_path = self._collect_initial_demos(num_initial_demos)

        self._versions.create_round_dataset(
            0, self._hdf5.read_all_trajectories(round0_path)
        )
        cumulative_path = self._versions.create_cumulative_dataset(0)

        checkpoint = self._trainer.train(
            cumulative_path, obs_keys, action_dim, round_num=0
        )
        self.best_checkpoint = checkpoint
        logger.info("Round 0 training complete: %s", checkpoint)

        # ---- Round 0: evaluate ----
        eval_summary = self._evaluate_round(checkpoint, 0)
        self.round_metrics.append(eval_summary)
        self.current_round = 1

        # ---- DAgger loop ----
        for round_num in range(1, self.config.dagger.max_rounds + 1):
            self.current_round = round_num

            sr = eval_summary["success_rate"]
            logger.info(
                "=== Round %d (current success_rate=%.1f%%) ===",
                round_num, sr * 100,
            )

            if sr >= self.config.dagger.target_success_rate:
                logger.info(
                    "Target success rate reached (%.1f%% >= %.1f%%). Stopping.",
                    sr * 100,
                    self.config.dagger.target_success_rate * 100,
                )
                break

            if round_num >= 2:
                improvement = sr - self.round_metrics[-2]["success_rate"]
                if improvement < self.config.dagger.min_improvement:
                    logger.info(
                        "Improvement too small (%.3f < %.3f). Stopping.",
                        improvement,
                        self.config.dagger.min_improvement,
                    )
                    break

            # ---- Analyze failures ----
            failure_path = eval_summary.get("failure_path")
            if not failure_path or not os.path.exists(failure_path):
                logger.warning("No failure trajectories found. Stopping.")
                break

            failures = self._analyze_failures(
                failure_path, task_description
            )

            if not failures:
                logger.info("No recoverable failures identified. Stopping.")
                break

            # ---- Collect recovery demos ----
            recovery_demos = self._collect_recovery_demos(
                failures, round_num
            )

            if not recovery_demos:
                logger.warning("No recovery demos collected. Stopping.")
                break

            # ---- Update dataset and retrain ----
            self._versions.create_round_dataset(round_num, recovery_demos)
            cumulative_path = self._versions.create_cumulative_dataset(round_num)

            if self.config.training.use_ra_bc:
                self._apply_ra_bc_weights(
                    cumulative_path, task_description
                )

            checkpoint = self._trainer.train(
                cumulative_path, obs_keys, action_dim, round_num=round_num
            )
            self.best_checkpoint = checkpoint

            # ---- Evaluate ----
            eval_summary = self._evaluate_round(checkpoint, round_num)
            self.round_metrics.append(eval_summary)

        # ---- Final summary ----
        final = self._build_final_summary(task_description)
        summary_path = str(self._work_dir / "final_summary.json")
        with open(summary_path, "w") as f:
            json.dump(final, f, indent=2, default=str)

        logger.info("=== AutoDAgger complete ===")
        logger.info("Best checkpoint: %s", self.best_checkpoint)
        logger.info("Final success rate: %.1f%%", final["best_success_rate"] * 100)

        return final

    # ==================================================================
    # Pipeline steps
    # ==================================================================

    def _collect_initial_demos(self, num_demos: int) -> str:
        """Round 0: collect initial demos with scripted expert."""
        from autodagger.sim.isaac_interface import IsaacInterface
        from autodagger.sim.expert_collector import ExpertCollector

        logger.info("Collecting %d initial demonstrations...", num_demos)

        isaac = IsaacInterface(self.config.isaac_sim)
        collector = ExpertCollector(isaac, self.config.expert)
        demos = collector.collect_initial_demos(num_demos)

        output_path = str(self._work_dir / "initial_demos.hdf5")
        self._hdf5.append_trajectories(output_path, demos)

        stats = self._hdf5.get_dataset_stats(output_path)
        logger.info(
            "Initial demos: %d total, %d successful (%.1f%%)",
            stats["num_demos"],
            stats["num_successful"],
            stats["num_successful"] / max(stats["num_demos"], 1) * 100,
        )
        return output_path

    def _evaluate_round(self, checkpoint: str, round_num: int) -> dict:
        """Evaluate a checkpoint and return summary."""
        eval_dir = str(self._work_dir / f"eval_round_{round_num}")
        summary = self._evaluator.evaluate(checkpoint, eval_dir)
        summary["round"] = round_num

        logger.info(
            "Round %d eval: %.1f%% success (%d/%d)",
            round_num,
            summary["success_rate"] * 100,
            summary["num_successes"],
            summary["num_episodes"],
        )
        return summary

    def _analyze_failures(
        self,
        failure_hdf5_path: str,
        task_description: str,
    ) -> list[dict]:
        """Analyze failed trajectories: detect stuck frames and annotate stages.

        Two-tier approach:
        - If rendered frames are available: use TOPReward + FailureDetector
          for precise stuck-frame detection, plus VLM stage annotation.
        - Fallback: use proprioceptive signals for boundary detection and
          a heuristic for failure frame.

        Returns a list of dicts with keys: trajectory, failure_frame, demo_idx,
        and optionally stage_annotation.
        """
        failed_trajs = self._hdf5.read_failed_trajectories(failure_hdf5_path)

        if not failed_trajs:
            return []

        failures = []
        annotations: list[TrajectoryStageAnnotation] = []

        for i, traj in enumerate(failed_trajs):
            if traj.num_steps == 0:
                continue

            traj_id = f"demo_{i}"
            failure_frame: int | None = None
            annotation: TrajectoryStageAnnotation | None = None

            # --- Tier 1: VLM-based analysis (if frames available) ---
            rendered_frames = self._get_rendered_frames(failure_hdf5_path, traj_id)

            if rendered_frames is not None and len(rendered_frames) > 0:
                failure_frame, annotation = self._vlm_failure_analysis(
                    traj, rendered_frames, traj_id, task_description
                )

            # --- Tier 2: proprioceptive-only fallback ---
            if failure_frame is None:
                failure_frame = self._proprioceptive_failure_estimate(traj)

            failures.append({
                "trajectory": traj,
                "failure_frame": failure_frame,
                "demo_idx": i,
                "stage_annotation": annotation,
            })

            if annotation is not None:
                annotations.append(annotation)

        if annotations:
            failure_inputs = [
                {
                    "trajectory_id": f["demo_idx"],
                    "progress_scores": [0.0] * f["trajectory"].num_steps,
                    "success": False,
                }
                for f in failures
            ]
            summary = self._failure_detector.summarize_failures(
                self._failure_detector.detect_failures_batch(failure_inputs),
                annotations,
            )
            logger.info("Failure stage summary: %s", summary)

        cap = self.config.dagger.max_recovery_demos_per_round
        if len(failures) > cap:
            logger.info("Capping recovery targets from %d to %d", len(failures), cap)
            failures = failures[:cap]

        logger.info("Identified %d failure cases for recovery", len(failures))
        return failures

    def _discover_task_stages(
        self,
        frames: list[np.ndarray],
        proprioceptive_data: dict | None,
        task_description: str,
    ) -> list[StageLabel]:
        """Discover stage definitions for a task (called once, then cached).

        Uses VLM + proprioceptive analysis on one representative trajectory
        to identify the canonical stage structure for this task.
        """
        if task_description in self._task_stage_cache:
            return self._task_stage_cache[task_description]

        logger.info("Discovering stage definitions for task: %s", task_description)

        annotator = VLMStageAnnotator(
            self.config.anthropic, self.config.stage_annotation
        )

        if proprioceptive_data is None:
            proprioceptive_data = {
                "gripper_width": np.zeros(len(frames)),
                "joint_velocities": np.zeros((len(frames), 7)),
                "ee_velocity": np.zeros((len(frames), 3)),
            }

        annotation = annotator.annotate_trajectory(
            frames, proprioceptive_data,
            trajectory_id="stage_discovery",
            task=task_description,
        )

        self._task_stage_cache[task_description] = annotation.stages

        stage_names = [s.name for s in annotation.stages]
        logger.info(
            "Task '%s' stage definition cached: %s",
            task_description, stage_names,
        )

        # Persist to disk for cross-session reuse
        cache_path = str(self._work_dir / "stage_definitions.json")
        cache_data = {
            "task": task_description,
            "stages": [
                {"id": s.id, "name": s.name, "key_signal": s.key_signal}
                for s in annotation.stages
            ],
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        return annotation.stages

    def _map_failure_to_stage(
        self,
        failure_frame: int,
        total_frames: int,
        task_description: str,
    ) -> tuple[TrajectoryStageAnnotation, int | None]:
        """Map a failure frame to the cached stage definition.

        Proportionally maps the cached stage boundaries onto this
        trajectory's frame count, then finds which stage the failure
        falls into.
        """
        cached_stages = self._task_stage_cache.get(task_description, [])
        if not cached_stages:
            return TrajectoryStageAnnotation(
                trajectory_id="", task=task_description,
                total_frames=total_frames,
            ), None

        # Scale cached stage boundaries to this trajectory's length
        ref_total = cached_stages[-1].end_frame + 1
        scaled_stages: list[StageLabel] = []
        for s in cached_stages:
            scaled_start = int(s.start_frame * total_frames / ref_total)
            scaled_end = int(s.end_frame * total_frames / ref_total)
            scaled_stages.append(StageLabel(
                id=s.id,
                name=s.name,
                start_frame=scaled_start,
                end_frame=min(scaled_end, total_frames - 1),
                key_signal=s.key_signal,
            ))

        annotation = TrajectoryStageAnnotation(
            trajectory_id="",
            task=task_description,
            stages=scaled_stages,
            total_frames=total_frames,
        )

        stage_id = self._failure_detector.identify_failure_stage(
            {"stuck_frame": failure_frame}, annotation
        )
        annotation.failure_stage_id = stage_id

        return annotation, stage_id

    def _vlm_failure_analysis(
        self,
        traj: TrajectoryData,
        frames: list[np.ndarray],
        traj_id: str,
        task_description: str,
    ) -> tuple[int | None, TrajectoryStageAnnotation | None]:
        """Detect failure frame via progress model, map to cached stages.

        Stage discovery is only done once (first call); subsequent calls
        reuse the cached stage definition and just do progress + mapping.
        """
        try:
            # Step 1: discover stages (first trajectory only, then cached)
            if task_description not in self._task_stage_cache:
                propri = self._extract_proprioceptive(traj)
                self._discover_task_stages(frames, propri, task_description)

            # Step 2: estimate progress and detect stuck frame
            progress_estimator = TOPRewardEstimator(
                self.config.anthropic, self.config.progress
            )
            scores = progress_estimator.estimate_progress(frames, task_description)

            is_stuck, stuck_frame = self._failure_detector.detect_stuck(scores)
            failure_frame = stuck_frame if is_stuck else max(0, len(scores) - 1)

            # Step 3: map failure to cached stage (no VLM call)
            annotation, stage_id = self._map_failure_to_stage(
                failure_frame, traj.num_steps, task_description
            )
            annotation.trajectory_id = traj_id
            annotation.success = False

            return failure_frame, annotation

        except Exception as e:
            logger.warning("VLM failure analysis failed for %s: %s", traj_id, e)
            return None, None

    @staticmethod
    def _proprioceptive_failure_estimate(traj: TrajectoryData) -> int:
        """Fallback: estimate failure frame from proprioceptive signals."""
        from autodagger.stage_annotation.proprioceptive import (
            ProprioceptiveAnalyzer,
        )
        from autodagger.config import StageAnnotationConfig

        # Try to use proprioceptive signals to find the last activity boundary
        propri = _try_extract_proprioceptive(traj)
        if propri is not None:
            try:
                analyzer = ProprioceptiveAnalyzer(
                    gripper_width=propri["gripper_width"],
                    joint_velocities=propri["joint_velocities"],
                    ee_velocity=propri["ee_velocity"],
                    config=StageAnnotationConfig(),
                )
                boundaries = analyzer.find_all_boundaries()
                if boundaries:
                    # Use the last detected boundary as the failure onset
                    return boundaries[-1][0]
            except Exception:
                pass

        # Final fallback: 80% of trajectory
        return max(0, int(traj.num_steps * 0.8) - 1)

    @staticmethod
    def _extract_proprioceptive(traj: TrajectoryData) -> dict | None:
        """Extract proprioceptive arrays from trajectory observations/states."""
        return _try_extract_proprioceptive(traj)

    @staticmethod
    def _get_rendered_frames(
        hdf5_path: str, traj_id: str
    ) -> list[np.ndarray] | None:
        """Try to load rendered RGB frames for a trajectory.

        Frames may be stored in:
        - data/<traj_id>/obs/rgb (if eval saved frames)
        - A companion directory <hdf5_stem>_frames/<traj_id>/
        """
        import h5py
        from pathlib import Path

        # Check HDF5 for rgb obs
        try:
            with h5py.File(hdf5_path, "r") as f:
                rgb_path = f"data/{traj_id}/obs/rgb"
                if rgb_path in f:
                    frames = np.array(f[rgb_path])
                    return [frames[i] for i in range(frames.shape[0])]
        except Exception:
            pass

        # Check companion frames directory
        frames_dir = Path(hdf5_path).with_suffix("") / f"{traj_id}_frames"
        if frames_dir.exists():
            import glob
            from PIL import Image
            frame_files = sorted(glob.glob(str(frames_dir / "*.png")))
            if frame_files:
                return [np.array(Image.open(p)) for p in frame_files]

        return None

    def _collect_recovery_demos(
        self,
        failures: list[dict],
        round_num: int,
    ) -> list[TrajectoryData]:
        """Collect recovery demos from failure states."""
        from autodagger.sim.isaac_interface import IsaacInterface
        from autodagger.sim.expert_collector import ExpertCollector

        isaac = IsaacInterface(self.config.isaac_sim)
        collector = ExpertCollector(isaac, self.config.expert)

        recovery_demos = collector.collect_recovery_demos_batch(
            failures, dagger_round=round_num
        )

        logger.info(
            "Round %d: collected %d / %d recovery demos",
            round_num,
            len(recovery_demos),
            len(failures),
        )
        return recovery_demos

    def _apply_ra_bc_weights(
        self,
        dataset_path: str,
        task_description: str,
    ) -> None:
        """Apply RA-BC weights to the training dataset.

        Uses VLM progress scores when frames are available, otherwise
        falls back to a temporal proxy (linear ramp for success,
        plateau for failure).
        """
        logger.info("Applying RA-BC weights to %s", dataset_path)

        weighter = RABCWeighter(
            delta=self.config.training.ra_bc_delta,
            normalize=self.config.training.ra_bc_normalize,
        )

        trajectories = self._hdf5.read_all_trajectories(dataset_path)
        progress_by_demo: dict[str, np.ndarray] = {}

        progress_estimator = None

        for i, traj in enumerate(trajectories):
            demo_name = f"demo_{i}"
            T = max(traj.num_steps, 1)

            frames = self._get_rendered_frames(dataset_path, demo_name)
            if frames is not None and len(frames) > 0:
                if progress_estimator is None:
                    progress_estimator = TOPRewardEstimator(
                        self.config.anthropic, self.config.progress
                    )
                try:
                    scores = progress_estimator.estimate_progress(
                        frames, task_description
                    )
                    progress_by_demo[demo_name] = np.array(scores)
                    continue
                except Exception as e:
                    logger.warning(
                        "VLM progress failed for %s, using proxy: %s",
                        demo_name, e,
                    )

            if traj.success:
                progress_by_demo[demo_name] = np.linspace(0.0, 1.0, T)
            else:
                progress_by_demo[demo_name] = np.linspace(0.0, 0.5, T)

        weighter.apply_weights_to_dataset(
            dataset_path, progress_by_demo, dataset_path
        )

    def _build_final_summary(self, task_description: str) -> dict:
        """Build the final pipeline summary."""
        comparison = self._metrics.compare_rounds(self.round_metrics)

        return {
            "task": task_description,
            "total_rounds": len(self.round_metrics),
            "best_checkpoint": self.best_checkpoint,
            "best_success_rate": comparison["best_success_rate"],
            "best_round": comparison["best_round"],
            "converged": comparison["converged"],
            "round_metrics": self.round_metrics,
            "comparison": comparison,
            "config": str(self._work_dir / "pipeline_config.json"),
            "dataset_history": self._versions.get_history(),
        }

    # ==================================================================
    # Step-by-step API for manual verification (Phase 1)
    # ==================================================================

    def step_train(
        self,
        dataset_path: str,
        obs_keys: list[str],
        action_dim: int,
        round_num: int = 0,
    ) -> str:
        """Manual step: train a policy and return checkpoint path."""
        return self._trainer.train(dataset_path, obs_keys, action_dim, round_num)

    def step_evaluate(self, checkpoint: str, round_num: int = 0) -> dict:
        """Manual step: evaluate a checkpoint."""
        return self._evaluate_round(checkpoint, round_num)

    def step_analyze(self, failure_path: str, task_desc: str) -> list[dict]:
        """Manual step: analyze failures."""
        return self._analyze_failures(failure_path, task_desc)

    def step_collect_recovery(
        self, failures: list[dict], round_num: int = 0
    ) -> list[TrajectoryData]:
        """Manual step: collect recovery demos."""
        return self._collect_recovery_demos(failures, round_num)

    def step_merge_and_version(
        self, round_num: int, new_trajectories: list[TrajectoryData]
    ) -> str:
        """Manual step: version new data and create cumulative dataset."""
        self._versions.create_round_dataset(round_num, new_trajectories)
        return self._versions.create_cumulative_dataset(round_num)


# ======================================================================
# Module-level helpers
# ======================================================================

_OBS_KEY_ALIASES = {
    "gripper_width": [
        "gripper_width", "gripper_finger_joint1",
        "gripper_actions", "gripper_pos",
    ],
    "joint_velocities": [
        "joint_vel", "joint_velocity", "joint_velocities",
    ],
    "ee_velocity": [
        "ee_vel", "ee_velocity", "end_effector_velocity",
        "ee_pose",  # will need derivative
    ],
}


def _try_extract_proprioceptive(traj: TrajectoryData) -> dict | None:
    """Best-effort extraction of proprioceptive signals from a trajectory.

    Looks in both traj.observations and traj.states for known key names.
    Returns dict with gripper_width, joint_velocities, ee_velocity or None.
    """
    T = traj.num_steps
    if T == 0:
        return None

    result: dict[str, np.ndarray] = {}

    search_space = {}
    search_space.update(_flatten(traj.observations))
    search_space.update(_flatten(traj.states))

    for target, aliases in _OBS_KEY_ALIASES.items():
        for alias in aliases:
            if alias in search_space:
                arr = search_space[alias]
                if isinstance(arr, np.ndarray) and arr.shape[0] == T:
                    result[target] = arr
                    break

    if len(result) < 3:
        # Fill missing signals with zeros so ProprioceptiveAnalyzer can run
        if "gripper_width" not in result:
            result["gripper_width"] = np.zeros(T)
        if "joint_velocities" not in result:
            result["joint_velocities"] = np.zeros((T, 7))
        if "ee_velocity" not in result:
            result["ee_velocity"] = np.zeros((T, 3))

    return result


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict keys with '/' separator."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
            out[k] = v  # also store the leaf key for alias matching
    return out

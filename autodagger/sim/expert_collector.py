"""Expert demonstration collection for the AutoDAgger pipeline.

This module handles two kinds of expert data collection:

1. **Initial demos** (round 0) -- the scripted expert solves the task
   from scratch under various scene configurations.
2. **Recovery demos** (round N) -- the scripted expert takes over from
   the exact simulator state where the trained policy failed and
   completes the task, producing corrective data for DAgger.

The scripted expert combines CuRobo motion planning with M2T2 grasp
planning.  Because these components have heavy Isaac-Sim-side
dependencies, the actual expert execution is isolated behind
``_run_scripted_expert`` which carries a TODO for final integration.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

import numpy as np

from autodagger.config import ScriptedExpertConfig
from autodagger.dataset.trajectory import (
    TrajectoryData,
    TrajectoryMetadata,
    TrajectorySource,
)
from autodagger.sim.isaac_interface import IsaacInterface

logger = logging.getLogger(__name__)


class ExpertCollector:
    """Collects demonstrations using the scripted expert in Isaac Sim.

    Usage::

        iface = IsaacInterface(isaac_cfg)
        iface.create_env()
        collector = ExpertCollector(iface, expert_cfg)

        # Round 0 -- initial demos
        demos = collector.collect_initial_demos(num_demos=50, scene_configs=["scene_a"])

        # Round N -- recovery demos from failures
        failures = [
            {"trajectory": failed_traj, "failure_frame": 42},
            ...
        ]
        recovery = collector.collect_recovery_demos_batch(failures)
    """

    def __init__(
        self,
        isaac_interface: IsaacInterface,
        expert_config: ScriptedExpertConfig,
    ) -> None:
        self.interface = isaac_interface
        self.expert_config = expert_config

    # ------------------------------------------------------------------
    # Recovery demonstrations (DAgger rounds)
    # ------------------------------------------------------------------

    def collect_recovery_demo(
        self,
        failed_trajectory: TrajectoryData,
        failure_frame: int,
        dagger_round: int = 1,
    ) -> TrajectoryData:
        """Collect a single recovery demonstration.

        Restores the simulator to the state at *failure_frame* of the
        failed trajectory and then runs the scripted expert to complete
        the task.

        Args:
            failed_trajectory: The trajectory where the policy failed.
            failure_frame: Frame index where failure was detected.
            dagger_round: Current DAgger round number (for metadata).

        Returns:
            A :class:`TrajectoryData` with source=RECOVERY and metadata
            linking back to the original failed trajectory.
        """
        traj_id = str(uuid.uuid4())[:8]
        orig_id = (
            failed_trajectory.metadata.original_trajectory_id
            or f"traj_{id(failed_trajectory)}"
        )

        logger.info(
            "Collecting recovery demo: failure_frame=%d, original_id=%s",
            failure_frame,
            orig_id,
        )

        # Restore simulator to the failure state
        from autodagger.sim.replay import StateReplay

        replayer = StateReplay(self.interface)
        replayer.replay_to_frame(failed_trajectory, failure_frame)

        # Run the scripted expert from this state
        recovery_traj = self._run_scripted_expert(self.interface.env)

        # Attach recovery metadata
        recovery_traj.metadata = TrajectoryMetadata(
            source=TrajectorySource.RECOVERY,
            dagger_round=dagger_round,
            target_stage=None,
            original_trajectory_id=orig_id,
            failure_frame=failure_frame,
            scene_id=failed_trajectory.metadata.scene_id,
        )

        logger.info(
            "Recovery demo collected: %d steps, success=%s (id=%s)",
            recovery_traj.num_steps,
            recovery_traj.success,
            traj_id,
        )

        return recovery_traj

    def collect_recovery_demos_batch(
        self,
        failures: list[dict],
        dagger_round: int = 1,
    ) -> list[TrajectoryData]:
        """Collect recovery demos for a batch of failures.

        Args:
            failures: List of dicts, each with keys:

                * ``trajectory`` (:class:`TrajectoryData`) -- the failed
                  trajectory.
                * ``failure_frame`` (int) -- frame where failure was detected.

            dagger_round: Current DAgger round number.

        Returns:
            List of recovery :class:`TrajectoryData` objects.  Failed
            collections are logged and skipped (the list may be shorter
            than *failures*).
        """
        logger.info(
            "Collecting %d recovery demos for DAgger round %d",
            len(failures),
            dagger_round,
        )

        recovery_demos: list[TrajectoryData] = []
        num_success = 0
        num_failed = 0

        for idx, failure in enumerate(failures):
            trajectory = failure["trajectory"]
            failure_frame = failure["failure_frame"]

            try:
                t0 = time.time()
                demo = self.collect_recovery_demo(
                    trajectory,
                    failure_frame,
                    dagger_round=dagger_round,
                )
                elapsed = time.time() - t0

                if demo.success:
                    recovery_demos.append(demo)
                    num_success += 1
                else:
                    num_failed += 1
                    logger.warning(
                        "Recovery %d/%d: expert failed to complete task, discarding",
                        idx + 1, len(failures),
                    )

                logger.info(
                    "Recovery %d/%d: success=%s, steps=%d, time=%.1fs",
                    idx + 1,
                    len(failures),
                    demo.success,
                    demo.num_steps,
                    elapsed,
                )

            except Exception:
                num_failed += 1
                logger.error(
                    "Recovery %d/%d failed", idx + 1, len(failures),
                    exc_info=True,
                )

        logger.info(
            "Batch recovery complete: %d/%d collected (%d expert-succeeded, "
            "%d expert-failed)",
            len(recovery_demos),
            len(failures),
            num_success,
            num_failed,
        )

        return recovery_demos

    # ------------------------------------------------------------------
    # Initial demonstrations (round 0)
    # ------------------------------------------------------------------

    def collect_initial_demos(
        self,
        num_demos: int,
        scene_configs: Optional[list[str]] = None,
    ) -> list[TrajectoryData]:
        """Collect initial demonstrations using the scripted expert.

        This is used at round 0 before any policy has been trained.
        The expert solves the full task from a fresh reset.

        Args:
            num_demos: Total number of demonstrations to collect.
            scene_configs: Optional list of scene configuration names.
                Demos are distributed evenly across scenes.  If ``None``,
                uses the default scene from the environment.

        Returns:
            List of :class:`TrajectoryData` objects with
            source=SCRIPTED.
        """
        scene_configs = scene_configs or [None]
        demos_per_scene = max(1, num_demos // len(scene_configs))

        logger.info(
            "Collecting %d initial demos across %d scene(s)",
            num_demos,
            len(scene_configs),
        )

        all_demos: list[TrajectoryData] = []
        demo_count = 0

        for scene_id in scene_configs:
            for i in range(demos_per_scene):
                if demo_count >= num_demos:
                    break

                try:
                    t0 = time.time()

                    # Reset the environment with optional scene configuration
                    self._setup_scene(scene_id)
                    obs, extras = self.interface.reset(seed=demo_count)
                    initial_state = self.interface.get_state()

                    # Run the scripted expert
                    demo = self._run_scripted_expert(self.interface.env)

                    # Attach metadata
                    demo.initial_state = _numpy_nested(initial_state)
                    demo.metadata = TrajectoryMetadata(
                        source=TrajectorySource.SCRIPTED,
                        dagger_round=0,
                        scene_id=scene_id,
                    )

                    elapsed = time.time() - t0
                    all_demos.append(demo)
                    demo_count += 1

                    logger.info(
                        "Initial demo %d/%d (scene=%s): success=%s, "
                        "steps=%d, time=%.1fs",
                        demo_count,
                        num_demos,
                        scene_id,
                        demo.success,
                        demo.num_steps,
                        elapsed,
                    )

                except Exception:
                    logger.error(
                        "Failed to collect initial demo %d/%d (scene=%s)",
                        demo_count + 1,
                        num_demos,
                        scene_id,
                        exc_info=True,
                    )

        num_success = sum(1 for d in all_demos if d.success)
        logger.info(
            "Initial demo collection complete: %d/%d collected, %d successful",
            len(all_demos),
            num_demos,
            num_success,
        )

        return all_demos

    # ------------------------------------------------------------------
    # Scripted expert execution
    # ------------------------------------------------------------------

    def _run_scripted_expert(self, env) -> TrajectoryData:
        """Execute the scripted expert (CuRobo + M2T2) in the simulator.

        This method runs the scripted expert from the current simulator
        state until the task is complete or the maximum number of steps
        is reached.

        Args:
            env: Isaac Lab environment instance (already at the desired
                starting state).

        Returns:
            A :class:`TrajectoryData` with the recorded trajectory.

        .. note::

            TODO: Integrate the actual CuRobo motion planner and M2T2
            grasp planner.  The current implementation is a placeholder
            that records the structure but does not execute real expert
            actions.

            Integration steps:

            1. Load CuRobo config from ``self.expert_config.curobo_config_dir``
            2. Load M2T2 model from ``self.expert_config.m2t2_checkpoint``
            3. At each step:
               a. Query M2T2 for grasp pose given current observation
               b. Plan motion to grasp pose with CuRobo
               c. Execute planned trajectory segment
               d. Close gripper, lift, move to placement
            4. Record all (obs, action, state) tuples

            Reference implementation:
            ``sage/server/isaaclab/data_generation_grasp_aug_pose.py``
        """
        # TODO: Replace this placeholder with actual expert implementation.
        # The structure below shows the expected interface.

        max_steps = 500  # Fallback; should come from task config
        all_states: list[dict] = []
        all_obs: list[dict] = []
        all_actions: list[np.ndarray] = []
        success = False

        # Capture initial state
        initial_state = self.interface.get_state()
        all_states.append(_numpy_nested(initial_state))

        # Try to find a usable expert action function
        expert_fn = self._get_expert_action_fn(env)

        for step_idx in range(max_steps):
            # Get expert action
            action = expert_fn(env)
            action_np = _to_numpy(action)
            all_actions.append(action_np)

            # Step environment
            obs, extras = self.interface.step(action)
            state = self.interface.get_state()

            all_states.append(_numpy_nested(state))
            all_obs.append(_detach_obs(obs))

            # Check for task success
            if self.interface.check_success():
                success = True
                break

        # Build trajectory
        actions_array = np.stack(all_actions, axis=0) if all_actions else None
        merged_states = _stack_nested(all_states) if len(all_states) > 1 else {}
        merged_obs = _stack_nested(all_obs) if all_obs else {}

        return TrajectoryData(
            states=merged_states,
            actions=actions_array,
            observations=merged_obs,
            initial_state=_numpy_nested(initial_state),
            success=success,
            metadata=TrajectoryMetadata(source=TrajectorySource.SCRIPTED),
        )

    @staticmethod
    def _get_expert_action_fn(env):
        """Resolve the scripted expert action function.

        Tries several integration paths in order:

        1. Environment's built-in ``get_scripted_action()`` method.
        2. SAGE data generation scripted action pipeline.
        3. Raises with integration instructions.
        """
        # Path 1: environment exposes a scripted action API
        if hasattr(env.unwrapped, "get_scripted_action"):
            logger.debug("Using env.unwrapped.get_scripted_action()")
            return lambda e: e.unwrapped.get_scripted_action()

        # Path 2: environment has a command manager that produces expert actions
        if hasattr(env.unwrapped, "command_manager"):
            logger.debug("Using env.unwrapped.command_manager.compute()")
            return lambda e: e.unwrapped.command_manager.compute()

        # Path 3: no built-in expert found
        raise NotImplementedError(
            "No scripted expert found for this environment. "
            "To integrate the SAGE scripted expert:\n"
            "  1. Import the grasp planner from sage/M2T2\n"
            "  2. Import the motion planner from CuRobo\n"
            "  3. Implement plan_and_execute() pipeline\n"
            "See sage/server/isaaclab/data_generation_grasp_aug_pose.py "
            "for the reference implementation."
        )

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self, scene_id: Optional[str]) -> None:
        """Configure the environment for a specific scene.

        Args:
            scene_id: Scene configuration identifier.  If ``None``, uses
                the default scene.

        .. note::

            TODO: Implement scene-specific configuration.  This may
            involve setting object positions, changing the target
            location, or loading a different USD scene file.
        """
        if scene_id is None:
            return

        # TODO: Apply scene-specific configuration to the environment.
        # Example approaches:
        #   - env.unwrapped.cfg.scene.object_cfg = load_scene(scene_id)
        #   - env.unwrapped.scene.set_object_poses(scene_poses[scene_id])
        logger.debug("Scene setup requested for '%s' (not yet implemented)", scene_id)


# ======================================================================
# Internal helpers
# ======================================================================


def _numpy_nested(d: dict) -> dict:
    """Recursively convert tensors to numpy arrays."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _numpy_nested(v)
        elif hasattr(v, "detach"):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = np.asarray(v)
    return out


def _to_numpy(tensor) -> np.ndarray:
    """Convert a torch Tensor or numpy array to numpy."""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _detach_obs(obs) -> dict:
    """Detach and convert observation tensors to numpy."""
    if isinstance(obs, dict):
        return {k: _detach_obs(v) for k, v in obs.items()}
    return _to_numpy(obs)


def _stack_nested(dicts: list[dict]) -> dict:
    """Stack matching keys across a list of nested dicts along axis 0."""
    if not dicts:
        return {}
    out: dict = {}
    for k in dicts[0]:
        vals = [d[k] for d in dicts if k in d]
        if not vals:
            continue
        if isinstance(vals[0], dict):
            out[k] = _stack_nested(vals)
        else:
            out[k] = np.stack([np.asarray(v) for v in vals], axis=0)
    return out

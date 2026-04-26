"""Policy evaluation in Isaac Sim for the AutoDAgger pipeline."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from autodagger.config import EvalConfig, IsaacSimConfig
from autodagger.dataset.trajectory import (
    TrajectoryData,
    TrajectoryMetadata,
    TrajectorySource,
)

try:
    import omni.isaac.lab  # noqa: F401

    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """Runs trained policies in Isaac Sim and collects evaluation metrics.

    Usage::

        evaluator = PolicyEvaluator(eval_cfg, isaac_cfg)
        results = evaluator.evaluate(checkpoint_path, output_dir)
        print(results["success_rate"])
    """

    def __init__(self, eval_config: EvalConfig, isaac_config: IsaacSimConfig) -> None:
        self.eval_config = eval_config
        self.isaac_config = isaac_config
        self._env = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, checkpoint_path: str, output_dir: str) -> dict:
        """Run full evaluation: load policy, roll out episodes, save results.

        Args:
            checkpoint_path: Path to trained policy checkpoint (Robomimic or
                raw PyTorch).
            output_dir: Directory where trajectory HDF5 files and optional
                rendered frames will be saved.

        Returns:
            Summary dict with keys: success_rate, avg_steps, std_steps,
            num_episodes, num_successes, num_failures,
            success_trajectories_path, failure_trajectories_path.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting evaluation: %d episodes, checkpoint=%s",
            self.eval_config.num_episodes,
            checkpoint_path,
        )

        # Load policy
        policy = self._load_policy(checkpoint_path)

        # Create environment
        env = self._create_env()

        # Collect rollouts
        trajectories: list[TrajectoryData] = []
        successes = 0
        step_counts: list[int] = []

        try:
            for ep_idx in range(self.eval_config.num_episodes):
                t0 = time.time()
                traj = self._run_episode(env, policy, ep_idx)
                elapsed = time.time() - t0

                trajectories.append(traj)
                step_counts.append(traj.num_steps)
                if traj.success:
                    successes += 1

                logger.info(
                    "Episode %d/%d: success=%s, steps=%d, time=%.1fs",
                    ep_idx + 1,
                    self.eval_config.num_episodes,
                    traj.success,
                    traj.num_steps,
                    elapsed,
                )
        finally:
            self._close_env()

        # Compute summary
        num_episodes = len(trajectories)
        success_rate = successes / num_episodes if num_episodes > 0 else 0.0
        avg_steps = float(np.mean(step_counts)) if step_counts else 0.0
        std_steps = float(np.std(step_counts)) if step_counts else 0.0

        # Save trajectories to HDF5
        success_path = ""
        failure_path = ""
        if self.eval_config.save_trajectories:
            success_path, failure_path = self._save_trajectories(
                trajectories, output_dir
            )

        summary = {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "std_steps": std_steps,
            "num_episodes": num_episodes,
            "num_successes": successes,
            "num_failures": num_episodes - successes,
            "success_trajectories_path": success_path,
            "failure_trajectories_path": failure_path,
        }

        logger.info(
            "Evaluation complete: success_rate=%.2f (%d/%d), avg_steps=%.1f",
            success_rate,
            successes,
            num_episodes,
            avg_steps,
        )

        return summary

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------

    def _create_env(self):
        """Create an Isaac Lab environment from the stored config.

        Returns:
            The unwrapped ManagerBasedEnv instance.

        Raises:
            RuntimeError: If Isaac Sim / Isaac Lab is not available.
        """
        if not ISAAC_AVAILABLE:
            raise RuntimeError(
                "Isaac Sim is not available in this environment. "
                "Please run inside an Isaac Sim Python environment "
                "(e.g. ~/.local/share/ov/pkg/isaac-sim-*/python.sh) "
                "or install omni.isaac.lab."
            )

        try:
            import gymnasium as gym

            # TODO: Isaac Lab environment registration may require importing
            # the task module first, e.g.:
            #   import omni.isaac.lab_tasks  # noqa: F401
            # Adjust the import below to match the actual task package used
            # in the sage project.
            import omni.isaac.lab_tasks  # noqa: F401

            env = gym.make(
                self.isaac_config.task,
                num_envs=self.isaac_config.num_envs,
                device=self.isaac_config.device,
            ).env

            self._env = env
            logger.info(
                "Created Isaac Lab env: task=%s, num_envs=%d, device=%s",
                self.isaac_config.task,
                self.isaac_config.num_envs,
                self.isaac_config.device,
            )
            return env

        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Isaac Lab environment "
                f"(task={self.isaac_config.task}): {exc}"
            ) from exc

    def _close_env(self) -> None:
        """Clean shutdown of the Isaac Lab environment."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                logger.warning("Error closing environment", exc_info=True)
            self._env = None

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def _load_policy(self, checkpoint_path: str):
        """Load a trained policy from a checkpoint file.

        Supports two formats:
        1. Robomimic checkpoint (contains 'algo_name' key) -- uses
           ``robomimic.utils.file_utils.policy_from_checkpoint``.
        2. Raw PyTorch state dict -- loads with ``torch.load``.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            A callable policy object that maps observation dicts to actions.
        """
        import torch

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        logger.info("Loading policy from %s", checkpoint_path)

        # Attempt Robomimic format first
        try:
            import robomimic.utils.file_utils as FileUtils

            policy, _ = FileUtils.policy_from_checkpoint(
                ckpt_path=checkpoint_path,
                device=self.isaac_config.device,
                verbose=False,
            )
            policy.start_episode()
            logger.info("Loaded Robomimic policy from checkpoint")
            return policy
        except Exception:
            logger.debug(
                "Not a Robomimic checkpoint, falling back to raw torch.load",
                exc_info=True,
            )

        # Fall back to raw PyTorch checkpoint
        # TODO: Adapt this to the actual model class used in the sage project.
        # For now we load the state dict and wrap it in a minimal interface.
        ckpt = torch.load(checkpoint_path, map_location=self.isaac_config.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            logger.info("Loaded raw PyTorch checkpoint (dict with 'model' key)")
            return _RawTorchPolicy(ckpt["model"], self.isaac_config.device)
        elif isinstance(ckpt, torch.nn.Module):
            logger.info("Loaded raw PyTorch nn.Module checkpoint")
            ckpt.eval()
            return _RawTorchPolicy(ckpt, self.isaac_config.device)
        else:
            raise ValueError(
                f"Unrecognised checkpoint format at {checkpoint_path}. "
                "Expected a Robomimic checkpoint or a dict with a 'model' key."
            )

    # ------------------------------------------------------------------
    # Episode rollout
    # ------------------------------------------------------------------

    def _run_episode(self, env, policy, episode_idx: int) -> TrajectoryData:
        """Execute a single evaluation episode.

        Args:
            env: Isaac Lab environment instance.
            policy: A callable policy (obs -> action).
            episode_idx: Index used for logging / seeding.

        Returns:
            A :class:`TrajectoryData` containing the full episode data.
        """
        max_steps = self.eval_config.max_steps

        # Reset environment
        obs, extras = env.reset()
        initial_state = env.unwrapped.scene.get_state(is_relative=True)

        all_states: list[dict] = [initial_state]
        all_obs: list[dict] = [_detach_obs(obs)]
        all_actions: list = []
        rendered_frames: list = []

        success = False

        for step_idx in range(max_steps):
            # Get action from policy
            action = self._get_action(policy, obs)
            all_actions.append(_to_numpy(action))

            # Step environment
            obs, extras = env.step(action)

            # Record state and observation
            state = env.unwrapped.scene.get_state(is_relative=True)
            all_states.append(state)
            all_obs.append(_detach_obs(obs))

            # Optionally capture rendered frame
            if self.eval_config.save_rendered_frames:
                # TODO: Implement frame capture from Isaac Sim viewport.
                # This depends on the rendering pipeline configuration.
                pass

            # Check termination
            try:
                success_tensor = env.unwrapped.termination_manager.get_term(
                    "success"
                )
                if success_tensor.any().item():
                    success = True
                    break
            except Exception:
                # Termination manager may not have a "success" term for all
                # tasks.  Fall back to checking extras.
                if extras.get("terminated", False) or extras.get("truncated", False):
                    success = extras.get("success", False)
                    break

        # Build trajectory
        actions_array = np.stack(all_actions, axis=0) if all_actions else None
        merged_states = _merge_state_list(all_states)
        merged_obs = _merge_obs_list(all_obs)

        trajectory = TrajectoryData(
            states=merged_states,
            actions=actions_array,
            observations=merged_obs,
            initial_state=_numpy_nested(initial_state),
            success=success,
            seed=episode_idx,
            metadata=TrajectoryMetadata(
                source=TrajectorySource.SCRIPTED,
                dagger_round=-1,  # Evaluation, not a training round
            ),
        )

        return trajectory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_action(policy, obs):
        """Query the policy for an action given an observation.

        Handles both Robomimic-style policies (with ``__call__``) and raw
        torch modules.
        """
        # Robomimic policies expose __call__(obs) directly
        if hasattr(policy, "__call__") and not isinstance(policy, _RawTorchPolicy):
            return policy(obs)

        # _RawTorchPolicy wrapper
        if isinstance(policy, _RawTorchPolicy):
            return policy(obs)

        raise TypeError(f"Unsupported policy type: {type(policy)}")

    def _save_trajectories(
        self, trajectories: list[TrajectoryData], output_dir: str
    ) -> tuple[str, str]:
        """Save trajectories to HDF5, split by success/failure.

        Returns (success_path, failure_path).
        """
        from autodagger.dataset.hdf5_manager import HDF5Manager
        from autodagger.config import DatasetConfig

        hdf5 = HDF5Manager(DatasetConfig())

        success_trajs = [t for t in trajectories if t.success]
        failure_trajs = [t for t in trajectories if not t.success]

        success_path = os.path.join(output_dir, "eval_success.hdf5")
        failure_path = os.path.join(output_dir, "eval_failure.hdf5")

        if success_trajs:
            hdf5.append_trajectories(success_path, success_trajs)
            logger.info(
                "Saved %d successful trajectories to %s",
                len(success_trajs),
                success_path,
            )
        if failure_trajs:
            hdf5.append_trajectories(failure_path, failure_trajs)
            logger.info(
                "Saved %d failed trajectories to %s",
                len(failure_trajs),
                failure_path,
            )

        return success_path, failure_path


# ======================================================================
# Internal helper classes / functions
# ======================================================================


class _RawTorchPolicy:
    """Minimal wrapper around a raw PyTorch model to provide a policy interface."""

    def __init__(self, model, device: str) -> None:
        import torch

        if isinstance(model, dict):
            # State dict -- caller must provide the actual model architecture.
            # TODO: Instantiate the correct model class and load state dict.
            raise NotImplementedError(
                "Loading from a raw state dict requires specifying the model "
                "architecture. Implement _RawTorchPolicy with the correct "
                "model class for the sage project."
            )
        self.model = model
        self.device = device
        self.model.eval()

    def __call__(self, obs):
        import torch

        with torch.no_grad():
            # TODO: Adapt observation pre-processing to match the model's
            # expected input format.
            if isinstance(obs, dict):
                obs_tensor = {
                    k: (
                        v.to(self.device)
                        if isinstance(v, torch.Tensor)
                        else torch.tensor(v, device=self.device, dtype=torch.float32)
                    )
                    for k, v in obs.items()
                }
            else:
                obs_tensor = obs

            action = self.model(obs_tensor)
        return action


def _to_numpy(tensor) -> np.ndarray:
    """Convert a torch Tensor or numpy array to a numpy array."""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _detach_obs(obs) -> dict:
    """Detach and convert observation tensors to numpy."""
    if isinstance(obs, dict):
        return {k: _detach_obs(v) for k, v in obs.items()}
    return _to_numpy(obs)


def _numpy_nested(d: dict) -> dict:
    """Recursively convert all tensors in a nested dict to numpy arrays."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _numpy_nested(v)
        else:
            out[k] = _to_numpy(v)
    return out


def _merge_state_list(states: list[dict]) -> dict:
    """Stack a list of state dicts (from successive frames) into arrays.

    Each element of *states* is a nested dict of scalars/1-D arrays from a
    single time step.  The result is a nested dict where each leaf is a
    2-D array with shape ``(T, ...)``.
    """
    if not states:
        return {}
    return _stack_nested([_numpy_nested(s) for s in states])


def _merge_obs_list(obs_list: list[dict]) -> dict:
    """Stack a list of observation dicts into arrays along axis 0."""
    if not obs_list:
        return {}
    return _stack_nested(obs_list)


def _stack_nested(dicts: list[dict]) -> dict:
    """Recursively stack matching keys across a list of dicts."""
    out: dict = {}
    keys = dicts[0].keys()
    for k in keys:
        vals = [d[k] for d in dicts]
        if isinstance(vals[0], dict):
            out[k] = _stack_nested(vals)
        else:
            out[k] = np.stack([np.asarray(v) for v in vals], axis=0)
    return out

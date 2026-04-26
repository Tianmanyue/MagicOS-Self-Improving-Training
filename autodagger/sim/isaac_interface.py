"""Isaac Sim / Isaac Lab interface for the AutoDAgger pipeline.

This module provides a thin wrapper around the Isaac Lab ManagerBasedEnv
API, centralising environment creation, state get/set, stepping, and
shutdown.  All Isaac-specific imports are lazy so the module can be
imported on machines without Isaac Sim (methods will raise at call time).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from autodagger.config import IsaacSimConfig

try:
    import omni.isaac.lab  # noqa: F401

    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

logger = logging.getLogger(__name__)


def _require_isaac() -> None:
    """Raise a helpful error if Isaac Sim is not available."""
    if not ISAAC_AVAILABLE:
        raise RuntimeError(
            "Isaac Sim is not available in this environment. "
            "Please run inside an Isaac Sim Python environment "
            "(e.g. ~/.local/share/ov/pkg/isaac-sim-*/python.sh) "
            "or install omni.isaac.lab."
        )


class IsaacInterface:
    """Unified interface for Isaac Lab environment interactions.

    This class owns the environment lifetime and exposes the core
    operations needed by the rest of the AutoDAgger pipeline:

    * ``create_env`` -- spin up a ManagerBasedEnv
    * ``get_state``  -- snapshot simulator state (relative coordinates)
    * ``set_state``  -- restore simulator to a prior state
    * ``step``       -- advance the simulation one action step
    * ``reset``      -- reset the environment
    * ``close``      -- clean shutdown

    Example::

        iface = IsaacInterface(isaac_cfg)
        env = iface.create_env()
        obs, extras = iface.reset()
        state = iface.get_state()
        obs, extras = iface.step(action)
        iface.set_state(state)
        iface.close()
    """

    def __init__(self, isaac_config: IsaacSimConfig) -> None:
        self.config = isaac_config
        self.env = None
        self._is_open = False

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------

    def create_env(self):
        """Create and return an Isaac Lab ManagerBasedEnv.

        The environment is cached on ``self.env`` for reuse.  Calling this
        method again while the previous environment is still open will
        close it first.

        Returns:
            The Isaac Lab environment (unwrapped ManagerBasedEnv).
        """
        _require_isaac()

        # Close any existing environment
        if self._is_open:
            self.close()

        try:
            import gymnasium as gym

            # TODO: The task registration import may need adjustment depending
            # on the exact Isaac Lab version and task package layout used in
            # the sage project.
            import omni.isaac.lab_tasks  # noqa: F401

            env = gym.make(
                self.config.task,
                num_envs=self.config.num_envs,
                device=self.config.device,
            ).env

            self.env = env
            self._is_open = True

            logger.info(
                "Created Isaac Lab env: task=%s, num_envs=%d, device=%s",
                self.config.task,
                self.config.num_envs,
                self.config.device,
            )
            return env

        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Isaac Lab environment "
                f"(task={self.config.task}): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_state(self, env_ids=None) -> dict:
        """Get the current simulator state in relative coordinates.

        Args:
            env_ids: Optional tensor of environment indices.  If ``None``,
                returns state for all environments.

        Returns:
            Nested dict of tensors representing the scene state.
        """
        self._require_env()
        state = self.env.unwrapped.scene.get_state(is_relative=True)

        if env_ids is not None:
            state = _index_nested(state, env_ids)

        return state

    def set_state(self, state: dict, env_ids=None) -> None:
        """Restore the simulator to a previously captured state.

        Args:
            state: Nested dict of tensors (same structure as returned by
                :meth:`get_state`).
            env_ids: Optional tensor of environment indices to reset.  If
                ``None``, resets all environments.
        """
        self._require_env()

        if env_ids is None:
            # Build env_ids tensor for all environments
            import torch

            env_ids = torch.arange(
                self.config.num_envs,
                device=self.config.device,
            )

        self.env.unwrapped.reset_to(state, env_ids, is_relative=True)
        logger.debug("Set state for env_ids=%s", env_ids)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, action) -> tuple:
        """Advance the simulation by one action step.

        Args:
            action: Action tensor with shape ``(num_envs, action_dim)``.

        Returns:
            Tuple of ``(obs, extras)`` from the underlying environment.
        """
        self._require_env()
        obs, extras = self.env.step(action)
        return obs, extras

    def reset(self, seed: Optional[int] = None) -> tuple:
        """Reset the environment.

        Args:
            seed: Optional seed for reproducibility.

        Returns:
            Tuple of ``(obs, extras)`` from the underlying environment.
        """
        self._require_env()
        if seed is not None:
            obs, extras = self.env.reset(seed=seed)
        else:
            obs, extras = self.env.reset()
        return obs, extras

    # ------------------------------------------------------------------
    # Termination checks
    # ------------------------------------------------------------------

    def check_success(self) -> bool:
        """Check whether the current episode has succeeded.

        Returns:
            True if any environment reports success.
        """
        self._require_env()
        try:
            success_tensor = self.env.unwrapped.termination_manager.get_term(
                "success"
            )
            return bool(success_tensor.any().item())
        except Exception:
            logger.debug(
                "Could not query 'success' termination term", exc_info=True
            )
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the Isaac Lab environment and release resources."""
        if self.env is not None:
            try:
                self.env.close()
                logger.info("Isaac Lab environment closed")
            except Exception:
                logger.warning("Error closing Isaac Lab environment", exc_info=True)
            self.env = None
        self._is_open = False

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_env(self) -> None:
        """Raise if the environment has not been created yet."""
        if self.env is None:
            raise RuntimeError(
                "Environment not initialised. Call create_env() first."
            )


# ======================================================================
# Module-level helpers
# ======================================================================


def _index_nested(d: dict, indices) -> dict:
    """Index into every leaf tensor of a nested dict."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _index_nested(v, indices)
        elif hasattr(v, "__getitem__"):
            out[k] = v[indices]
        else:
            out[k] = v
    return out

"""Robomimic training wrapper for the AutoDAgger pipeline.

Generates Robomimic-compatible JSON configs from TrainingConfig,
launches training as a subprocess, and locates the best checkpoint
after training completes.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Optional

from autodagger.config import TrainingConfig

logger = logging.getLogger(__name__)

# Robomimic entry-point script, resolved relative to this repo.
_ROBOMIMIC_TRAIN_SCRIPT = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "robomimic", "robomimic", "scripts", "train.py",
)


class RobomimicTrainer:
    """Wraps Robomimic's BC / Diffusion Policy training for AutoDAgger."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def generate_config(
        self,
        dataset_path: str,
        obs_keys: list[str],
        action_dim: int,
        round_num: int = 0,
    ) -> dict:
        """Build a full Robomimic JSON config dict from *TrainingConfig*.

        Parameters
        ----------
        dataset_path:
            Path to the HDF5 dataset for training.
        obs_keys:
            List of low-dimensional observation key names.
        action_dim:
            Dimensionality of the action space (used for network sizing).
        round_num:
            Current DAgger round; appended to the output directory so that
            successive rounds do not overwrite each other.

        Returns
        -------
        dict
            A config dict ready to be saved to JSON and consumed by
            ``robomimic/scripts/train.py``.
        """
        cfg = self.config
        output_dir = os.path.join(cfg.output_dir, f"round_{round_num}")

        config_dict: dict = {
            "algo_name": cfg.algo,
            "experiment": {
                "name": f"autodagger_round{round_num}",
                "validate": True,
                "save": {
                    "enabled": True,
                    "every_n_epochs": 50,
                    "on_best_validation": True,
                    "on_best_rollout_return": False,
                    "on_best_rollout_success_rate": False,
                },
                "rollout": {"enabled": False},
            },
            "train": {
                "data": dataset_path,
                "output_dir": output_dir,
                "batch_size": cfg.batch_size,
                "num_epochs": cfg.num_epochs,
                "seed": cfg.seed,
                "cuda": cfg.cuda,
                "seq_length": cfg.seq_length,
                "hdf5_cache_mode": "all",
            },
            "observation": {
                "modalities": {
                    "obs": {
                        "low_dim": obs_keys,
                        "rgb": [],
                        "depth": [],
                        "scan": [],
                    }
                },
            },
        }

        # Algorithm-specific blocks ----------------------------------------
        if cfg.algo == "bc":
            config_dict["algo"] = {
                "optim_params": {
                    "policy": {
                        "optimizer_type": "adam",
                        "learning_rate": {
                            "initial": cfg.learning_rate,
                            "decay_factor": 0.1,
                            "epoch_schedule": [],
                            "scheduler_type": "multistep",
                        },
                    },
                },
                "actor_layer_dims": list(cfg.actor_layer_dims),
                "gmm": {
                    "enabled": False,
                    "num_modes": 5,
                    "min_std": 0.0001,
                },
                "rnn": {"enabled": False},
            }

        elif cfg.algo == "diffusion_policy":
            config_dict["algo"] = {
                "optim_params": {
                    "policy": {
                        "optimizer_type": "adamw",
                        "learning_rate": {
                            "initial": cfg.learning_rate,
                            "decay_factor": 0.1,
                            "epoch_schedule": [],
                            "scheduler_type": "multistep",
                        },
                    },
                },
                "horizon": {
                    "observation_horizon": 2,
                    "action_horizon": 8,
                    "prediction_horizon": 16,
                },
                "unet": {
                    "enabled": True,
                    "diffusion_step_embed_dim": 256,
                    "down_dims": [256, 512, 1024],
                    "kernel_size": 5,
                    "n_groups": 8,
                },
                "noise_scheduler": {
                    "type": "ddpm",
                    "num_train_timesteps": 100,
                    "beta_schedule": "squaredcos_cap_v2",
                },
                "action_dim": action_dim,
            }

        else:
            logger.warning(
                "Unknown algo '%s'; producing a minimal config. "
                "You may need to populate algo-specific fields manually.",
                cfg.algo,
            )
            config_dict["algo"] = {
                "optim_params": {
                    "policy": {
                        "learning_rate": {
                            "initial": cfg.learning_rate,
                        },
                    },
                },
            }

        # If the user supplied a template, merge it (template values serve
        # as defaults; anything we set above takes priority).
        if cfg.config_template and os.path.isfile(cfg.config_template):
            config_dict = _merge_template(cfg.config_template, config_dict)

        return config_dict

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_config(config_dict: dict, path: str) -> None:
        """Serialize *config_dict* to a JSON file at *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info("Saved Robomimic config to %s", path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_path: str,
        obs_keys: list[str],
        action_dim: int,
        round_num: int = 0,
    ) -> str:
        """Generate a config, save it, run training, return best checkpoint.

        Raises
        ------
        RuntimeError
            If the training subprocess exits with a non-zero code.
        FileNotFoundError
            If no checkpoint is found after training finishes.
        """
        config_dict = self.generate_config(
            dataset_path, obs_keys, action_dim, round_num
        )

        output_dir = config_dict["train"]["output_dir"]
        config_path = os.path.join(output_dir, "config.json")
        self.save_config(config_dict, config_path)

        return self.train_from_config(config_path)

    def train_from_config(self, config_path: str) -> str:
        """Launch training from an existing JSON config file.

        Returns
        -------
        str
            Path to the best checkpoint produced by the run.
        """
        train_script = os.path.normpath(_ROBOMIMIC_TRAIN_SCRIPT)
        if not os.path.isfile(train_script):
            raise FileNotFoundError(
                f"Robomimic train script not found at {train_script}. "
                "Check that the robomimic package is installed alongside sage."
            )

        cmd = [sys.executable, train_script, "--config", config_path]
        logger.info("Launching Robomimic training: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(
                "Robomimic training failed (exit %d).\nstdout:\n%s\nstderr:\n%s",
                result.returncode,
                result.stdout[-2000:] if result.stdout else "",
                result.stderr[-2000:] if result.stderr else "",
            )
            raise RuntimeError(
                f"Robomimic training exited with code {result.returncode}. "
                f"See logs for details."
            )

        logger.info("Training completed successfully.")

        # Resolve the output_dir from the config to find the checkpoint.
        with open(config_path) as f:
            saved_cfg = json.load(f)
        output_dir = saved_cfg["train"]["output_dir"]

        return self.get_best_checkpoint(output_dir)

    # ------------------------------------------------------------------
    # Checkpoint resolution
    # ------------------------------------------------------------------

    @staticmethod
    def get_best_checkpoint(output_dir: str) -> str:
        """Locate the best model checkpoint inside *output_dir*.

        Search order:
        1. ``models/model_best_validation*.pth``
        2. ``models/model_best_training*.pth``
        3. ``models/model_epoch_*.pth`` (highest epoch number)
        4. Any ``.pth`` file under *output_dir* (recursive)

        Returns
        -------
        str
            Absolute path to the selected checkpoint.

        Raises
        ------
        FileNotFoundError
            If no checkpoint could be found.
        """
        models_dir = os.path.join(output_dir, "models")

        # 1. Best-validation checkpoint
        best_val = sorted(glob(os.path.join(models_dir, "model_best_validation*.pth")))
        if best_val:
            logger.info("Using best-validation checkpoint: %s", best_val[-1])
            return best_val[-1]

        # 2. Best-training checkpoint
        best_train = sorted(glob(os.path.join(models_dir, "model_best_training*.pth")))
        if best_train:
            logger.info("Using best-training checkpoint: %s", best_train[-1])
            return best_train[-1]

        # 3. Latest epoch checkpoint
        epoch_ckpts = sorted(glob(os.path.join(models_dir, "model_epoch_*.pth")))
        if epoch_ckpts:
            logger.info("Using latest epoch checkpoint: %s", epoch_ckpts[-1])
            return epoch_ckpts[-1]

        # 4. Fallback: any .pth under the output tree
        any_ckpt = sorted(glob(os.path.join(output_dir, "**", "*.pth"), recursive=True))
        if any_ckpt:
            logger.warning(
                "No standard checkpoint found; falling back to %s", any_ckpt[-1]
            )
            return any_ckpt[-1]

        raise FileNotFoundError(
            f"No checkpoint (.pth) found under {output_dir}. "
            "Training may have failed silently."
        )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _merge_template(template_path: str, overrides: dict) -> dict:
    """Deep-merge *overrides* on top of a JSON template file.

    Values in *overrides* win over values in the template.
    """
    with open(template_path) as f:
        base = json.load(f)
    return _deep_merge(base, overrides)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* (in-place), returning *base*."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base

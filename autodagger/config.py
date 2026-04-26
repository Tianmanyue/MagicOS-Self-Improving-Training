"""Central configuration for the AutoDAgger pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class AnthropicConfig:
    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.3
    max_retries: int = 3


@dataclass
class IsaacSimConfig:
    task: str = "Isaac-Mobile-Manipulation-Obj-Scene-v0"
    num_envs: int = 1
    device: str = "cuda:0"
    episode_length_s: float = 5.0
    decimation: int = 2
    physics_dt: float = 0.01


@dataclass
class ScriptedExpertConfig:
    """Config for the scripted expert (CuRobo + M2T2)."""
    curobo_config_dir: str = ""
    m2t2_checkpoint: str = ""
    grasp_offset: float = 0.0
    approach_distance: float = 0.15
    lift_height: float = 0.15


@dataclass
class StageAnnotationConfig:
    coarse_num_frames: int = 5
    gripper_change_threshold: float = 0.01
    joint_velocity_threshold: float = 0.5
    ee_velocity_zero_threshold: float = 0.02
    vlm_max_refinement_calls: int = 5
    vlm_model: str = "claude-sonnet-4-20250514"


@dataclass
class ProgressModelConfig:
    method: str = "top_reward"  # "top_reward" or "sarm"
    stuck_window: int = 30
    stuck_threshold: float = 0.02
    vlm_model: str = "claude-sonnet-4-20250514"
    # SARM-specific (Phase 2)
    sarm_checkpoint: str = ""
    sarm_num_stages: int = 5


@dataclass
class TrainingConfig:
    algo: str = "bc"  # "bc", "diffusion_policy", "act"
    config_template: str = ""
    output_dir: str = "./trained_models"
    batch_size: int = 100
    num_epochs: int = 200
    learning_rate: float = 1e-4
    actor_layer_dims: tuple = (1024, 1024)
    seq_length: int = 1
    seed: int = 42
    cuda: bool = True
    # RA-BC
    use_ra_bc: bool = False
    ra_bc_delta: int = 10
    ra_bc_normalize: bool = True


@dataclass
class EvalConfig:
    num_episodes: int = 100
    render: bool = False
    save_trajectories: bool = True
    save_rendered_frames: bool = False
    success_threshold: float = 0.02
    max_steps: int = 500


@dataclass
class DatasetConfig:
    base_dir: str = "./datasets"
    filename_prefix: str = "autodagger"
    export_mode: str = "all"  # "all", "succeeded_only", "split"


@dataclass
class DAggerConfig:
    max_rounds: int = 10
    target_success_rate: float = 0.90
    min_improvement: float = 0.02
    max_recovery_demos_per_round: int = 200
    recovery_from_failure_only: bool = True


@dataclass
class PipelineConfig:
    """Top-level config aggregating all sub-configs."""
    project_name: str = "autodagger_pick_and_place"
    work_dir: str = "./autodagger_runs"
    log_dir: str = ""

    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    isaac_sim: IsaacSimConfig = field(default_factory=IsaacSimConfig)
    expert: ScriptedExpertConfig = field(default_factory=ScriptedExpertConfig)
    stage_annotation: StageAnnotationConfig = field(default_factory=StageAnnotationConfig)
    progress: ProgressModelConfig = field(default_factory=ProgressModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dagger: DAggerConfig = field(default_factory=DAggerConfig)

    def __post_init__(self):
        if not self.log_dir:
            self.log_dir = os.path.join(self.work_dir, "logs")

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**{
            k: _reconstruct(k, v) for k, v in d.items()
        })


_FIELD_MAP = {
    "anthropic": AnthropicConfig,
    "isaac_sim": IsaacSimConfig,
    "expert": ScriptedExpertConfig,
    "stage_annotation": StageAnnotationConfig,
    "progress": ProgressModelConfig,
    "training": TrainingConfig,
    "eval": EvalConfig,
    "dataset": DatasetConfig,
    "dagger": DAggerConfig,
}


def _reconstruct(key: str, value):
    if key in _FIELD_MAP and isinstance(value, dict):
        cls = _FIELD_MAP[key]
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in value.items() if k in valid_fields}
        return cls(**filtered)
    return value

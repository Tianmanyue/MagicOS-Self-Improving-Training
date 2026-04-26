#!/usr/bin/env python3
"""Run the full AutoDAgger pipeline end-to-end.

Usage:
    python -m autodagger.scripts.run_full_pipeline \
        --config pipeline_config.json \
        --task "pick and place red block" \
        --obs-keys joint_pos,joint_vel,object_position,target_object_position \
        --action-dim 8 \
        [--initial-demos path/to/demos.hdf5] \
        [--num-initial-demos 200]
"""

from __future__ import annotations

import argparse
import json
import logging

from autodagger.config import PipelineConfig
from autodagger.pipeline import AutoDAggerPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full AutoDAgger pipeline.")
    parser.add_argument("--config", required=True, help="Path to pipeline config JSON")
    parser.add_argument("--task", required=True, help="Task description for VLM prompts")
    parser.add_argument("--obs-keys", required=True, help="Comma-separated observation keys")
    parser.add_argument("--action-dim", type=int, required=True, help="Action dimensionality")
    parser.add_argument("--initial-demos", default=None, help="Path to pre-existing initial demos")
    parser.add_argument("--num-initial-demos", type=int, default=200, help="Demos to collect if no initial path")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config)
    pipeline = AutoDAggerPipeline(config)

    obs_keys = [k.strip() for k in args.obs_keys.split(",")]

    result = pipeline.run(
        obs_keys=obs_keys,
        action_dim=args.action_dim,
        task_description=args.task,
        initial_demos_path=args.initial_demos,
        num_initial_demos=args.num_initial_demos,
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

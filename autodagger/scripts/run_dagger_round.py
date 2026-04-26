#!/usr/bin/env python3
"""Run a single DAgger round manually — useful for Phase 1 step-by-step verification.

Usage:
    python -m autodagger.scripts.run_dagger_round \
        --config pipeline_config.json \
        --round 1 \
        --checkpoint trained_models/round_0/models/model_best_validation.pth \
        --task "pick and place red block" \
        --obs-keys joint_pos,joint_vel,object_position,target_object_position \
        --action-dim 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from autodagger.config import PipelineConfig
from autodagger.pipeline import AutoDAggerPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single AutoDAgger round.")
    parser.add_argument("--config", required=True, help="Path to pipeline config JSON")
    parser.add_argument("--round", type=int, required=True, help="DAgger round number")
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint from previous round")
    parser.add_argument("--task", required=True, help="Task description for VLM")
    parser.add_argument("--obs-keys", required=True, help="Comma-separated observation key names")
    parser.add_argument("--action-dim", type=int, required=True, help="Action dimensionality")
    parser.add_argument("--initial-demos", default=None, help="Path to initial demos (round 0 only)")
    args = parser.parse_args()

    config = PipelineConfig.load(args.config)
    pipeline = AutoDAggerPipeline(config)
    obs_keys = [k.strip() for k in args.obs_keys.split(",")]
    round_num = args.round

    logger.info("=== Manual DAgger Round %d ===", round_num)

    # Step 1: Evaluate current checkpoint
    logger.info("Step 1: Evaluating checkpoint %s", args.checkpoint)
    eval_summary = pipeline.step_evaluate(args.checkpoint, round_num - 1)
    logger.info("Success rate: %.1f%%", eval_summary["success_rate"] * 100)

    # Step 2: Analyze failures
    failure_path = eval_summary.get("failure_path")
    if not failure_path:
        logger.info("No failures to analyze. Done.")
        return

    logger.info("Step 2: Analyzing failures from %s", failure_path)
    failures = pipeline.step_analyze(failure_path, args.task)
    logger.info("Found %d failure cases", len(failures))

    if not failures:
        logger.info("No recoverable failures. Done.")
        return

    # Step 3: Collect recovery demos
    logger.info("Step 3: Collecting recovery demonstrations")
    recovery_demos = pipeline.step_collect_recovery(failures, round_num)
    logger.info("Collected %d recovery demos", len(recovery_demos))

    if not recovery_demos:
        logger.info("No recovery demos collected. Done.")
        return

    # Step 4: Merge dataset and retrain
    logger.info("Step 4: Merging dataset and retraining")
    cumulative_path = pipeline.step_merge_and_version(round_num, recovery_demos)
    logger.info("Cumulative dataset: %s", cumulative_path)

    checkpoint = pipeline.step_train(
        cumulative_path, obs_keys, args.action_dim, round_num
    )
    logger.info("New checkpoint: %s", checkpoint)

    # Step 5: Evaluate new checkpoint
    logger.info("Step 5: Evaluating new policy")
    new_eval = pipeline.step_evaluate(checkpoint, round_num)
    logger.info(
        "Round %d: %.1f%% -> %.1f%%",
        round_num,
        eval_summary["success_rate"] * 100,
        new_eval["success_rate"] * 100,
    )

    result = {
        "round": round_num,
        "previous_success_rate": eval_summary["success_rate"],
        "new_success_rate": new_eval["success_rate"],
        "improvement": new_eval["success_rate"] - eval_summary["success_rate"],
        "num_recovery_demos": len(recovery_demos),
        "checkpoint": checkpoint,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

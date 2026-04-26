#!/usr/bin/env python3
"""Generate a default pipeline config JSON file.

Usage:
    python -m autodagger.scripts.create_default_config [--output config.json]
"""

from __future__ import annotations

import argparse
from autodagger.config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Create default AutoDAgger config.")
    parser.add_argument("--output", default="autodagger_config.json", help="Output path")
    parser.add_argument("--task", default="pick_and_place", choices=["pick_and_place", "mobile_manipulation"])
    parser.add_argument("--api-key", default="", help="Anthropic API key")
    args = parser.parse_args()

    config = PipelineConfig()

    if args.task == "pick_and_place":
        config.project_name = "autodagger_pick_and_place"
        config.isaac_sim.task = "Isaac-Franka-Pick-Place-v0"
        config.training.algo = "bc"
        config.training.num_epochs = 200
        config.training.actor_layer_dims = (1024, 1024)
    elif args.task == "mobile_manipulation":
        config.project_name = "autodagger_mobile_manipulation"
        config.isaac_sim.task = "Isaac-Mobile-Manipulation-Obj-Scene-v0"
        config.training.algo = "bc"
        config.training.num_epochs = 300

    if args.api_key:
        config.anthropic.api_key = args.api_key

    config.save(args.output)
    print(f"Config saved to {args.output}")


if __name__ == "__main__":
    main()

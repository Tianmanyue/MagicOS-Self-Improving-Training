# AutoDAgger: Fully Automated DAgger in Simulation

Closed-loop self-improvement pipeline for robot manipulation policies.

## Core Idea

```
Train policy → Evaluate → Collect failures → VLM stage annotation →
Detect failure stage → Expert recovery demos → Retrain → Re-evaluate
```

No human intervention: scripted policy as expert + VLM automatic stage discovery + progress model for failure detection.

## Module Structure

```
autodagger/
├── config.py                     # All configuration dataclasses
├── pipeline.py                   # Main DAgger orchestration loop
├── stage_annotation/             # VLM two-pass stage discovery
│   ├── schema.py                 # StageLabel, TrajectoryStageAnnotation
│   ├── proprioceptive.py         # Signal analysis (gripper, velocity, pause)
│   └── vlm_annotator.py          # Claude Vision API annotator
├── progress/                     # Progress estimation & failure detection
│   ├── top_reward.py             # TOPReward zero-shot VLM progress
│   └── failure_detector.py       # Stuck detection from progress scores
├── dataset/                      # HDF5 data management
│   ├── trajectory.py             # TrajectoryData, TrajectoryMetadata
│   ├── hdf5_manager.py           # Read/write/merge HDF5 (Isaac Lab format)
│   └── versioning.py             # Per-round dataset versioning
├── training/                     # Policy training
│   ├── robomimic_trainer.py      # Robomimic BC/DiffusionPolicy wrapper
│   └── ra_bc.py                  # Reward-Aware BC sample weighting
├── eval/                         # Policy evaluation
│   ├── evaluator.py              # Isaac Sim evaluation loop
│   └── metrics.py                # Success rate, stage failure distribution
├── sim/                          # Simulation interface
│   ├── isaac_interface.py        # State get/set, step, reset
│   ├── replay.py                 # Replay trajectory to specific frame
│   └── expert_collector.py       # Scripted expert demo collection
├── mcp_tools/                    # MCP server (7 tools)
│   └── server.py                 # FastMCP tool definitions
└── scripts/                      # CLI entry points
    ├── run_full_pipeline.py      # End-to-end automated DAgger
    ├── run_dagger_round.py       # Single round (manual verification)
    └── create_default_config.py  # Generate config template
```

## Quick Start

### 1. Generate config

```bash
python -m autodagger.scripts.create_default_config \
    --task pick_and_place \
    --api-key "sk-ant-..." \
    --output my_config.json
```

### 2. Run full pipeline

```bash
python -m autodagger.scripts.run_full_pipeline \
    --config my_config.json \
    --task "pick and place red block" \
    --obs-keys joint_pos,joint_vel,object_position,target_object_position \
    --action-dim 8 \
    --initial-demos path/to/round0_demos.hdf5
```

### 3. Or run step-by-step (Phase 1 manual verification)

```bash
python -m autodagger.scripts.run_dagger_round \
    --config my_config.json \
    --round 1 \
    --checkpoint trained_models/round_0/models/model_best_validation.pth \
    --task "pick and place red block" \
    --obs-keys joint_pos,joint_vel,object_position,target_object_position \
    --action-dim 8
```

## Dependencies

**Required (all environments):**
- Python 3.10+
- numpy, h5py, anthropic, Pillow

**Required (GPU machine):**
- Isaac Sim 4.2.0
- IsaacLab (`pip install -e ../IsaacLab`)
- M2T2 (`pip install -e ../M2T2`)
- Robomimic (`pip install -e ../robomimic`)
- CuRobo
- PyTorch 2.5+

## GPU Requirements

| Component | GPUs |
|-----------|------|
| Isaac Sim | 1 |
| M2T2 grasp planning | 1 |
| Policy training | 1 |
| Progress model (optional) | 1 |
| **Total** | **3-4** |

## DAgger Algorithm

```
Round 0:
  Scripted expert (CuRobo + M2T2) collects initial demos → Train π₁

Round N:
  πₙ rollout → success rate ~X%
  → Collect all failed trajectories
  → VLM auto-annotates stage boundaries
  → Progress model detects stuck stage
  → Replay to failure state, scripted expert collects recovery demo
  → D_n = D_{n-1} ∪ recovery demos
  → Train π_{n+1}
```

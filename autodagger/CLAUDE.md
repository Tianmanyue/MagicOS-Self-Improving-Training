# AutoDAgger - Agent Onboarding Guide

## What This Is

Fully automated DAgger (Dataset Aggregation) pipeline for robot manipulation in NVIDIA Isaac Sim. Trains a behavior cloning policy, evaluates it, identifies failures via VLM progress estimation, collects expert recovery demos from failure states, and retrains iteratively until a target success rate is reached.

The code was written offline (no GPU) and needs to be debugged on a machine with Isaac Sim + GPU. All Isaac Sim imports are lazy-guarded so the code compiles and passes syntax checks without GPU.

## Architecture at a Glance

```
pipeline.py (orchestrator)
  ├── sim/isaac_interface.py      → Isaac Sim state get/set/step/reset
  ├── sim/expert_collector.py     → Scripted expert (CuRobo+M2T2) demo collection
  ├── sim/replay.py               → Restore simulator to a specific trajectory frame
  ├── eval/evaluator.py           → Roll out trained policy, save success/failure HDF5
  ├── eval/metrics.py             → Success rate, stage failure distribution
  ├── progress/top_reward.py      → VLM zero-shot progress scoring (Claude API)
  ├── progress/failure_detector.py→ Sliding-window stuck detection on progress scores
  ├── stage_annotation/           → Two-pass VLM stage discovery + proprioceptive refinement
  ├── dataset/hdf5_manager.py     → HDF5 CRUD (Isaac Lab format)
  ├── dataset/versioning.py       → Per-round dataset files + cumulative merge
  ├── training/robomimic_trainer.py → Generate Robomimic config + subprocess training
  ├── training/ra_bc.py           → Reward-aware BC: weight samples by progress gain
  ├── mcp_tools/server.py         → 7 MCP tools for agent-loop integration
  └── scripts/                    → CLI entry points
```

Read `docs/pipeline_flowchart.md` for Mermaid diagrams of data flow, stage annotation, and module dependencies.

## Key Design Decisions

- **Stage definitions are per-task, not per-trajectory.** `_discover_task_stages()` calls VLM once on the first trajectory, caches the result in `self._task_stage_cache`, and persists to `stage_definitions.json`. All subsequent failure-to-stage mappings use proportional scaling, no VLM.
- **Recovery demos only keep successes.** `ExpertCollector.collect_recovery_demos_batch()` discards demos where the expert itself failed.
- **Two-tier failure analysis.** Tier 1: VLM progress + failure detection (requires rendered RGB frames). Tier 2: proprioceptive signal boundaries or 80% heuristic.
- **Isaac imports are lazy.** All `from omni.*` / `from isaacsim.*` imports happen inside functions, guarded by `ISAAC_AVAILABLE`. Code compiles on CPU-only machines.
- **HDF5 format matches Isaac Lab.** `data/demo_N/{states/articulation/..., actions, obs/...}` with metadata as HDF5 attributes.

## How to Verify / Debug

### Level 0: Syntax + Import (no GPU needed)

```bash
# All 30 files must pass:
find autodagger -name '*.py' | xargs -I{} python -m py_compile {}

# Verify imports (no Isaac Sim needed for this):
python -c "from autodagger.config import PipelineConfig; print('config OK')"
python -c "from autodagger.dataset.trajectory import TrajectoryData; print('trajectory OK')"
python -c "from autodagger.stage_annotation.schema import StageLabel; print('schema OK')"
python -c "from autodagger.progress.failure_detector import FailureDetector; print('detector OK')"
```

### Level 1: Unit Tests (no GPU, no API key)

These modules have pure logic that can be tested with numpy arrays:

1. **ProprioceptiveAnalyzer** (`stage_annotation/proprioceptive.py`): feed synthetic gripper_width (step function), joint_velocities (with spikes), ee_velocity (with pauses) and verify detected boundaries.
2. **FailureDetector** (`progress/failure_detector.py`): feed a progress curve that plateaus and verify `detect_stuck()` returns the correct frame.
3. **TrajectoryData** (`dataset/trajectory.py`): test `get_state_at()`, `slice()`, `num_steps`.
4. **RABCWeighter** (`training/ra_bc.py`): feed known progress scores, check weight computation and normalization.
5. **HDF5Manager** (`dataset/hdf5_manager.py`): write/read/merge small trajectories in a temp dir.
6. **DatasetVersionManager** (`dataset/versioning.py`): create round + cumulative datasets, verify manifest.json.
7. **EvalMetrics** (`eval/metrics.py`): feed mock round_metrics dicts, check `compare_rounds()`.

### Level 2: VLM Integration (needs Anthropic API key, no GPU)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Test VLM stage annotation with a few saved PNG frames:
python -c "
from autodagger.stage_annotation.vlm_annotator import VLMStageAnnotator
from autodagger.config import AnthropicConfig, StageAnnotationConfig
import numpy as np
annotator = VLMStageAnnotator(AnthropicConfig(api_key='$ANTHROPIC_API_KEY'), StageAnnotationConfig())
# Use real frames here; dummy frames won't produce meaningful results
"
```

### Level 3: Isaac Sim Integration (needs GPU + Isaac Sim)

This is where most debugging will happen. Test in order:

1. **IsaacInterface** — `create_env()`, `reset()`, `get_state()`, `set_state()`, `step()`, `check_success()`.
2. **StateReplay** — `replay_to_frame()` with a known trajectory; compare resulting state.
3. **ExpertCollector** — `_get_expert_action_fn()` resolves correctly for your env; `_run_scripted_expert()` produces a valid TrajectoryData.
4. **PolicyEvaluator** — `_load_policy()` loads a Robomimic checkpoint; `_run_episode()` completes without crash.
5. **RobomimicTrainer** — `generate_config()` produces valid JSON; `train()` subprocess completes.

### Level 4: Full Pipeline (needs everything)

```bash
python -m autodagger.scripts.run_full_pipeline \
    --config my_config.json \
    --task "pick and place red block" \
    --obs-keys joint_pos,joint_vel,object_position,target_object_position \
    --action-dim 8 \
    --initial-demos path/to/round0_demos.hdf5
```

Watch for: config generation, Round 0 training completes, eval produces success/failure HDF5, failure analysis identifies stuck frames, recovery demo collection succeeds, cumulative dataset merge, Round 1 training starts.

## Known TODOs (must be resolved on GPU machine)

- `sim/expert_collector.py:_run_scripted_expert()` — placeholder; needs CuRobo + M2T2 integration or env's built-in `get_scripted_action()`.
- `sim/expert_collector.py:_setup_scene()` — scene-specific configuration not implemented.
- `sim/isaac_interface.py` — `set_state()` / `get_state()` keys depend on the actual Isaac Lab env structure; may need adjustment.
- `eval/evaluator.py:_load_policy()` — Robomimic `policy_from_checkpoint` import path may vary by version.
- `training/robomimic_trainer.py:generate_config()` — config structure depends on Robomimic version (0.3 vs 0.4).

## Reference Documents

- `project_plan.md` (repo root) — full project plan in Chinese, describes the research motivation, algorithm, phased implementation, and GPU requirements.
- `docs/pipeline_flowchart.md` — Mermaid diagrams of data flow, stage annotation, progress model, MCP tools, and file dependency graph.

## Coding Conventions

- All config lives in `config.py` dataclasses. No magic constants in module code.
- Heavy imports (Isaac Sim, torch, robomimic) are always lazy/conditional.
- HDF5 paths follow Isaac Lab convention: `data/demo_{N}/...`.
- VLM calls go through `anthropic` Python SDK with exponential backoff retries.
- Logging via `logging.getLogger(__name__)` — no print statements.

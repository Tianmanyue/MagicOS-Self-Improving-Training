"""HDF5 dataset manager for Isaac Lab / Robomimic trajectory data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from autodagger.config import DatasetConfig
from autodagger.dataset.trajectory import (
    TrajectoryData,
    TrajectoryMetadata,
    TrajectorySource,
)

logger = logging.getLogger(__name__)


class HDF5Manager:
    """Read, write, merge, and query HDF5 trajectory datasets in Isaac Lab format."""

    def __init__(self, config: DatasetConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_trajectory(self, hdf5_path: str, demo_name: str) -> TrajectoryData:
        with h5py.File(hdf5_path, "r") as f:
            if f"data/{demo_name}" not in f:
                raise KeyError(f"Demo '{demo_name}' not found in {hdf5_path}")
            grp = f[f"data/{demo_name}"]
            return self._read_demo_group(grp)

    def read_all_trajectories(self, hdf5_path: str) -> list[TrajectoryData]:
        trajectories: list[TrajectoryData] = []
        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]
            for demo_name in sorted(data_grp.keys(), key=_demo_sort_key):
                trajectories.append(self._read_demo_group(data_grp[demo_name]))
        logger.info("Read %d trajectories from %s", len(trajectories), hdf5_path)
        return trajectories

    def read_failed_trajectories(self, hdf5_path: str) -> list[TrajectoryData]:
        failed: list[TrajectoryData] = []
        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]
            for demo_name in sorted(data_grp.keys(), key=_demo_sort_key):
                grp = data_grp[demo_name]
                if not grp.attrs.get("success", False):
                    failed.append(self._read_demo_group(grp))
        logger.info("Read %d failed trajectories from %s", len(failed), hdf5_path)
        return failed

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_trajectory(
        self,
        hdf5_path: str,
        trajectory: TrajectoryData,
        demo_name: Optional[str] = None,
    ) -> str:
        Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5_path, "a") as f:
            data_grp = f.require_group("data")

            if demo_name is None:
                demo_name = self._next_demo_name(data_grp)

            if demo_name in data_grp:
                logger.warning("Overwriting existing demo '%s' in %s", demo_name, hdf5_path)
                del data_grp[demo_name]

            grp = data_grp.create_group(demo_name)
            self._write_demo_group(grp, trajectory)
            data_grp.attrs["total"] = len(data_grp.keys())

        logger.info("Wrote demo '%s' to %s", demo_name, hdf5_path)
        return demo_name

    def append_trajectories(
        self, hdf5_path: str, trajectories: list[TrajectoryData]
    ) -> None:
        Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5_path, "a") as f:
            data_grp = f.require_group("data")
            for traj in trajectories:
                name = self._next_demo_name(data_grp)
                grp = data_grp.create_group(name)
                self._write_demo_group(grp, traj)
            total = len(data_grp.keys())
            data_grp.attrs["total"] = total
        logger.info(
            "Appended %d trajectories to %s (total now %d)",
            len(trajectories),
            hdf5_path,
            total,
        )

    # ------------------------------------------------------------------
    # Merging / filtering
    # ------------------------------------------------------------------

    def merge_datasets(
        self,
        source_paths: list[str],
        output_path: str,
        deduplicate: bool = False,
    ) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        seen_ids: set[str] = set()
        demo_idx = 0

        with h5py.File(output_path, "w") as out_f:
            out_data = out_f.create_group("data")

            for src_path in source_paths:
                if not Path(src_path).exists():
                    logger.warning("Source file not found, skipping: %s", src_path)
                    continue

                with h5py.File(src_path, "r") as src_f:
                    if "data" not in src_f:
                        logger.warning("No 'data' group in %s, skipping", src_path)
                        continue

                    src_data = src_f["data"]
                    env_args = src_data.attrs.get("env_args", None)
                    if env_args is not None and "env_args" not in out_data.attrs:
                        out_data.attrs["env_args"] = env_args

                    for demo_name in sorted(src_data.keys(), key=_demo_sort_key):
                        if deduplicate:
                            orig_id = src_data[demo_name].attrs.get(
                                "original_trajectory_id", None
                            )
                            if orig_id and orig_id in seen_ids:
                                continue
                            if orig_id:
                                seen_ids.add(orig_id)

                        dst_name = f"demo_{demo_idx}"
                        src_data.copy(demo_name, out_data, name=dst_name)
                        demo_idx += 1

            out_data.attrs["total"] = demo_idx

        logger.info(
            "Merged %d source files into %s (%d demos)",
            len(source_paths),
            output_path,
            demo_idx,
        )

    def filter_by_round(
        self, hdf5_path: str, round_num: int, output_path: str
    ) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        demo_idx = 0

        with h5py.File(hdf5_path, "r") as src_f, h5py.File(output_path, "w") as out_f:
            src_data = src_f["data"]
            out_data = out_f.create_group("data")

            if "env_args" in src_data.attrs:
                out_data.attrs["env_args"] = src_data.attrs["env_args"]

            for demo_name in sorted(src_data.keys(), key=_demo_sort_key):
                grp = src_data[demo_name]
                if grp.attrs.get("dagger_round", -1) == round_num:
                    dst_name = f"demo_{demo_idx}"
                    src_data.copy(demo_name, out_data, name=dst_name)
                    demo_idx += 1

            out_data.attrs["total"] = demo_idx

        logger.info(
            "Filtered round %d from %s -> %s (%d demos)",
            round_num,
            hdf5_path,
            output_path,
            demo_idx,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_dataset_stats(self, hdf5_path: str) -> dict:
        stats = {
            "num_demos": 0,
            "num_successful": 0,
            "num_failed": 0,
            "total_steps": 0,
            "source_counts": {},
        }

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                return stats
            data_grp = f["data"]
            for demo_name in data_grp.keys():
                grp = data_grp[demo_name]
                stats["num_demos"] += 1

                success = bool(grp.attrs.get("success", False))
                if success:
                    stats["num_successful"] += 1
                else:
                    stats["num_failed"] += 1

                if "actions" in grp:
                    stats["total_steps"] += grp["actions"].shape[0]
                else:
                    stats["total_steps"] += int(grp.attrs.get("num_samples", 0))

                source = grp.attrs.get("source", "unknown")
                stats["source_counts"][source] = (
                    stats["source_counts"].get(source, 0) + 1
                )

        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_demo_name(data_grp: h5py.Group) -> str:
        existing = [k for k in data_grp.keys() if k.startswith("demo_")]
        if not existing:
            return "demo_0"
        max_idx = max(_demo_sort_key(k) for k in existing)
        return f"demo_{max_idx + 1}"

    @staticmethod
    def _read_demo_group(grp: h5py.Group) -> TrajectoryData:
        attrs = dict(grp.attrs)

        states = _read_nested(grp["states"]) if "states" in grp else {}
        initial_state = _read_nested(grp["initial_state"]) if "initial_state" in grp else {}
        observations = _read_nested(grp["obs"]) if "obs" in grp else {}
        actions = np.array(grp["actions"]) if "actions" in grp else None

        source_str = attrs.get("source", TrajectorySource.SCRIPTED.value)
        try:
            source = TrajectorySource(source_str)
        except ValueError:
            source = TrajectorySource.SCRIPTED

        metadata = TrajectoryMetadata(
            source=source,
            dagger_round=int(attrs.get("dagger_round", 0)),
            target_stage=attrs.get("target_stage", None) or None,
            original_trajectory_id=attrs.get("original_trajectory_id", None) or None,
            failure_frame=int(attrs["failure_frame"]) if "failure_frame" in attrs else None,
            scene_id=attrs.get("scene_id", None) or None,
        )

        seed_val = attrs.get("seed", None)
        seed = int(seed_val) if seed_val is not None else None

        return TrajectoryData(
            states=states,
            actions=actions,
            observations=observations,
            initial_state=initial_state,
            success=bool(attrs.get("success", False)),
            seed=seed,
            metadata=metadata,
        )

    @staticmethod
    def _write_demo_group(grp: h5py.Group, traj: TrajectoryData) -> None:
        grp.attrs["num_samples"] = traj.num_steps
        grp.attrs["success"] = traj.success
        if traj.seed is not None:
            grp.attrs["seed"] = traj.seed

        meta = traj.metadata
        grp.attrs["source"] = meta.source.value
        grp.attrs["dagger_round"] = meta.dagger_round
        if meta.target_stage is not None:
            grp.attrs["target_stage"] = meta.target_stage
        if meta.original_trajectory_id is not None:
            grp.attrs["original_trajectory_id"] = meta.original_trajectory_id
        if meta.failure_frame is not None:
            grp.attrs["failure_frame"] = meta.failure_frame
        if meta.scene_id is not None:
            grp.attrs["scene_id"] = meta.scene_id

        if traj.actions is not None:
            grp.create_dataset("actions", data=traj.actions)

        if traj.states:
            _write_nested(grp.create_group("states"), traj.states)
        if traj.initial_state:
            _write_nested(grp.create_group("initial_state"), traj.initial_state)
        if traj.observations:
            _write_nested(grp.create_group("obs"), traj.observations)


# ------------------------------------------------------------------
# Module-level helpers for nested HDF5 structures
# ------------------------------------------------------------------


def _demo_sort_key(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _read_nested(group: h5py.Group) -> dict:
    result: dict = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            result[key] = _read_nested(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = np.array(item)
    return result


def _write_nested(group: h5py.Group, data: dict) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            _write_nested(group.create_group(key), value)
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        else:
            group.create_dataset(key, data=np.asarray(value))

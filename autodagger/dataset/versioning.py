"""Dataset versioning for the DAgger loop, tracking per-round and cumulative datasets."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from autodagger.config import DatasetConfig
from autodagger.dataset.hdf5_manager import HDF5Manager
from autodagger.dataset.trajectory import TrajectoryData

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


class DatasetVersionManager:
    """Manages per-round and cumulative HDF5 datasets with a JSON manifest."""

    def __init__(self, base_dir: str, prefix: str = "autodagger"):
        self.base_dir = Path(base_dir)
        self.prefix = prefix
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._hdf5 = HDF5Manager(DatasetConfig(base_dir=base_dir, filename_prefix=prefix))
        self._manifest_path = self.base_dir / MANIFEST_FILENAME
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Path conventions
    # ------------------------------------------------------------------

    def get_round_path(self, round_num: int) -> str:
        return str(self.base_dir / f"{self.prefix}_round_{round_num}.hdf5")

    def get_cumulative_path(self, round_num: int) -> str:
        return str(self.base_dir / f"{self.prefix}_cumulative_{round_num}.hdf5")

    # ------------------------------------------------------------------
    # Round dataset creation
    # ------------------------------------------------------------------

    def create_round_dataset(
        self, round_num: int, trajectories: list[TrajectoryData]
    ) -> str:
        path = self.get_round_path(round_num)
        self._hdf5.append_trajectories(path, trajectories)

        stats = self._hdf5.get_dataset_stats(path)
        self._update_manifest(round_num, stats, path)

        logger.info(
            "Created round %d dataset at %s (%d demos)",
            round_num,
            path,
            stats["num_demos"],
        )
        return path

    def create_cumulative_dataset(self, round_num: int) -> str:
        source_paths: list[str] = []
        for r in range(round_num + 1):
            rp = self.get_round_path(r)
            if Path(rp).exists():
                source_paths.append(rp)
            else:
                logger.warning("Round %d file not found at %s, skipping", r, rp)

        output_path = self.get_cumulative_path(round_num)
        self._hdf5.merge_datasets(source_paths, output_path)

        stats = self._hdf5.get_dataset_stats(output_path)
        entry = self._manifest.get("rounds", {}).get(str(round_num), {})
        entry["cumulative_path"] = output_path
        entry["cumulative_stats"] = stats
        self._manifest.setdefault("rounds", {})[str(round_num)] = entry
        self._save_manifest()

        logger.info(
            "Created cumulative dataset up to round %d at %s (%d demos)",
            round_num,
            output_path,
            stats["num_demos"],
        )
        return output_path

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_latest_round(self) -> int:
        rounds = self._manifest.get("rounds", {})
        if not rounds:
            return self._scan_latest_round()
        return max(int(r) for r in rounds.keys())

    def get_round_summary(self, round_num: int) -> dict:
        rounds = self._manifest.get("rounds", {})
        key = str(round_num)

        if key in rounds:
            return rounds[key]

        path = self.get_round_path(round_num)
        if not Path(path).exists():
            raise FileNotFoundError(f"No dataset found for round {round_num} at {path}")

        stats = self._hdf5.get_dataset_stats(path)
        summary = {
            "round": round_num,
            "path": path,
            "stats": stats,
        }
        return summary

    def get_history(self) -> list[dict]:
        rounds = self._manifest.get("rounds", {})
        history: list[dict] = []
        for key in sorted(rounds.keys(), key=int):
            entry = dict(rounds[key])
            entry["round"] = int(key)
            history.append(entry)
        return history

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            with open(self._manifest_path, "r") as f:
                return json.load(f)
        return {"prefix": self.prefix, "rounds": {}}

    def _save_manifest(self) -> None:
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def _update_manifest(self, round_num: int, stats: dict, path: str) -> None:
        key = str(round_num)
        entry = self._manifest.setdefault("rounds", {}).get(key, {})
        entry.update(
            {
                "round": round_num,
                "path": path,
                "stats": stats,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._manifest["rounds"][key] = entry
        self._save_manifest()

    def _scan_latest_round(self) -> int:
        latest = -1
        for p in self.base_dir.glob(f"{self.prefix}_round_*.hdf5"):
            try:
                num = int(p.stem.rsplit("_", 1)[1])
                latest = max(latest, num)
            except (ValueError, IndexError):
                continue
        return latest

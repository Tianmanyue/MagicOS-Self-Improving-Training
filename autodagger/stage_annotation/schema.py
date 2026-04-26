"""Data models for stage annotation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class StageLabel:
    id: int
    name: str
    start_frame: int
    end_frame: int
    key_signal: str = ""
    confidence: float = 1.0

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class TrajectoryStageAnnotation:
    trajectory_id: str
    task: str
    stages: list[StageLabel] = field(default_factory=list)
    total_frames: int = 0
    success: bool = False
    failure_stage_id: Optional[int] = None

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def failure_stage(self) -> Optional[StageLabel]:
        if self.failure_stage_id is not None:
            for s in self.stages:
                if s.id == self.failure_stage_id:
                    return s
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryStageAnnotation":
        stages = [StageLabel(**s) for s in d.get("stages", [])]
        return cls(
            trajectory_id=d["trajectory_id"],
            task=d["task"],
            stages=stages,
            total_frames=d.get("total_frames", 0),
            success=d.get("success", False),
            failure_stage_id=d.get("failure_stage_id"),
        )

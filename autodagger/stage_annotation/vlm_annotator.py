"""VLM-based stage annotator implementing two-pass adaptive stage discovery.

Pass 1 -- Coarse discovery:
    Sample a small number of frames uniformly from the trajectory, send them
    to a VLM, and ask it to name the manipulation stages it observes.

Pass 2 -- Boundary refinement:
    Run the ``ProprioceptiveAnalyzer`` to obtain candidate boundary frames.
    If the number of proprioceptive boundaries matches the number of inter-
    stage transitions (i.e. ``num_stages - 1``), adopt them directly.
    Otherwise, send the ambiguous boundary frames back to the VLM so it can
    confirm or adjust which frames mark true stage transitions.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
import random
from typing import Any

import numpy as np

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from autodagger.config import AnthropicConfig, StageAnnotationConfig
from autodagger.stage_annotation.schema import StageLabel, TrajectoryStageAnnotation
from autodagger.stage_annotation.proprioceptive import ProprioceptiveAnalyzer

logger = logging.getLogger(__name__)


class VLMStageAnnotator:
    """Two-pass VLM stage annotator for robot manipulation trajectories.

    Parameters
    ----------
    anthropic_config : AnthropicConfig
        Anthropic API credentials and generation parameters.
    stage_config : StageAnnotationConfig
        Configuration controlling coarse sampling, thresholds, etc.
    """

    def __init__(
        self,
        anthropic_config: AnthropicConfig,
        stage_config: StageAnnotationConfig,
    ) -> None:
        if anthropic is None:
            raise ImportError(
                "The `anthropic` package is required.  "
                "Install it with: pip install anthropic"
            )

        self.anthropic_config = anthropic_config
        self.stage_config = stage_config

        self._client = anthropic.Anthropic(api_key=anthropic_config.api_key)

    # ==================================================================
    # Public API
    # ==================================================================

    def annotate_trajectory(
        self,
        trajectory_frames: list[np.ndarray],
        proprioceptive_data: dict[str, np.ndarray],
        trajectory_id: str,
        task: str,
    ) -> TrajectoryStageAnnotation:
        """Run the full two-pass annotation pipeline.

        Parameters
        ----------
        trajectory_frames : list[np.ndarray]
            RGB images, each of shape (H, W, 3), uint8.
        proprioceptive_data : dict
            Must contain keys ``"gripper_width"``, ``"joint_velocities"``,
            and ``"ee_velocity"`` as numpy arrays.
        trajectory_id : str
            Unique identifier for this trajectory (used in the output).
        task : str
            Human-readable task description (e.g. "pick and place red block").

        Returns
        -------
        TrajectoryStageAnnotation
        """
        total_frames = len(trajectory_frames)
        logger.info(
            "Annotating trajectory %s (%d frames, task=%r)",
            trajectory_id,
            total_frames,
            task,
        )

        # ---- Pass 1: coarse VLM discovery ----
        stage_names = self._pass1_coarse_discovery(trajectory_frames, task)
        logger.info("Pass 1 discovered %d stages: %s", len(stage_names), stage_names)

        if not stage_names:
            logger.warning("VLM returned no stage names; returning empty annotation.")
            return TrajectoryStageAnnotation(
                trajectory_id=trajectory_id,
                task=task,
                stages=[],
                total_frames=total_frames,
                success=False,
            )

        # ---- Pass 2: proprioceptive analysis + optional VLM refinement ----
        analyzer = ProprioceptiveAnalyzer(
            gripper_width=proprioceptive_data["gripper_width"],
            joint_velocities=proprioceptive_data["joint_velocities"],
            ee_velocity=proprioceptive_data["ee_velocity"],
            config=self.stage_config,
        )
        candidate_boundaries = analyzer.find_all_boundaries()

        num_expected_boundaries = len(stage_names) - 1
        logger.info(
            "Proprioceptive pass found %d boundaries; expected %d (for %d stages)",
            len(candidate_boundaries),
            num_expected_boundaries,
            len(stage_names),
        )

        if num_expected_boundaries <= 0:
            # Single stage -- the entire trajectory is one stage.
            boundary_frames: list[int] = []
        elif len(candidate_boundaries) == num_expected_boundaries:
            # Perfect match -- use proprioceptive boundaries directly.
            boundary_frames = [b[0] for b in candidate_boundaries]
            logger.info("Boundary count matches; using proprioceptive boundaries directly.")
        else:
            # Mismatch -- call VLM to refine.
            boundary_frames = self._pass2_vlm_refinement(
                trajectory_frames,
                candidate_boundaries,
                stage_names,
                total_frames,
            )

        # ---- Build stage labels ----
        stages = self._build_stage_labels(
            stage_names, boundary_frames, total_frames, candidate_boundaries
        )

        annotation = TrajectoryStageAnnotation(
            trajectory_id=trajectory_id,
            task=task,
            stages=stages,
            total_frames=total_frames,
            success=True,
        )
        logger.info(
            "Annotation complete for %s: %d stages covering frames 0-%d",
            trajectory_id,
            len(stages),
            total_frames - 1,
        )
        return annotation

    # ==================================================================
    # Pass 1 -- coarse discovery
    # ==================================================================

    def _pass1_coarse_discovery(
        self,
        trajectory_frames: list[np.ndarray],
        task: str,
    ) -> list[str]:
        """Sample frames uniformly and ask the VLM to name the stages."""
        total = len(trajectory_frames)
        num_samples = min(self.stage_config.coarse_num_frames, total)
        indices = np.linspace(0, total - 1, num_samples, dtype=int).tolist()

        logger.debug("Coarse sampling indices: %s", indices)

        content: list[dict[str, Any]] = [
            {"type": "text", "text": self._build_coarse_prompt(task)},
        ]
        for idx in indices:
            content.append(
                {
                    "type": "text",
                    "text": f"[Frame {idx}/{total - 1}]",
                }
            )
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self._encode_frame(trajectory_frames[idx]),
                    },
                }
            )

        messages = [{"role": "user", "content": content}]
        vlm_text = self._call_vlm(messages)
        return self._parse_stage_names(vlm_text)

    # ==================================================================
    # Pass 2 -- VLM refinement
    # ==================================================================

    def _pass2_vlm_refinement(
        self,
        trajectory_frames: list[np.ndarray],
        candidate_boundaries: list[tuple[int, str]],
        stage_names: list[str],
        total_frames: int,
    ) -> list[int]:
        """Send ambiguous boundary frames to the VLM for refinement.

        Returns a list of ``len(stage_names) - 1`` boundary frame indices.
        """
        num_expected = len(stage_names) - 1

        # Collect frames around each candidate boundary for the VLM to inspect.
        # Also include a few extra evenly-spaced frames for context.
        frames_to_send: list[int] = [b[0] for b in candidate_boundaries]

        # Add evenly-spaced context frames (avoid duplicates).
        context_indices = np.linspace(
            0, total_frames - 1, min(self.stage_config.coarse_num_frames, total_frames), dtype=int
        ).tolist()
        for ci in context_indices:
            if ci not in frames_to_send:
                frames_to_send.append(ci)
        frames_to_send.sort()

        # Cap the number of VLM calls.
        if len(frames_to_send) > self.stage_config.vlm_max_refinement_calls * 5:
            # Subsample to keep the request manageable.
            step = max(1, len(frames_to_send) // (self.stage_config.vlm_max_refinement_calls * 5))
            frames_to_send = frames_to_send[::step]

        prompt_text = self._build_refinement_prompt(
            stage_names, candidate_boundaries
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for idx in frames_to_send:
            idx = min(idx, total_frames - 1)
            content.append({"type": "text", "text": f"[Frame {idx}/{total_frames - 1}]"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self._encode_frame(trajectory_frames[idx]),
                    },
                }
            )

        messages = [{"role": "user", "content": content}]
        vlm_text = self._call_vlm(messages)

        refined = self._parse_refined_boundaries(vlm_text, num_expected, total_frames)
        return refined

    # ==================================================================
    # Stage-label construction
    # ==================================================================

    @staticmethod
    def _build_stage_labels(
        stage_names: list[str],
        boundary_frames: list[int],
        total_frames: int,
        candidate_boundaries: list[tuple[int, str]],
    ) -> list[StageLabel]:
        """Convert stage names + boundary frames into ``StageLabel`` objects."""
        boundary_frames = sorted(boundary_frames)

        # Build a lookup from frame -> signal type.
        signal_lookup: dict[int, str] = {b[0]: b[1] for b in candidate_boundaries}

        stages: list[StageLabel] = []
        starts = [0] + [b + 1 for b in boundary_frames]
        ends = boundary_frames + [total_frames - 1]

        for i, name in enumerate(stage_names):
            if i >= len(starts):
                break
            start = starts[i]
            end = ends[i] if i < len(ends) else total_frames - 1
            # Determine the key signal that triggered this boundary.
            key_signal = ""
            if i < len(boundary_frames):
                key_signal = signal_lookup.get(boundary_frames[i], "vlm_refined")
            stages.append(
                StageLabel(
                    id=i,
                    name=name,
                    start_frame=start,
                    end_frame=end,
                    key_signal=key_signal,
                    confidence=1.0,
                )
            )

        return stages

    # ==================================================================
    # VLM prompt builders
    # ==================================================================

    @staticmethod
    def _build_coarse_prompt(task: str) -> str:
        """Build the first-pass prompt for coarse stage discovery."""
        return (
            "You are an expert robot manipulation analyst. You are given a "
            "sequence of frames sampled uniformly from a robot manipulation "
            "trajectory.\n\n"
            f"Task description: {task}\n\n"
            "Examine the frames carefully and identify the distinct manipulation "
            "stages the robot goes through to complete (or attempt) this task. "
            "Typical stages might include approaching, grasping, lifting, "
            "transporting, placing, and retreating, but use whatever stage "
            "names best describe what you observe.\n\n"
            "Respond ONLY with a JSON object in the following format (no "
            "other text):\n"
            "```json\n"
            "{\n"
            '  "stages": [\n'
            '    {"name": "stage_name_1", "description": "brief description"},\n'
            '    {"name": "stage_name_2", "description": "brief description"}\n'
            "  ]\n"
            "}\n"
            "```\n\n"
            "List the stages in chronological order. Use short, descriptive "
            "names in snake_case (e.g. \"approaching_object\", \"grasping\", "
            "\"lifting\")."
        )

    @staticmethod
    def _build_refinement_prompt(
        stage_names: list[str],
        candidate_boundaries: list[tuple[int, str]],
    ) -> str:
        """Build the second-pass prompt for boundary refinement."""
        stages_str = ", ".join(f'"{s}"' for s in stage_names)
        boundaries_str = json.dumps(
            [{"frame": b[0], "signal": b[1]} for b in candidate_boundaries],
            indent=2,
        )
        num_expected = len(stage_names) - 1

        return (
            "You are an expert robot manipulation analyst performing boundary "
            "refinement for stage annotation.\n\n"
            f"The trajectory has been divided into {len(stage_names)} stages "
            f"(in order): [{stages_str}].\n\n"
            f"We need exactly {num_expected} boundary frame(s) separating "
            "consecutive stages. A proprioceptive signal analysis produced the "
            "following candidate boundaries:\n"
            f"{boundaries_str}\n\n"
            "However, the number of candidates does not match the expected "
            f"count of {num_expected}. Examine the frames below and determine "
            "which frames mark the true transitions between stages.\n\n"
            "Respond ONLY with a JSON object in the following format (no "
            "other text):\n"
            "```json\n"
            "{\n"
            '  "boundaries": [\n'
            '    {"frame": <frame_index>, "transition": "stage_A -> stage_B"},\n'
            '    {"frame": <frame_index>, "transition": "stage_B -> stage_C"}\n'
            "  ]\n"
            "}\n"
            "```\n\n"
            f"Return exactly {num_expected} boundary/boundaries in ascending "
            "frame order. Each frame index must be a valid integer within the "
            "trajectory range shown in the frame labels."
        )

    # ==================================================================
    # VLM response parsers
    # ==================================================================

    @staticmethod
    def _parse_stage_names(vlm_response: str) -> list[str]:
        """Extract stage names from the VLM's JSON response.

        Handles both clean JSON and JSON wrapped in markdown code fences.
        """
        text = vlm_response.strip()

        # Strip markdown code fences if present.
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse VLM coarse response as JSON: %s", text[:200])
            # Fallback: try to find any JSON object in the text.
            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                try:
                    data = json.loads(obj_match.group(0))
                except json.JSONDecodeError:
                    logger.error("Fallback JSON parse also failed.")
                    return []
            else:
                return []

        stages_list = data.get("stages", [])
        names: list[str] = []
        for entry in stages_list:
            if isinstance(entry, dict) and "name" in entry:
                names.append(str(entry["name"]))
            elif isinstance(entry, str):
                names.append(entry)
        return names

    @staticmethod
    def _parse_refined_boundaries(
        vlm_response: str,
        num_expected: int,
        total_frames: int,
    ) -> list[int]:
        """Extract boundary frame indices from the VLM's refinement response."""
        text = vlm_response.strip()

        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse VLM refinement response: %s", text[:200])
            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                try:
                    data = json.loads(obj_match.group(0))
                except json.JSONDecodeError:
                    logger.error("Fallback JSON parse also failed for refinement.")
                    return _fallback_even_boundaries(num_expected, total_frames)
            else:
                return _fallback_even_boundaries(num_expected, total_frames)

        boundaries_raw = data.get("boundaries", [])
        frames: list[int] = []
        for entry in boundaries_raw:
            if isinstance(entry, dict) and "frame" in entry:
                f = int(entry["frame"])
                f = max(0, min(f, total_frames - 1))
                frames.append(f)
            elif isinstance(entry, (int, float)):
                f = int(entry)
                f = max(0, min(f, total_frames - 1))
                frames.append(f)

        frames.sort()

        # Validate count.
        if len(frames) != num_expected:
            logger.warning(
                "VLM refinement returned %d boundaries but expected %d; "
                "falling back to evenly-spaced boundaries.",
                len(frames),
                num_expected,
            )
            return _fallback_even_boundaries(num_expected, total_frames)

        return frames

    # ==================================================================
    # Image encoding
    # ==================================================================

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        """Encode an RGB numpy array (H, W, 3) as a base64 PNG string."""
        # Import PIL lazily to keep the module importable without it at
        # parse time (PIL is only needed when actually encoding).
        from PIL import Image

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        img = Image.fromarray(frame, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ==================================================================
    # VLM call with retries
    # ==================================================================

    def _call_vlm(self, messages: list[dict[str, Any]]) -> str:
        """Call the Anthropic Messages API with exponential-backoff retries.

        Returns the text content of the first content block.
        """
        last_exc: Exception | None = None

        for attempt in range(self.anthropic_config.max_retries):
            try:
                response = self._client.messages.create(
                    model=self.stage_config.vlm_model,
                    max_tokens=self.anthropic_config.max_tokens,
                    temperature=self.anthropic_config.temperature,
                    messages=messages,
                )
                text = response.content[0].text
                logger.debug("VLM response (attempt %d): %s", attempt + 1, text[:300])
                return text
            except Exception as exc:
                last_exc = exc
                delay = min(2 ** attempt + random.random(), 60.0)
                logger.warning(
                    "VLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    self.anthropic_config.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        # All retries exhausted.
        raise RuntimeError(
            f"VLM call failed after {self.anthropic_config.max_retries} attempts"
        ) from last_exc


# ======================================================================
# Module-level helpers
# ======================================================================

def _fallback_even_boundaries(num_boundaries: int, total_frames: int) -> list[int]:
    """Produce evenly-spaced boundary indices as a last-resort fallback."""
    if num_boundaries <= 0:
        return []
    step = total_frames / (num_boundaries + 1)
    return [int(round(step * (i + 1))) for i in range(num_boundaries)]

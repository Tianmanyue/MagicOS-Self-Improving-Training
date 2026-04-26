"""TOPReward: Token-probability-as-reward progress estimation via VLM."""

from __future__ import annotations

import base64
import logging
import re
from typing import Optional

import anthropic
import numpy as np

from autodagger.config import AnthropicConfig, ProgressModelConfig

logger = logging.getLogger(__name__)

PROGRESS_PROMPT = (
    "On a scale of 0.0 to 1.0, how much progress has been made toward "
    "completing the task: {task}? Reply with only a number."
)

COMPLETION_PROMPT = (
    "Has this task been completed: {task}? "
    "Reply with a confidence score 0.0-1.0."
)

_DEFAULT_SAMPLE_INTERVAL = 10
_MIN_FRAMES_FOR_SAMPLING = 20


class TOPRewardEstimator:
    """Zero-shot progress estimation using VLM confidence as a reward signal.

    Sends sampled frames to a VLM and asks for a scalar progress score,
    then linearly interpolates scores for non-sampled frames.
    """

    def __init__(
        self,
        anthropic_config: AnthropicConfig,
        progress_config: ProgressModelConfig,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=anthropic_config.api_key)
        self._model = progress_config.vlm_model
        self._max_tokens = anthropic_config.max_tokens
        self._temperature = anthropic_config.temperature
        self._max_retries = anthropic_config.max_retries

    def estimate_progress(
        self,
        frames: list[np.ndarray],
        task_description: str,
        sample_interval: int = _DEFAULT_SAMPLE_INTERVAL,
    ) -> list[float]:
        """Estimate per-frame progress scores for an entire trajectory.

        For efficiency, only every *sample_interval*-th frame is sent to the
        VLM.  Scores for intermediate frames are linearly interpolated.  If the
        trajectory is short enough, every frame is scored directly.
        """
        n = len(frames)
        if n == 0:
            return []

        sampled_indices = self._pick_sample_indices(n, sample_interval)
        logger.info(
            "Scoring %d / %d frames for task: %s",
            len(sampled_indices), n, task_description,
        )

        prompt = PROGRESS_PROMPT.format(task=task_description)
        sampled_scores: dict[int, float] = {}
        for idx in sampled_indices:
            score = self._score_frame(frames[idx], prompt)
            sampled_scores[idx] = score
            logger.debug("Frame %d -> progress %.3f", idx, score)

        return self._interpolate(sampled_scores, n)

    def estimate_completion(
        self,
        frames: list[np.ndarray],
        task_description: str,
    ) -> float:
        """Quick single-frame check of task completion confidence."""
        if len(frames) == 0:
            return 0.0

        prompt = COMPLETION_PROMPT.format(task=task_description)
        return self._score_frame(frames[-1], prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_sample_indices(n: int, interval: int) -> list[int]:
        if n < _MIN_FRAMES_FOR_SAMPLING:
            return list(range(n))

        indices = list(range(0, n, interval))
        if indices[-1] != n - 1:
            indices.append(n - 1)
        return indices

    def _score_frame(self, frame: np.ndarray, prompt: str) -> float:
        text = self._call_vlm_single(frame, prompt)
        return self._parse_score(text)

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        """Encode an RGB numpy array as a base64 JPEG string."""
        try:
            import cv2
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        except ImportError:
            from PIL import Image
            import io
            img = Image.fromarray(frame)
            bio = io.BytesIO()
            img.save(bio, format="JPEG")
            buf = bio.getvalue()
            return base64.b64encode(buf).decode("utf-8")

        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _call_vlm_single(self, frame: np.ndarray, prompt: str) -> str:
        encoded = self._encode_frame(frame)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        last_err: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    messages=messages,
                )
                return response.content[0].text
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "VLM call failed (attempt %d/%d): %s",
                    attempt, self._max_retries, exc,
                )

        raise RuntimeError(
            f"VLM call failed after {self._max_retries} attempts"
        ) from last_err

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract the first float in [0, 1] from VLM response text."""
        match = re.search(r"(\d+\.?\d*)", text.strip())
        if match is None:
            logger.warning("Could not parse score from VLM response: %r", text)
            return 0.0
        value = float(match.group(1))
        return max(0.0, min(1.0, value))

    @staticmethod
    def _interpolate(sampled: dict[int, float], n: int) -> list[float]:
        """Linearly interpolate between sampled frame scores."""
        if n == 0:
            return []

        scores = [0.0] * n
        sorted_keys = sorted(sampled.keys())

        if len(sorted_keys) == 1:
            return [sampled[sorted_keys[0]]] * n

        for key in sorted_keys:
            scores[key] = sampled[key]

        # Fill before the first sampled index with its value.
        for i in range(0, sorted_keys[0]):
            scores[i] = sampled[sorted_keys[0]]

        # Interpolate between consecutive sampled indices.
        for seg in range(len(sorted_keys) - 1):
            lo = sorted_keys[seg]
            hi = sorted_keys[seg + 1]
            lo_val = sampled[lo]
            hi_val = sampled[hi]
            span = hi - lo
            for i in range(lo + 1, hi):
                alpha = (i - lo) / span
                scores[i] = lo_val + alpha * (hi_val - lo_val)

        # Fill after the last sampled index with its value.
        for i in range(sorted_keys[-1] + 1, n):
            scores[i] = sampled[sorted_keys[-1]]

        return scores

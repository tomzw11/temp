"""Custom reward functions for QwenPaw task evaluation.

Provides a pluggable reward interface with a default GSM8K math-answer scorer.
Users can implement custom reward logic by satisfying the ``RewardFunction``
protocol and passing it to ``run_qwenpaw``.
"""

from __future__ import annotations

import re
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class RewardFunction(Protocol):
    """Protocol for custom reward functions.

    Implement this protocol to define your own scoring logic::

        class MyReward:
            async def compute(self, content: str, ground_truth: str, **kwargs: Any) -> float:
                return 1.0 if content.strip() == ground_truth.strip() else 0.0
    """

    async def compute(self, content: str, ground_truth: str, **kwargs: Any) -> float:
        """Compute a reward score given the model output and ground truth.

        Args:
            content: The model's full text output.
            ground_truth: The expected answer (format depends on the task).
            **kwargs: Additional metadata (e.g. task_id, extra_info).

        Returns:
            A float reward (typically 0.0 or 1.0, but can be continuous).
        """
        ...


# ---------------------------------------------------------------------------
# GSM8K Reward
# ---------------------------------------------------------------------------


class GSM8KReward:
    """GSM8K-style math reasoning reward.

    GSM8K problems typically end with a line like ``#### 42``. This scorer
    extracts the final answer after ``####`` (or the last number in the output)
    and compares it to the ground-truth number.

    Usage::

        reward_fn = GSM8KReward()
        score = await reward_fn.compute("The answer is #### 42", "42")
        # score == 1.0

    Customisation::

        class GSM8KExactMatch(GSM8KReward):
            def _extract_answer(self, content: str) -> str | None:
                # Override with your own extraction logic
                ...
    """

    # Matches the final "#### <number>" line (GSM8K convention)
    _HASH_ANSWER_PATTERN = re.compile(r"####\s*(-?\d+(?:[.,]\d+)?)", re.IGNORECASE)

    # Matches any number in the text (fallback: pick the last one)
    _LAST_NUMBER_PATTERN = re.compile(r"(-?\d+(?:[.,]\d+)?)")

    async def compute(
        self,
        content: str,
        ground_truth: str,
        **kwargs: Any,
    ) -> float:
        """Score a GSM8K response.

        Args:
            content: The model's full text output.
            ground_truth: The expected numeric answer as a string.
            **kwargs: Ignored (reserved for future use).

        Returns:
            1.0 if the extracted answer matches the ground truth, else 0.0.
        """
        predicted = self._extract_answer(content)
        if predicted is None:
            return 0.0

        return 1.0 if self._normalize(predicted) == self._normalize(ground_truth) else 0.0

    def _extract_answer(self, content: str) -> str | None:
        """Extract the final answer from the model output.

        Strategy:
            1. Look for the ``#### <number>`` pattern (GSM8K convention).
            2. Fall back to the last number in the text.

        Override this method to implement custom extraction logic.
        """
        match = self._HASH_ANSWER_PATTERN.search(content)
        if match:
            return match.group(1)

        numbers = self._LAST_NUMBER_PATTERN.findall(content)
        if numbers:
            return numbers[-1]

        return None

    @staticmethod
    def _normalize(value: str) -> str:
        """Normalize a numeric string for comparison.

        Strips commas, trailing zeros, and leading zeros.
        """
        cleaned = value.replace(",", "").replace(" ", "")
        try:
            if "." in cleaned:
                return str(float(cleaned))
            return str(int(cleaned))
        except ValueError:
            return cleaned.strip()


# ---------------------------------------------------------------------------
# Composite reward (weighted average of multiple scorers)
# ---------------------------------------------------------------------------


class CompositeReward:
    """Weighted combination of multiple reward functions.

    Usage::

        composite = CompositeReward([
            (GSM8KReward(), 0.7),
            (MyFormatReward(), 0.3),
        ])
        score = await composite.compute(content, ground_truth)
    """

    def __init__(self, scorers: list[tuple[RewardFunction, float]]) -> None:
        self._scorers = scorers

    async def compute(self, content: str, ground_truth: str, **kwargs: Any) -> float:
        total = 0.0
        for scorer, weight in self._scorers:
            total += weight * await scorer.compute(content, ground_truth, **kwargs)
        return total
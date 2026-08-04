"""Tests for reward functions."""

from __future__ import annotations

import pytest

from uni_agent.agents.qwenpaw.reward import CompositeReward, GSM8KReward, RewardFunction


class TestGSM8KReward:
    """Tests for the GSM8K math-answer reward scorer."""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Correct answer after #### returns 1.0."""
        reward = GSM8KReward()
        score = await reward.compute(
            "Step 1: ...\nStep 2: ...\n#### 42",
            "42",
        )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_wrong_answer(self):
        """Wrong answer returns 0.0."""
        reward = GSM8KReward()
        score = await reward.compute(
            "The answer is #### 99",
            "42",
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_no_hash_format(self):
        """Fallback: last number in text is used."""
        reward = GSM8KReward()
        score = await reward.compute(
            "The result is 42.",
            "42",
        )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_negative_number(self):
        """Negative numbers are handled correctly."""
        reward = GSM8KReward()
        score = await reward.compute(
            "#### -5",
            "-5",
        )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_decimal_number(self):
        """Decimal numbers are normalized before comparison."""
        reward = GSM8KReward()
        score = await reward.compute(
            "#### 3.14000",
            "3.14",
        )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_comma_separated_number(self):
        """Comma-separated numbers are normalized."""
        reward = GSM8KReward()
        score = await reward.compute(
            "#### 1,000",
            "1000",
        )
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_no_number_in_text(self):
        """Returns 0.0 when no number can be extracted."""
        reward = GSM8KReward()
        score = await reward.compute(
            "I don't know the answer.",
            "42",
        )
        assert score == 0.0


class TestCompositeReward:
    """Tests for weighted composite reward."""

    @pytest.mark.asyncio
    async def test_weighted_average(self):
        """Composite reward computes weighted average."""
        class AlwaysOne:
            async def compute(self, content: str, ground_truth: str, **kwargs):
                return 1.0

        class AlwaysZero:
            async def compute(self, content: str, ground_truth: str, **kwargs):
                return 0.0

        composite = CompositeReward([
            (AlwaysOne(), 0.7),
            (AlwaysZero(), 0.3),
        ])
        score = await composite.compute("any", "any")
        assert score == 0.7
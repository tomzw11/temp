"""Multi-turn conversation runner for QwenPaw.

The default ``run_qwenpaw`` sends a single message and waits for the final
response.  For tasks that require back-and-forth interaction (tool results
feedback, clarification, iterative refinement), this module provides a
multi-turn runner that maintains conversation state across multiple exchanges.

Usage::

    runner = MultiTurnQwenPawRunner(
        client=QwenPawClient(config),
        max_turns=5,
        stop_condition=lambda text: "FINAL ANSWER" in text,
    )
    result = await runner.run(
        session=session_handle,
        initial_message="Solve this step by step...",
        ground_truth="42",
        reward_fn=GSM8KReward(),
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from uni_agent.gateway.session import SessionHandle
from uni_agent.tasks.base import TaskResult

from .agent import QwenPawClient, QwenPawConfig
from .reward import RewardFunction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StopCondition = Callable[[str], bool]
"""A callable that returns True when the conversation should stop."""


# ---------------------------------------------------------------------------
# Multi-turn runner
# ---------------------------------------------------------------------------


class MultiTurnQwenPawRunner:
    """Run a multi-turn QwenPaw conversation against a Gateway session.

    On each turn the runner sends a message (starting with the initial prompt,
    then follow-up messages) and checks a stop condition.  If the condition
    is not met, it calls ``on_turn`` to produce the next message.

    Key methods you can override for custom behaviour:
        - ``_build_next_message`` — produce the next turn's message
        - ``_should_stop`` — decide when to stop
        - ``_compute_reward`` — score the final result
    """

    def __init__(
        self,
        *,
        client: QwenPawClient | None = None,
        config: QwenPawConfig | None = None,
        max_turns: int = 5,
        stop_condition: StopCondition | None = None,
    ) -> None:
        """
        Args:
            client: Pre-configured QwenPawClient (takes precedence over config).
            config: QwenPawConfig to create a client if none is provided.
            max_turns: Maximum conversation turns (safety limit).
            stop_condition: Optional callable that inspects the assistant text
                and returns True when the task is done.
        """
        if client is not None:
            self._client = client
        elif config is not None:
            self._client = QwenPawClient(config)
        else:
            raise ValueError("Either client or config must be provided")

        self._max_turns = max_turns
        self._stop_condition = stop_condition

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        session: SessionHandle,
        initial_message: str,
        ground_truth: str | None = None,
        reward_fn: RewardFunction | None = None,
        **kwargs: Any,
    ) -> TaskResult:
        """Execute a multi-turn conversation.

        Args:
            session: Gateway session handle (provides base_url).
            initial_message: The first user message.
            ground_truth: Expected answer for reward computation.
            reward_fn: Reward function (defaults to GSM8K if ground_truth is set).
            **kwargs: Passed through to reward computation.

        Returns:
            TaskResult with final reward and conversation history.
        """
        if session.base_url is None:
            raise ValueError("multi_turn runner: session.base_url is None")

        turn_history: list[dict[str, Any]] = []
        current_message = initial_message

        for turn in range(1, self._max_turns + 1):
            logger.info("multi_turn: turn %d/%d", turn, self._max_turns)

            result = await self._client.chat(
                current_message,
                model_endpoint=session.base_url,
            )

            turn_record = {
                "turn": turn,
                "user_message": current_message,
                "assistant_content": result.get("content", ""),
                "finish_reason": result.get("finish_reason"),
                "usage": result.get("usage"),
            }
            turn_history.append(turn_record)

            if result.get("finish_reason") == "error":
                return TaskResult(
                    reward=0.0,
                    finished=False,
                    extra_info={
                        "error": result.get("error"),
                        "turns": turn_history,
                    },
                )

            assistant_text = result.get("content", "")

            if self._should_stop(assistant_text, turn):
                break

            current_message = self._build_next_message(assistant_text, turn)

        # Compute reward
        if reward_fn is not None and ground_truth is not None:
            final_content = turn_history[-1]["assistant_content"] if turn_history else ""
            reward = await reward_fn.compute(final_content, ground_truth, **kwargs)
        else:
            reward = 1.0 if turn_history and turn_history[-1].get("finish_reason") == "stop" else 0.0

        return TaskResult(
            reward=reward,
            finished=True,
            extra_info={
                "turns": turn_history,
                "total_turns": len(turn_history),
            },
        )

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _should_stop(self, assistant_text: str, turn: int) -> bool:
        """Decide whether to stop the conversation.

        Override to implement custom stop logic (e.g. regex match on final
        answer markers, or an LLM-based judge).
        """
        if self._stop_condition is not None:
            return self._stop_condition(assistant_text)
        return False

    def _build_next_message(self, assistant_text: str, turn: int) -> str:
        """Build the next user message based on the assistant's response.

        Override to implement task-specific follow-up prompts (e.g.
        "Please continue", "Are you sure?", tool result feedback).
        """
        return "Please continue."
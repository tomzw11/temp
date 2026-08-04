"""Prefix KV-Cache via vLLM built-in prefix caching.

Problem
-------
In QwenPaw's ReAct loop every LLM call re-computes the KV-cache for the full
prompt prefix (system prompt + task + tool definitions).  With 5 turns and a
2 000 token prefix that is ~4× wasted compute.

Solution
--------
vLLM already supports ``--enable-prefix-caching``, which automatically detects
shared prefixes across requests and reuses the cached KV.  No custom headers
or cache identifiers are needed — vLLM matches by token hash.

The only missing piece is **Gateway session routing affinity**: the Gateway
must route all requests belonging to the same session to the same vLLM worker.
Otherwise Turn 1 hits worker A (which caches the prefix) and Turn 2 hits
worker B (which has no cache).

QwenPaw runner side: **zero changes needed**.  Each turn's request already
carries the same prompt prefix naturally — vLLM handles the rest.

    ┌─── QwenPaw ReAct loop ───────────────────────────────────────┐
    │                                                               │
    │  Turn 1: [sys + task + tools + t1]  → Gateway → worker A     │
    │                                            │                  │
    │                                      vLLM hash(syn+task+tools)│
    │                                      → cache KV               │
    │                                                               │
    │  Turn 2: [sys + task + tools + t1 + t2]  → Gateway            │
    │                                              │                │
    │                                        must route to worker A │
    │                                              │                │
    │                                      vLLM hash(syn+task+tools)│
    │                                      → HIT → reuse KV         │
    │                                                               │
    │  Turn 3: [sys + task + tools + t1 + t2 + t3]  → Gateway      │
    │                                              │                │
    │                                        must route to worker A │
    │                                              │                │
    │                                      vLLM hash(syn+task+tools)│
    │                                      → HIT → reuse KV         │
    └───────────────────────────────────────────────────────────────┘

Estimated throughput improvement: 2–5× for multi-turn ReAct loops
(depends on prefix-to-total token ratio and turn count).

Gateway changes (implemented)
-----------------------------
The following changes have been applied to ``uni_agent/gateway/``.

1. **GatewayActorConfig** — new ``enable_prefix_caching`` flag
   (``uni_agent/gateway/config.py``)::

        enable_prefix_caching: bool = False

2. **GatewayActor.__init__** — store ``_enable_prefix_cache`` and
   ``_session_workers`` dict (``uni_agent/gateway/gateway.py``)::

        self._enable_prefix_cache = config.enable_prefix_caching
        self._session_workers: dict[str, int] = {}

3. **GatewayActor._get_worker_for_session()** — assign and retrieve
   worker affinity for a session (``uni_agent/gateway/gateway.py``)::

        def _get_worker_for_session(self, session_id: str) -> int | None:
            if not self._enable_prefix_cache:
                return None
            if session_id in self._session_workers:
                return self._session_workers[session_id]
            worker_count = getattr(self._backend, "worker_count", 1)
            worker_id = hash(session_id) % worker_count
            self._session_workers[session_id] = worker_id
            return worker_id

4. **GatewayActor._handle_openai_chat_completions / _handle_anthropic_messages**
   — pass ``worker_id`` to ``session.run_generation()``
   (``uni_agent/gateway/gateway.py``)::

        outcome = await session.run_generation(
            internal, self._backend,
            worker_id=self._get_worker_for_session(session_id),
        )

5. **GatewaySession.run_generation** — accept and forward ``worker_id``
   to the backend (``uni_agent/gateway/session/session.py``)::

        async def run_generation(
            self, request, backend, *, worker_id: int | None = None
        ) -> GenerationOutcome:
            ...
            generate_kwargs: dict[str, Any] = {...}
            if worker_id is not None:
                generate_kwargs["worker_id"] = worker_id
            output = await backend.generate(**generate_kwargs)

6. **GatewayActor.clear_prefix_cache()** — clear all affinities on
   weight update (``uni_agent/gateway/gateway.py``)::

        async def clear_prefix_cache(self) -> None:
            self._session_workers.clear()

7. **verl LLMServerClient** — needs to accept ``worker_id`` in
   ``generate()`` and route to the specified worker (verl side, not
   in this repo).  If the backend does not yet support ``worker_id``,
   it is passed as a kwarg and ignored — the feature degrades
   gracefully.

   Pseudo-code for verl::

        class LLMServerClient:
            @property
            def worker_count(self) -> int:
                return len(self._workers)

            async def generate(self, ..., worker_id: int | None = None, **kwargs):
                if worker_id is not None and worker_id < len(self._workers):
                    worker = self._workers[worker_id]
                else:
                    worker = self._pick_worker()
                return await worker.generate(...)

Why no QwenPaw-side changes?
----------------------------
- vLLM's prefix caching is **token-hash-based**, not identifier-based.
  It automatically detects ``[sys + task]`` in every request and reuses
  the cached KV.
- Each turn's request naturally contains the same prefix — no special
  header or field is needed.
- The only requirement is that all turns in a session hit the same
  vLLM worker, which is purely a Gateway routing concern.
"""

from __future__ import annotations

# This module is intentionally code-free on the QwenPaw side.
# vLLM's built-in prefix caching (--enable-prefix-caching) handles
# KV reuse automatically via token hash matching.
# The only missing piece is Gateway session routing affinity,
# documented above.
#
# See also:
#   https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html
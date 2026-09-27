from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .actions import ACTION_DESCRIPTIONS, Action
from .state import MarioSnapshot

DEFAULT_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v4/ai"
DEFAULT_GATEWAY_MODEL = "typesafe-ai/jev"
GATEWAY_MAX_RETRIES = 2
REALTIME_TIMEOUT_SECONDS = 2.0


class PolicyUnavailableError(RuntimeError):
    """The model service failed transiently; callers may keep the current action and retry."""


@dataclass(frozen=True)
class Decision:
    action: Action
    confidence: float
    probabilities: Mapping[str, float]
    latency_ms: float
    jump_needed_probability: float | None = None
    danger_score: float | None = None


class Policy(Protocol):
    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision: ...


class _GatewayAnswers:
    """Adapts a Vercel AI Gateway evaluation response to the TypeSafe SDK answer shape."""

    def __init__(self, answers: Mapping[str, Mapping[str, Any]]) -> None:
        self._answers = answers

    def choice(self, question_id: str) -> tuple[str, float, dict[str, float]]:
        answer = self._answers[question_id]
        choice = str(answer["choice"])
        probabilities = {str(k): float(v) for k, v in answer.get("probabilities", {}).items()}
        return choice, probabilities.get(choice, 1.0), probabilities

    def noul(self, question_id: str) -> float:
        return float(self._answers[question_id]["probability"])

    def score(self, question_id: str) -> float:
        return float(self._answers[question_id]["score"])


class _GatewayClient:
    """Calls Jev through Vercel AI Gateway, mirroring `experimental_evaluate` from the AI SDK."""

    def __init__(
        self, api_key: str, model: str, base_url: str, timeout: float = REALTIME_TIMEOUT_SECONDS
    ) -> None:
        import httpx2

        self._httpx = httpx2
        self._url = f"{base_url.rstrip('/')}/evaluation-model"
        self._http = httpx2.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "ai-gateway-protocol-version": "0.0.1",
                "ai-gateway-auth-method": "api-key",
                "ai-evaluation-model-specification-version": "4",
                "ai-model-id": model,
            },
        )

    def close(self) -> None:
        self._http.close()

    def evaluate(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> _GatewayAnswers:
        # The gateway spells TypeSafe's `noul` question type as `boolean`.
        body = {
            "state": state,
            "questions": {
                question_id: {**q, "type": "boolean" if q["type"] == "noul" else q["type"]}
                for question_id, q in questions.items()
            },
        }
        # Retry transient failures like the AI SDK does (maxRetries=2), with a short backoff
        # because decisions are latency-sensitive.
        for attempt in range(GATEWAY_MAX_RETRIES + 1):
            last_attempt = attempt == GATEWAY_MAX_RETRIES
            try:
                response = self._http.post(self._url, json=body)
            except self._httpx.TransportError as exc:
                if last_attempt:
                    raise PolicyUnavailableError(f"AI Gateway request failed: {exc}") from exc
            else:
                if response.status_code < 400:
                    return _GatewayAnswers(response.json()["answers"])
                retryable = response.status_code in (408, 409, 429) or response.status_code >= 500
                message = f"AI Gateway evaluation failed ({response.status_code}): {response.text}"
                if not retryable:
                    raise RuntimeError(message)
                if last_attempt:
                    raise PolicyUnavailableError(message)
            time.sleep(0.1 * 2**attempt)
        raise AssertionError("unreachable")


class TypeSafePolicy:
    """Jev policy. Uses `TYPESAFE_API_KEY` directly, or `AI_GATEWAY_API_KEY` /
    `VERCEL_API_KEY` via Vercel AI Gateway."""

    def __init__(self) -> None:
        self._gateway: _GatewayClient | None = None
        self._client: Any = None
        gateway_key = (
            os.environ.get("AI_GATEWAY_API_KEY", "").strip()
            or os.environ.get("VERCEL_API_KEY", "").strip()
        )
        if gateway_key and not os.environ.get("TYPESAFE_API_KEY", "").strip():
            self._gateway = _GatewayClient(
                api_key=gateway_key,
                model=os.environ.get("AI_GATEWAY_MODEL", "").strip() or DEFAULT_GATEWAY_MODEL,
                base_url=os.environ.get("AI_GATEWAY_BASE_URL", "").strip()
                or DEFAULT_GATEWAY_BASE_URL,
            )
            return
        try:
            from typesafe_sdk import (
                RetryPolicy,
                TypeSafeAPIConnectionError,
                TypeSafeClient,
                TypeSafeInternalServerError,
                TypeSafeRateLimitError,
            )
        except ImportError as exc:
            raise RuntimeError(
                "typesafe-sdk is not installed. Install the project before using Jev."
            ) from exc
        self._transient_errors = (
            TypeSafeAPIConnectionError,
            TypeSafeInternalServerError,
            TypeSafeRateLimitError,
        )
        # The SDK defaults (0.5 s backoff, 10 s timeout) suit batch work, not a game: a stalled
        # request would hold a slot for seconds while newer requests are already in flight.
        self._client = TypeSafeClient(
            timeout=REALTIME_TIMEOUT_SECONDS,
            retry=RetryPolicy(max_retries=1, backoff_initial=0.05, backoff_max=0.1),
        )

    def close(self) -> None:
        if self._gateway is not None:
            self._gateway.close()
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _answer(response: Any, question_id: str, typed_collection: str) -> Any:
        collection = getattr(response, typed_collection, None)
        if collection is not None and question_id in collection:
            return collection[question_id]
        answers = getattr(response, "answers", None)
        if answers is not None and question_id in answers:
            return answers[question_id]
        raise KeyError(f"TypeSafe response omitted {question_id!r}")

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        criteria = {action.value: ACTION_DESCRIPTIONS[action] for action in actions}
        questions = {
            "next_action": {
                "type": "choice",
                "instructions": {
                    "question": "Which controller macro should Mario commit to next?",
                    "goal": "Advance toward the stage flag while avoiding death.",
                    "timing": "The selected action is held for at least 8 emulator frames.",
                    "geometry": (
                        "Use `terrain.observation_reliability`. While airborne, prefer "
                        "`terrain.last_grounded_preview` over low-reliability current geometry. "
                        "A trusted obstacle or gap within three tiles requires a forward jump."
                    ),
                    "trajectory": (
                        "Use `trajectory`. If `crossing_known_gap` is true, preserve forward "
                        "speed and keep a forward jump held while rising. Do not switch to noop "
                        "or left over a gap."
                    ),
                    "stall": (
                        "If `episode.stalled_frames` is increasing and "
                        "`recent_control.outcome` is blocked, the current non-jump action failed."
                    ),
                    "enemy_timing": (
                        "Use `hazard` projections, not distance alone. Code has already accounted "
                        "for inference delay, action cadence, and the frames needed to clear an "
                        "enemy. If `jump_must_start_this_decision` is true, choose a forward jump "
                        "now; another right-run decision will miss the takeoff deadline. If "
                        "`contact_within_reaction_horizon` is true, also jump immediately. If "
                        "`will_land_before_contact` is true, the current jump will not clear the "
                        "enemy and another takeoff will be needed after landing. Use "
                        "`upcoming_enemies` and their spacing to avoid landing on a second or "
                        "third enemy hidden behind the nearest one."
                    ),
                    "delay": (
                        "`reaction_timing` describes how far the world moves before this choice "
                        "takes effect. Judge urgency from projected rather than current distance."
                    ),
                },
                "criteria": criteria,
            },
            "jump_needed": {
                "type": "noul",
                "instructions": (
                    "Do trusted `terrain`, projected `hazard`, `trajectory`, and "
                    "`player.jump_phase` indicate that a forward jump should begin or remain "
                    "held now? `hazard.jump_must_start_this_decision=true` is unambiguously yes. "
                    "Also count a trusted obstacle/gap within three tiles, immediate projected "
                    "contact, or a rising jump over a known gap as yes."
                ),
            },
            "danger": {
                "type": "score",
                "instructions": "How dangerous is Mario's immediate situation?",
                "criteria": [
                    "Safe open movement",
                    "Potential obstacle or enemy soon",
                    "Immediate collision, fall, or enemy threat",
                ],
            },
        }
        started = time.perf_counter()
        if self._gateway is not None:
            answers = self._gateway.evaluate(snapshot.to_state(), questions)
            latency_ms = (time.perf_counter() - started) * 1000
            choice, confidence, probabilities = answers.choice("next_action")
            return Decision(
                action=Action(choice),
                confidence=confidence,
                probabilities=probabilities,
                latency_ms=latency_ms,
                jump_needed_probability=answers.noul("jump_needed"),
                danger_score=answers.score("danger"),
            )
        try:
            response = self._client.system_one(state=snapshot.to_state(), questions=questions)
        except self._transient_errors as exc:
            raise PolicyUnavailableError(f"TypeSafe request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        action_answer = self._answer(response, "next_action", "choices")
        jump_answer = self._answer(response, "jump_needed", "nouls")
        danger_answer = self._answer(response, "danger", "scores")
        action = Action(str(action_answer.choice))
        probabilities = {
            str(key): float(value) for key, value in dict(action_answer.probabilities).items()
        }
        return Decision(
            action=action,
            confidence=float(action_answer.confidence),
            probabilities=probabilities,
            latency_ms=latency_ms,
            jump_needed_probability=float(jump_answer.noul),
            danger_score=float(danger_answer.score),
        )


class HeuristicPolicy:
    """Offline smoke-test policy; not intended as the Mario benchmark baseline."""

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        allowed = set(actions)
        action = Action.RIGHT_RUN if Action.RIGHT_RUN in allowed else actions[0]
        if snapshot.stalled_steps >= 2 and Action.RIGHT_RUN_JUMP in allowed:
            action = Action.RIGHT_RUN_JUMP
        return Decision(
            action=action,
            confidence=1.0,
            probabilities={candidate.value: float(candidate == action) for candidate in actions},
            latency_ms=0.0,
        )

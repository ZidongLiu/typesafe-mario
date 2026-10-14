from __future__ import annotations

import json
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
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_CLAUDE_MODEL = "claude-opus-5"
# An LLM answering with thinking takes seconds, not milliseconds; allow for it.
CLAUDE_TIMEOUT_SECONDS = 20.0
CLAUDE_MAX_TOKENS = 4096
BACKENDS = ("auto", "typesafe", "vercel", "local", "claude")


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
    source: str | None = None


class Policy(Protocol):
    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision: ...


def build_questions(actions: Sequence[Action]) -> dict[str, dict[str, Any]]:
    """The typed questions every backend answers, so their results are comparable."""
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
                    "A trusted obstacle within three tiles requires a forward jump. For gaps, "
                    "jump only when `terrain.gap_takeoff_now` is true; until then keep moving "
                    "toward the edge, because a jump started early falls into the gap. If a step "
                    "comes before the gap, hop onto it with a short jump (choose `right` once "
                    "airborne so you land on the step), then take off from its edge."
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
                "Also count a trusted obstacle within three tiles, `terrain.gap_takeoff_now`, "
                "immediate projected contact, or a rising jump over a known gap as yes."
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
    return questions


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
    """Jev-style System One policy: TypeSafe directly, via Vercel AI Gateway, or a local server
    that speaks the same `/v1/systemone` API (e.g. a self-hosted Jev-compatible wrapper)."""

    def __init__(
        self,
        backend: str = "typesafe",
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._gateway: _GatewayClient | None = None
        self._client: Any = None
        if backend == "vercel":
            gateway_key = (
                os.environ.get("AI_GATEWAY_API_KEY", "").strip()
                or os.environ.get("VERCEL_API_KEY", "").strip()
            )
            if not gateway_key:
                raise RuntimeError("Set AI_GATEWAY_API_KEY or VERCEL_API_KEY to use Vercel.")
            model = model or os.environ.get("AI_GATEWAY_MODEL", "").strip() or DEFAULT_GATEWAY_MODEL
            self._gateway = _GatewayClient(
                api_key=gateway_key,
                model=model,
                base_url=base_url
                or os.environ.get("AI_GATEWAY_BASE_URL", "").strip()
                or DEFAULT_GATEWAY_BASE_URL,
            )
            self.source = f"vercel:{model}"
            return
        if backend not in ("typesafe", "local"):
            raise ValueError(f"TypeSafePolicy does not handle backend {backend!r}")
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
        api_key = None
        if backend == "local":
            # The SDK insists on a key; a local server only checks it if configured to.
            api_key = (
                os.environ.get("LOCAL_JEV_API_KEY", "").strip()
                or os.environ.get("SHIM_TOKEN", "").strip()
                or "local"
            )
            base_url = (
                base_url
                or os.environ.get("LOCAL_JEV_BASE_URL", "").strip()
                or DEFAULT_LOCAL_BASE_URL
            )
            model = model or os.environ.get("LOCAL_JEV_MODEL", "").strip() or None
        # The SDK defaults (0.5 s backoff, 10 s timeout) suit batch work, not a game: a stalled
        # request would hold a slot for seconds while newer requests are already in flight.
        self._client = TypeSafeClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=REALTIME_TIMEOUT_SECONDS,
            retry=RetryPolicy(max_retries=1, backoff_initial=0.05, backoff_max=0.1),
        )
        config = self._client._config
        self.source = f"{backend}:{config.default_model}@{config.base_url}"

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
        questions = build_questions(actions)
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
                source=self.source,
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
            source=self.source,
        )


class ClaudePolicy:
    """Answers the same questions with a Claude model, for comparison with System One models.

    An LLM has no native choice/score distributions, so like the AI SDK's language-model
    evaluation adapters it returns a choice plus prompted estimates (not calibrated
    probabilities) for its confidence, the boolean, and the score.
    """

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, *, effort: str = "low") -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                'The Claude backend needs the anthropic SDK: pip install -e ".[claude]"'
            ) from exc
        self._anthropic = anthropic
        # Credentials resolve from ANTHROPIC_API_KEY (loaded from .env) or an `ant` profile.
        self._client = anthropic.Anthropic(timeout=CLAUDE_TIMEOUT_SECONDS, max_retries=1)
        self._model = model
        self._effort = effort
        self._system_prompts: dict[tuple[Action, ...], str] = {}
        self.source = f"claude:{model}:{effort}"

    def close(self) -> None:
        self._client.close()

    def _system_prompt(self, actions: Sequence[Action]) -> str:
        key = tuple(actions)
        if key not in self._system_prompts:
            questions = json.dumps(build_questions(actions), indent=1)
            self._system_prompts[key] = (
                "You control Mario in Super Mario Bros. in real time. Each user message is the "
                "current game state as JSON. Answer the questions below using only that state.\n\n"
                f"Questions:\n{questions}\n\n"
                "Reply with JSON only:\n"
                "- next_action: one option key from next_action.criteria.\n"
                "- next_action_confidence: your probability (0-1) that next_action is the best "
                "option.\n"
                "- jump_needed: your probability (0-1) that the answer to jump_needed is yes.\n"
                "- danger: a number from 0 to 2 on the danger rubric, where 0 is its first level."
            )
        return self._system_prompts[key]

    def _request(self, **params: Any) -> Any:
        # Server-side fallbacks re-run a (rare) safety refusal on another model in the same call;
        # they are offered for Opus 5 / Fable models only.
        if self._model.startswith(("claude-opus-5", "claude-fable-5")):
            return self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params
            )
        return self._client.messages.create(**params)

    def choose(self, snapshot: MarioSnapshot, actions: Sequence[Action]) -> Decision:
        schema = {
            "type": "object",
            "properties": {
                "next_action": {"type": "string", "enum": [action.value for action in actions]},
                "next_action_confidence": {"type": "number"},
                "jump_needed": {"type": "number"},
                "danger": {"type": "number"},
            },
            "required": ["next_action", "next_action_confidence", "jump_needed", "danger"],
            "additionalProperties": False,
        }
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if not self._model.startswith("claude-haiku"):  # Haiku 4.5 rejects `effort`
            output_config["effort"] = self._effort
        anthropic = self._anthropic
        started = time.perf_counter()
        try:
            response = self._request(
                model=self._model,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": self._system_prompt(actions),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(snapshot.to_state(), separators=(",", ":")),
                    }
                ],
                output_config=output_config,
            )
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            raise PolicyUnavailableError(f"Claude request failed: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise PolicyUnavailableError(f"Claude request failed: {exc}") from exc
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        if response.stop_reason in ("refusal", "max_tokens"):
            raise PolicyUnavailableError(
                f"Claude returned no usable answer: {response.stop_reason}"
            )
        answer = json.loads(next(block.text for block in response.content if block.type == "text"))
        action = Action(answer["next_action"])
        confidence = min(1.0, max(0.0, float(answer["next_action_confidence"])))
        return Decision(
            action=action,
            confidence=confidence,
            probabilities={action.value: confidence},
            latency_ms=latency_ms,
            jump_needed_probability=min(1.0, max(0.0, float(answer["jump_needed"]))),
            danger_score=min(2.0, max(0.0, float(answer["danger"]))),
            source=self.source,
        )


def create_policy(
    backend: str = "auto",
    *,
    model: str | None = None,
    base_url: str | None = None,
    claude_effort: str = "low",
) -> TypeSafePolicy | ClaudePolicy:
    """Build the decision policy for a `--backend` choice."""
    if backend == "auto":
        has_gateway_key = bool(
            os.environ.get("AI_GATEWAY_API_KEY", "").strip()
            or os.environ.get("VERCEL_API_KEY", "").strip()
        )
        has_typesafe_key = bool(os.environ.get("TYPESAFE_API_KEY", "").strip())
        backend = "vercel" if has_gateway_key and not has_typesafe_key else "typesafe"
    if backend == "claude":
        return ClaudePolicy(model or DEFAULT_CLAUDE_MODEL, effort=claude_effort)
    return TypeSafePolicy(backend, model=model, base_url=base_url)


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

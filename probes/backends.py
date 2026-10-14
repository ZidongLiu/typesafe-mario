"""One multiple-choice interface over Jev (direct, preview, Vercel), a local server, and Claude.

Every backend gets the same input: a JSON `context`, a `question`, and lettered `options`.
System One models answer by reading option probabilities; Claude writes a structured answer.

Model specs:
    jev                         Jev direct (jev-latest)
    jev:jev-preview             another hosted Jev model
    vercel                      Jev through Vercel AI Gateway (typesafe-ai/jev)
    local                       local Jev-compatible server (LOCAL_JEV_BASE_URL or :3000 below)
    local:http://host:port      local server at an explicit URL
    claude:claude-opus-5        a Claude model, default effort `low`
    claude:claude-opus-5@high   ... with an explicit effort
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_LOCAL_URL = "http://192.168.1.161:3000"
GATEWAY_URL = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
# Long inputs take a while to prefill on an LLM; don't cut the 100k-token runs short.
TIMEOUT_SECONDS = 300.0


@dataclass
class Answer:
    choice: str | None
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    # Claude only: time until the first output token arrived (prefill + queueing).
    first_token_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class Backend(Protocol):
    name: str

    def ask(self, context: Any, question: str, options: dict[str, str]) -> Answer: ...

    def close(self) -> None: ...


def load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from `.env` without overriding variables already set."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def system_one_questions(question: str, options: dict[str, str]) -> dict[str, Any]:
    """The `questions` object sent to Jev and local servers: one `choice` keyed by letter."""
    return {
        "answer": {"type": "choice", "instructions": {"question": question}, "criteria": options}
    }


def claude_prompt(context: Any, question: str, options: dict[str, str]) -> str:
    """The single user message sent to Claude (there is no system prompt)."""
    return (
        f"{json.dumps(context, ensure_ascii=False)}\n\n"
        f"Question: {question}\n"
        + "\n".join(f"{key}. {text}" for key, text in options.items())
        + "\n\nAnswer with the letter of the single best option."
    )


def claude_answer_schema(options: dict[str, str]) -> dict[str, Any]:
    """Structured-output schema that limits Claude's reply to one option letter."""
    return {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": list(options)}},
        "required": ["answer"],
        "additionalProperties": False,
    }


class JevBackend:
    """TypeSafe's hosted Jev, or any server speaking `POST /v1/systemone`, via the SDK."""

    def __init__(self, name: str, *, model: str | None, base_url: str | None, api_key: str):
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        self.name = name
        self._model = model
        self._client = TypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            timeout=TIMEOUT_SECONDS,
            retry=RetryPolicy(max_retries=3, backoff_initial=0.5, backoff_max=4.0),
        )

    def ask(self, context: Any, question: str, options: dict[str, str]) -> Answer:
        started = time.perf_counter()
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = self._client.system_one(
                state=context, questions=system_one_questions(question, options), **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - recorded per item; one bad item must not end a run
            return Answer(None, latency_ms=_ms(started), error=f"{type(exc).__name__}: {exc}")
        latency_ms = _ms(started)
        answer = response.answers["answer"]
        usage = getattr(response, "usage", None)
        return Answer(
            choice=answer.choice,
            probabilities=dict(answer.probabilities),
            latency_ms=latency_ms,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )

    def close(self) -> None:
        self._client.close()


class VercelJevBackend:
    """Jev through Vercel AI Gateway's evaluation-model endpoint."""

    def __init__(self, name: str, *, model: str, api_key: str):
        import httpx2

        self.name = name
        self._http = httpx2.Client(
            timeout=TIMEOUT_SECONDS,
            headers={
                "Authorization": f"Bearer {api_key}",
                "ai-gateway-protocol-version": "0.0.1",
                "ai-gateway-auth-method": "api-key",
                "ai-evaluation-model-specification-version": "4",
                "ai-model-id": model,
            },
        )

    def ask(self, context: Any, question: str, options: dict[str, str]) -> Answer:
        body = {"state": context, "questions": system_one_questions(question, options)}
        started = time.perf_counter()
        for attempt in range(4):
            try:
                response = self._http.post(GATEWAY_URL, json=body)
            except Exception as exc:  # noqa: BLE001 - retried, then recorded
                error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400:
                    break
                error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in (408, 409, 429) and response.status_code < 500:
                    return Answer(None, latency_ms=_ms(started), error=error)
            if attempt == 3:
                return Answer(None, latency_ms=_ms(started), error=error)
            time.sleep(0.5 * 2**attempt)
            started = time.perf_counter()  # time the successful attempt, not the backoff
        latency_ms = _ms(started)
        payload = response.json()
        answer = payload["answers"]["answer"]
        usage = payload.get("usage") or {}
        return Answer(
            choice=str(answer["choice"]),
            probabilities={k: float(v) for k, v in answer.get("probabilities", {}).items()},
            latency_ms=latency_ms,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
        )

    def close(self) -> None:
        self._http.close()


class ClaudeBackend:
    """A Claude model answering with a single option letter via structured output."""

    def __init__(self, name: str, *, model: str, effort: str | None):
        import anthropic

        self.name = name
        self._model = model
        self._effort = effort
        self._client = anthropic.Anthropic(timeout=TIMEOUT_SECONDS, max_retries=4)

    def ask(self, context: Any, question: str, options: dict[str, str]) -> Answer:
        prompt = claude_prompt(context, question, options)
        schema = claude_answer_schema(options)
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self._effort and not self._model.startswith("claude-haiku-4"):  # Haiku 4.5: no effort
            output_config["effort"] = self._effort
        started = time.perf_counter()
        first_token_ms = None
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
                output_config=output_config,
            ) as stream:
                for event in stream:
                    if first_token_ms is None and event.type == "content_block_start":
                        first_token_ms = _ms(started)
                message = stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - recorded per item
            return Answer(None, latency_ms=_ms(started), error=f"{type(exc).__name__}: {exc}")
        latency_ms = _ms(started)
        usage = message.usage
        input_tokens = (
            usage.input_tokens
            + (usage.cache_read_input_tokens or 0)
            + (usage.cache_creation_input_tokens or 0)
        )
        text = next((block.text for block in message.content if block.type == "text"), None)
        if message.stop_reason != "end_turn" or text is None:
            return Answer(
                None,
                latency_ms=latency_ms,
                first_token_ms=first_token_ms,
                input_tokens=input_tokens,
                output_tokens=usage.output_tokens,
                error=f"stop_reason={message.stop_reason}",
            )
        choice = json.loads(text)["answer"]
        return Answer(
            choice=choice,
            probabilities={choice: 1.0},
            latency_ms=latency_ms,
            first_token_ms=first_token_ms,
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
        )

    def close(self) -> None:
        self._client.close()


def make_backend(spec: str) -> Backend:
    load_dotenv()
    kind, _, arg = spec.partition(":")
    if kind == "jev":
        return JevBackend(
            spec, model=arg or "jev-latest", base_url=None, api_key=os.environ["TYPESAFE_API_KEY"]
        )
    if kind == "vercel":
        key = os.environ.get("AI_GATEWAY_API_KEY") or os.environ["VERCEL_API_KEY"]
        return VercelJevBackend(spec, model=arg or "typesafe-ai/jev", api_key=key)
    if kind == "local":
        url = arg or os.environ.get("LOCAL_JEV_BASE_URL") or DEFAULT_LOCAL_URL
        key = os.environ.get("LOCAL_JEV_API_KEY") or os.environ.get("SHIM_TOKEN") or "local"
        return JevBackend(spec, model=os.environ.get("LOCAL_JEV_MODEL"), base_url=url, api_key=key)
    if kind == "claude":
        model, _, effort = arg.partition("@")
        return ClaudeBackend(spec, model=model, effort=effort or "low")
    raise ValueError(f"Unknown model spec {spec!r}; see probes/backends.py")


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000

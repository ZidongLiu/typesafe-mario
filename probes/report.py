"""Summarize probe results as markdown tables (also written to probes/results/report.md).

python -m probes.report
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_TITLES = {
    "arc": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "mmlu": "MMLU",
    "mmlu-pro": "MMLU-Pro",
}
# Models shown in the examples (Jev via Vercel is left out: it answers like Jev).
EXAMPLE_MODELS = {
    "jev": "Jev",
    "local": "Local",
    "claude:claude-haiku-4-5": "Haiku 4.5",
    "claude:claude-sonnet-5": "Sonnet 5",
    "claude:claude-opus-5": "Opus 5",
}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _median(values: list[float]) -> str:
    return f"{statistics.median(values):,.0f}" if values else "–"


def _p90(values: list[float]) -> str:
    if not values:
        return "–"
    ordered = sorted(values)
    return f"{ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]:,.0f}"


def accuracy_table() -> list[str]:
    files = sorted((RESULTS_DIR / "accuracy").glob("*.jsonl"))
    if not files:
        return []
    lines = [
        "## Accuracy",
        "",
        "| Dataset | Model | Accuracy | 95% CI | Answered | Errors | Chance | Median latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for path in files:
        dataset, _, model = path.stem.partition("__")
        rows = _read(path)
        latest: dict[str, dict] = {}
        for row in rows:  # a later successful retry replaces an earlier error
            if row["id"] not in latest or latest[row["id"]]["error"] is not None:
                latest[row["id"]] = row
        answered = [row for row in latest.values() if row["error"] is None]
        model = rows[0].get("model", model)
        errors = len(latest) - len(answered)
        correct = sum(row["correct"] for row in answered)
        low, high = _wilson(correct, len(answered))
        chance = statistics.mean(1 / row["n_options"] for row in latest.values())
        accuracy = correct / len(answered) if answered else 0.0
        lines.append(
            f"| {dataset} | {model} | **{accuracy:.1%}** | {low:.0%}–{high:.0%} | {len(answered)} "
            f"| {errors} | {chance:.0%} | {_median([r['latency_ms'] for r in answered])} |"
        )
    return lines + [""]


def latency_table() -> list[str]:
    files = sorted((RESULTS_DIR / "latency").glob("*.jsonl"))
    if not files:
        return []
    lines = [
        "## Latency vs. input size",
        "",
        (
            "Size is the target in ~tokens; *reported tokens* is what the backend billed. "
            "First token is Claude's time to first output (mostly input processing)."
        ),
        "",
        "| Model | Size | Reported tokens | Median ms | p90 ms | First token ms | Needle correct |",
        "|---|---|---|---|---|---|---|",
    ]
    limits = []
    for path in files:
        rows = _read(path)
        by_size: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            if "limit_tokens" in row:
                limits.append(f"- {row['model']}: largest accepted input ≈ {row['limit_tokens']:,}")
            else:
                by_size[row["size"]].append(row)
        for size in sorted(by_size):
            group = by_size[size]
            ok = [row for row in group if row["error"] is None]
            model = group[0]["model"]
            if not ok:
                reason = "too long" if any(row["too_long"] for row in group) else "errors"
                lines.append(f"| {model} | {size:,} | – | {reason} | | | |")
                continue
            tokens = [row["input_tokens"] for row in ok if row["input_tokens"]]
            first = [row["first_token_ms"] for row in ok if row["first_token_ms"] is not None]
            correct = sum(row["correct"] for row in ok)
            lines.append(
                f"| {model} | {size:,} | {_median(tokens)} | {_median([r['latency_ms'] for r in ok])}"
                f" | {_p90([r['latency_ms'] for r in ok])} | {_median(first)} "
                f"| {correct}/{len(ok)} |"
            )
    if limits:
        lines += ["", "Input limits found with `--find-limit`:", "", *limits]
    return lines + [""]


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _example(title: str, item, answers: dict[str, dict]) -> list[str]:
    context = item.context
    if "context" in context:  # HellaSwag
        prompt = f"*{context['activity']}:* {context['context']} …"
    else:
        topic = context.get("subject") or context.get("category")
        prompt = (f"*{topic}:* " if topic else "") + context["question"]
    lines = [f"**{title}**", "", f"> {_clip(prompt, 500)}", ""]
    for key, text in item.options.items():
        option = f"{key}. {_clip(text, 140)}"
        lines.append(f"- **{option}** ✅" if key == item.answer else f"- {option}")
    cells = []
    for model in EXAMPLE_MODELS:
        row = answers[model]
        mark = "✓" if row["correct"] else "✗"
        cell = f"{row['choice']} {mark}"
        if model in ("jev", "local"):  # only System One models give a real probability
            cell += f" ({row['probabilities'][row['choice']]:.0%})"
        cells.append(cell)
    lines += [
        "",
        "| " + " | ".join(EXAMPLE_MODELS.values()) + " |",
        "|" + "---|" * len(EXAMPLE_MODELS),
        "| " + " | ".join(cells) + " |",
        "",
    ]
    return lines


def _dataset_examples(name: str, title: str) -> list[str]:
    from . import datasets

    answers: dict[str, dict[str, dict]] = defaultdict(dict)
    for model in EXAMPLE_MODELS:
        safe = model.replace(":", "_").replace("/", "_").replace("@", "_")
        path = RESULTS_DIR / "accuracy" / f"{name}__{safe}.jsonl"
        if not path.exists():
            return []
        for row in _read(path):
            if row["error"] is None:
                answers[row["id"]][model] = row
    items = {item.id: item for item in datasets.load(name)}
    # Items every model answered, shortest first so the examples stay readable.
    complete = sorted(
        (
            items[item_id]
            for item_id, by_model in answers.items()
            if len(by_model) == len(EXAMPLE_MODELS)
        ),
        key=lambda it: len(json.dumps(it.context)) + len(json.dumps(it.options)),
    )
    picks = [
        ("Every model right", lambda right: all(right.values())),
        ("Models split, Jev right", lambda right: right["jev"] and not all(right.values())),
        ("Jev wrong", lambda right: not right["jev"]),
    ]
    lines = [f"### {title}", ""]
    for label, test in picks:
        for item in complete:
            if test({model: row["correct"] for model, row in answers[item.id].items()}):
                lines += _example(label, item, answers[item.id])
                break
    return lines


def examples_section() -> list[str]:
    """Three short questions per benchmark: all models right, a split, and a Jev miss."""
    lines = [
        "## Example questions",
        "",
        (
            "A few of the actual items, with each model's pick. ✅ marks the correct option; the"
            " percentage is the System One model's probability for its pick. The shortest item"
            " of each kind is shown, so these are easier than a typical question."
        ),
        "",
    ]
    for name, title in DATASET_TITLES.items():
        lines += _dataset_examples(name, title)
    return lines


def formatting_section() -> list[str]:
    """How one item is packaged for each kind of model, rendered by the backends' own code."""
    from . import datasets
    from .backends import claude_answer_schema, claude_prompt, system_one_questions

    arc = datasets.load("arc")
    item = next((it for it in arc if "pH level for pure water" in it.context["question"]), arc[0])
    system_one = {
        "state": item.context,
        "questions": system_one_questions(item.question, item.options),
    }
    return [
        "## How questions are presented",
        "",
        "Every model gets the same three pieces; only the packaging differs.",
        "",
        "| Piece | ARC / MMLU / MMLU-Pro | HellaSwag |",
        "|---|---|---|",
        (
            '| **context** (JSON) | `{"question": ...}` plus `subject` (MMLU) or `category` '
            '(MMLU-Pro) | `{"activity": ..., "context": ...}` |'
        ),
        (
            '| **question** | "Which option correctly answers the question?" '
            '| "Which ending most plausibly continues the context?" |'
        ),
        "| **options** | letter → answer text | letter → ending |",
        "",
        "### Jev, Jev via Vercel, and the local server",
        "",
        (
            "A System One request: the context is the `state`, the options are the `criteria` of a "
            "`choice` question. No prompt text is added; Jev applies its own template. The model "
            "returns a choice and a probability for every letter."
        ),
        "",
        "```json",
        json.dumps(system_one, indent=2, ensure_ascii=False),
        "```",
        "",
        "### Claude",
        "",
        (
            "One user message, no system prompt. Structured output limits the reply to one option "
            "letter, so there is no probability. Effort is `low` for Sonnet 5 and Opus 5 "
            "(Haiku 4.5 does not take it); at that effort no model thought before answering."
        ),
        "",
        "```text",
        claude_prompt(item.context, item.question, item.options),
        "```",
        "",
        "Output schema:",
        "",
        "```json",
        json.dumps(claude_answer_schema(item.options), ensure_ascii=False),
        "```",
        "",
        "### Latency experiment",
        "",
        (
            'Same packaging, with context `{"document": "<prose padded to the target size, '
            'containing the needle>"}`, question "What is the archive access code stated in the '
            'document?", and four 4-digit codes as options.'
        ),
        "",
        "### Differences to keep in mind",
        "",
        (
            '- Claude gets one extra line, "Answer with the letter of the single best option." The '
            "`choice` type already tells Jev to pick a key."
        ),
        (
            "- Claude sees the options as a list in its prompt. Jev sees them as a separate "
            "`criteria` field, and how its template presents them is not visible."
        ),
        (
            "- All runs are zero-shot (no worked examples first), unlike the 5-shot setup behind "
            "many published MMLU numbers. Scores are comparable across these models, not with "
            "leaderboards."
        ),
        "",
    ]


def _charts(*charts: tuple[str, str]) -> list[str]:
    """Image links to PNGs from `python -m probes.charts` that have been generated."""
    lines = []
    for name, alt in charts:
        if (RESULTS_DIR / "charts" / name).exists():
            lines += [f"![{alt}](charts/{name})", ""]
    return lines


def _with_charts(section: list[str], *charts: tuple[str, str]) -> list[str]:
    """Insert chart images right after a section's heading line."""
    if not section:
        return section
    return [*section[:2], *_charts(*charts), *section[2:]]


def main() -> None:
    lines = [
        "# Model probe results",
        "",
        *examples_section(),
        *formatting_section(),
        *_with_charts(
            accuracy_table(),
            ("accuracy_vs_latency.png", "Accuracy vs. latency"),
            ("accuracy_by_benchmark.png", "Accuracy by benchmark"),
        ),
        *_with_charts(latency_table(), ("latency_vs_input.png", "Latency vs. input size")),
    ]
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "report.md").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

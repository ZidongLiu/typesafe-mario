"""Experiment 2: response latency as the input grows from ~100 to ~100k tokens.

    python -m probes.latency --models jev local claude:claude-haiku-4-5 --reps 5

Each request is a needle-in-a-haystack question: English prose padded to the target size with a
four-option question about one fact hidden at a random depth. So every model must read the whole
input, and each latency point also says whether the answer was still right at that size.

Sizes are approximate tokens (4 characters per token); each row also records the token count the
backend itself reported. Once a model rejects a size as too long, larger sizes are skipped for it
and `--find-limit` bisects its actual maximum. Results go to probes/results/latency/<model>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from .backends import Answer, Backend, make_backend
from .datasets import filler_corpus

RESULTS_DIR = Path(__file__).parent / "results" / "latency"
CHARS_PER_TOKEN = 4
DEFAULT_SIZES = [100, 300, 1000, 3000, 10000, 30000, 100000]
QUESTION = "What is the archive access code stated in the document?"
TOO_LONG_MARKERS = ("max_tokens_exceeded", "too long", "prompt is too long", "context length")


def build_probe(
    corpus: str, size: int, rng: random.Random
) -> tuple[dict, dict[str, str], str, float]:
    codes = rng.sample(range(1000, 10000), 4)
    letters = "ABCD"
    needle = f" The archive access code is {codes[0]}. "
    chars = max(0, size * CHARS_PER_TOKEN - len(needle))
    # A different slice of the corpus every request, so no backend can reuse a cached prefix.
    start = rng.randrange(0, len(corpus) - chars - 1)
    text = corpus[start : start + chars]
    depth = rng.random()
    cut = text.rfind(" ", 0, int(len(text) * depth)) + 1 if text else 0
    document = text[:cut] + needle + text[cut:]
    order = rng.sample(range(4), 4)
    options = {letters[i]: str(codes[j]) for i, j in enumerate(order)}
    answer = letters[order.index(0)]
    return {"document": document}, options, answer, depth


def _too_long(result: Answer) -> bool:
    return result.error is not None and any(m in result.error.lower() for m in TOO_LONG_MARKERS)


def _record(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _probe(
    backend: Backend, corpus: str, size: int, rng: random.Random
) -> tuple[Answer, str, float]:
    context, options, answer, depth = build_probe(corpus, size, rng)
    return backend.ask(context, QUESTION, options), answer, depth


def find_limit(backend: Backend, corpus: str, ok: int, too_long: int, rng: random.Random) -> int:
    """Largest accepted size between `ok` and `too_long`, to within ~5%."""
    while too_long - ok > max(100, ok // 20):
        mid = (ok + too_long) // 2
        result, _, _ = _probe(backend, corpus, mid, rng)
        if _too_long(result):
            too_long = mid
        elif result.error:
            print(f"  limit search stopped on an unrelated error: {result.error[:200]}")
            break
        else:
            ok = mid
        print(f"  limit search: {ok} accepted, {too_long} rejected")
    return ok


def run(model: str, sizes: list[int], reps: int, seed: int, corpus: str, limit: bool) -> None:
    path = RESULTS_DIR / f"{model.replace(':', '_').replace('/', '_').replace('@', '_')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    backend = make_backend(model)
    try:
        warm, _, _ = _probe(
            backend, corpus, 100, rng
        )  # connection setup / cold start, not recorded
        if warm.error:
            print(f"{model}: warm-up failed: {warm.error[:300]}")
        rejected: int | None = None
        for rep in range(reps):
            for size in rng.sample(sizes, len(sizes)):  # shuffled, so drift doesn't track size
                if rejected is not None and size >= rejected:
                    continue
                result, answer, depth = _probe(backend, corpus, size, rng)
                if _too_long(result):
                    rejected = size if rejected is None else min(rejected, size)
                    print(f"{model}: {size} tokens rejected as too long; skipping larger sizes")
                _record(
                    path,
                    {
                        "model": model,
                        "size": size,
                        "rep": rep,
                        "needle_depth": round(depth, 3),
                        "answer": answer,
                        "correct": result.choice == answer,
                        "too_long": _too_long(result),
                        **asdict(result),
                    },
                )
                status = (
                    result.error[:120] if result.error else f"correct={result.choice == answer}"
                )
                print(
                    f"{model}  size={size:>6}  rep={rep}  {result.latency_ms:8.0f} ms"
                    f"  in_tokens={result.input_tokens}  {status}"
                )
        if limit and rejected is not None:
            accepted = max((s for s in sizes if s < rejected), default=100)
            maximum = find_limit(backend, corpus, accepted, rejected, rng)
            print(f"{model}: largest accepted input is about {maximum} tokens")
            _record(path, {"model": model, "limit_tokens": maximum})
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True, help="see probes/backends.py")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--find-limit", action="store_true", help="bisect each model's max input")
    args = parser.parse_args()
    corpus = filler_corpus()
    total = sum(args.sizes) * args.reps
    print(f"Up to ~{total:,} input tokens per model ({args.reps} reps of {args.sizes})")
    for model in args.models:
        run(model, args.sizes, args.reps, args.seed, corpus, args.find_limit)


if __name__ == "__main__":
    main()

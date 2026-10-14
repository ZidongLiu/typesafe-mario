"""Experiment 1: multiple-choice accuracy of Jev, a local model, and Claude on the same items.

    python -m probes.accuracy --models jev local claude:claude-haiku-4-5 --datasets arc mmlu --n 200

Results append to probes/results/accuracy/<dataset>__<model>.jsonl, one line per item, so an
interrupted run resumes where it stopped; `python -m probes.report` summarizes them.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from . import datasets
from .backends import make_backend

RESULTS_DIR = Path(__file__).parent / "results" / "accuracy"


def result_path(dataset: str, model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_").replace("@", "_")
    return RESULTS_DIR / f"{dataset}__{safe}.jsonl"


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["id"] for row in rows if row["error"] is None}


def run(model: str, dataset: str, items: list[datasets.Item], workers: int) -> None:
    path = result_path(dataset, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(path)
    todo = [item for item in items if item.id not in done]
    if not todo:
        print(f"{model} / {dataset}: all {len(items)} items already answered")
        return
    backend = make_backend(model)
    lock = threading.Lock()
    counts = {"answered": 0, "correct": 0, "errors": 0}

    def answer(item: datasets.Item) -> None:
        result = backend.ask(item.context, item.question, item.options)
        row = {
            "id": item.id,
            "model": model,
            "group": item.group,
            "answer": item.answer,
            "correct": result.choice == item.answer,
            "n_options": len(item.options),
            **asdict(result),
        }
        with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if result.error:
                counts["errors"] += 1
                if counts["errors"] <= 3:
                    print(f"  error on {item.id}: {result.error[:200]}")
            else:
                counts["answered"] += 1
                counts["correct"] += row["correct"]
            finished = counts["answered"] + counts["errors"]
            if finished % 25 == 0 or finished == len(todo):
                accuracy = counts["correct"] / max(1, counts["answered"])
                print(
                    f"  {model} / {dataset}: {finished}/{len(todo)}"
                    f"  accuracy so far {accuracy:.1%}  errors {counts['errors']}"
                )

    print(f"{model} / {dataset}: {len(todo)} items to answer ({len(done)} done before)")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(answer, todo))
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True, help="see probes/backends.py")
    parser.add_argument(
        "--datasets", nargs="+", default=["arc", "mmlu"], choices=list(datasets.SOURCES)
    )
    parser.add_argument("--n", type=int, default=200, help="items per dataset (seeded sample)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel requests per model; use 1 for a local server you don't want to queue",
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        items = datasets.load(dataset, args.n, args.seed)
        for model in args.models:
            run(model, dataset, items, args.workers)


if __name__ == "__main__":
    main()

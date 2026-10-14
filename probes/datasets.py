"""Multiple-choice benchmarks, downloaded once from Hugging Face and cached under probes/.cache."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).parent / ".cache"
HF = "https://huggingface.co/api/datasets"
LETTERS = "ABCDEFGHIJ"

SOURCES = {
    # name: (parquet URL, description)
    "arc": (f"{HF}/allenai/ai2_arc/parquet/ARC-Challenge/test/0.parquet", "grade-school science"),
    "mmlu": (f"{HF}/cais/mmlu/parquet/all/test/0.parquet", "57 academic subjects, 4 options"),
    "hellaswag": (
        f"{HF}/Rowan/hellaswag/parquet/default/validation/0.parquet",
        "commonsense sentence completion",
    ),
    "mmlu-pro": (
        f"{HF}/TIGER-Lab/MMLU-Pro/parquet/default/test/0.parquet",
        "harder MMLU, up to 10 options",
    ),
}


@dataclass(frozen=True)
class Item:
    id: str
    context: dict[str, Any]
    question: str
    options: dict[str, str]
    answer: str
    group: str  # subject / category, for per-group breakdowns


def _rows(name: str) -> list[dict[str, Any]]:
    import httpx2
    import pyarrow.parquet as pq

    path = CACHE_DIR / f"{name}.parquet"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        url = SOURCES[name][0]
        print(f"Downloading {name} from {url}")
        response = httpx2.get(url, follow_redirects=True, timeout=300)
        response.raise_for_status()
        path.write_bytes(response.content)
    return pq.read_table(path).to_pylist()


def _lettered(texts: list[str]) -> dict[str, str]:
    return {LETTERS[i]: str(text).strip() for i, text in enumerate(texts)}


def _convert(name: str, index: int, row: dict[str, Any]) -> Item:
    if name == "mmlu":
        subject = row["subject"].replace("_", " ")
        return Item(
            id=f"mmlu-{index}",
            context={"subject": subject, "question": row["question"]},
            question="Which option correctly answers the question?",
            options=_lettered(row["choices"]),
            answer=LETTERS[int(row["answer"])],
            group=subject,
        )
    if name == "arc":
        labels = list(row["choices"]["label"])
        return Item(
            id=row["id"],
            context={"question": row["question"]},
            question="Which option correctly answers the question?",
            options=_lettered(row["choices"]["text"]),
            answer=LETTERS[labels.index(row["answerKey"])],
            group="science",
        )
    if name == "hellaswag":
        return Item(
            id=f"hellaswag-{index}",  # the `ind` column is not unique
            context={"activity": row["activity_label"], "context": row["ctx"]},
            question="Which ending most plausibly continues the context?",
            options=_lettered(row["endings"]),
            answer=LETTERS[int(row["label"])],
            group=row["activity_label"],
        )
    if name == "mmlu-pro":
        options = [text for text in row["options"] if str(text).strip() != "N/A"]
        return Item(
            id=f"mmlu-pro-{row['question_id']}",
            context={"category": row["category"], "question": row["question"]},
            question="Which option correctly answers the question?",
            options=_lettered(options),
            answer=row["answer"],
            group=row["category"],
        )
    raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(SOURCES)}")


def load(name: str, n: int | None = None, seed: int = 0) -> list[Item]:
    """A seeded sample of `n` items (all items if n is None); the same seed gives the same set."""
    if name not in SOURCES:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(SOURCES)}")
    items = [_convert(name, index, row) for index, row in enumerate(_rows(name))]
    if n is not None and n < len(items):
        items = random.Random(seed).sample(items, n)
    return items


def filler_corpus() -> str:
    """Plain English prose for padding long-input probes: HellaSwag contexts, in a fixed order."""
    rows = _rows("hellaswag")
    return " ".join(f"{row['ctx']} {row['endings'][int(row['label'])]}" for row in rows)

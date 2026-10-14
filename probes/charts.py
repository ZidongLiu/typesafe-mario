"""Charts comparing accuracy and latency, written to probes/results/charts/*.png.

python -m probes.charts
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from .report import RESULTS_DIR, _read, _wilson

CHARTS_DIR = RESULTS_DIR / "charts"
BENCHMARKS = {"arc": "ARC", "hellaswag": "HellaSwag", "mmlu": "MMLU", "mmlu-pro": "MMLU-Pro"}
# model spec -> (label, color); System One models warm, Claude cool.
MODELS = {
    "jev": ("Jev", "#e8590c"),
    "vercel": ("Jev via Vercel", "#ffa94d"),
    "local": ("Local", "#2f9e44"),
    "claude:claude-haiku-4-5": ("Haiku 4.5", "#74c0fc"),
    "claude:claude-sonnet-5": ("Sonnet 5", "#1c7ed6"),
    "claude:claude-opus-5": ("Opus 5", "#1b2a6b"),
}


def _safe(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace("@", "_")


def _accuracy_rows(dataset: str, model: str) -> list[dict]:
    path = RESULTS_DIR / "accuracy" / f"{dataset}__{_safe(model)}.jsonl"
    if not path.exists():
        return []
    latest: dict[str, dict] = {}
    for row in _read(path):
        if row["id"] not in latest or latest[row["id"]]["error"] is not None:
            latest[row["id"]] = row
    return [row for row in latest.values() if row["error"] is None]


def _scatter_panel(ax, points: dict[str, tuple[float, float, float, float]], title: str) -> None:
    for model, (latency, accuracy, low, high) in points.items():
        label, color = MODELS[model]
        ax.errorbar(
            latency,
            accuracy * 100,
            yerr=[[(accuracy - low) * 100], [(high - accuracy) * 100]],
            fmt="o",
            color=color,
            markersize=11,
            capsize=4,
            elinewidth=1.2,
            markeredgecolor="white",
            zorder=3,
        )
        ax.annotate(
            label,
            (latency, accuracy * 100),
            xytext=(9, -4 if "Vercel" not in label else -14),
            textcoords="offset points",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xlim(30, 3000)
    ax.set_xticks([50, 100, 200, 500, 1000, 2000])
    ax.set_xticklabels(["50", "100", "200", "500", "1,000", "2,000"])
    ax.set_xlabel("Median latency per question (ms, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, which="major", alpha=0.3)
    ax.annotate(
        "better ↖",
        xy=(0.04, 0.94),
        xycoords="axes fraction",
        fontsize=10,
        color="#868e96",
        style="italic",
    )


def accuracy_vs_latency(plt) -> Path:
    overall: dict[str, tuple[float, float, float, float]] = {}
    hardest: dict[str, tuple[float, float, float, float]] = {}
    for model in MODELS:
        rows = [row for dataset in BENCHMARKS for row in _accuracy_rows(dataset, model)]
        if not rows:
            continue
        correct = sum(row["correct"] for row in rows)
        low, high = _wilson(correct, len(rows))
        latency = statistics.median(row["latency_ms"] for row in rows)
        overall[model] = (latency, correct / len(rows), low, high)
        pro = _accuracy_rows("mmlu-pro", model)
        correct = sum(row["correct"] for row in pro)
        low, high = _wilson(correct, len(pro))
        latency = statistics.median(row["latency_ms"] for row in pro)
        hardest[model] = (latency, correct / len(pro), low, high)

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.8))
    _scatter_panel(left, overall, "All four benchmarks (800 questions per model)")
    left.set_ylim(80, 100)
    _scatter_panel(right, hardest, "MMLU-Pro, the hardest (200 questions, ~11% chance)")
    right.set_ylim(50, 100)
    fig.suptitle(
        "Accuracy vs. latency: Jev is near Sonnet/Opus accuracy at ~1/10 the latency",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Error bars: 95% confidence interval. Claude answered without thinking (effort low). "
        "Latency from the accuracy runs (Claude 4 requests in parallel, others 1).",
        ha="center",
        fontsize=9,
        color="#495057",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save(fig, "accuracy_vs_latency.png")


def accuracy_by_benchmark(plt) -> Path:
    models = [m for m in MODELS if _accuracy_rows("arc", m)]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    width = 0.8 / len(models)
    for index, model in enumerate(models):
        label, color = MODELS[model]
        values = []
        for dataset in BENCHMARKS:
            rows = _accuracy_rows(dataset, model)
            values.append(100 * sum(row["correct"] for row in rows) / max(1, len(rows)))
        positions = [i + (index - (len(models) - 1) / 2) * width for i in range(len(BENCHMARKS))]
        bars = ax.bar(positions, values, width, label=label, color=color)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.6,
                f"{value:.1f}",
                ha="center",
                fontsize=7,
                color="#343a40",
            )
    ax.set_xticks(range(len(BENCHMARKS)))
    ax.set_xticklabels(BENCHMARKS.values(), fontsize=11)
    ax.set_ylim(50, 102)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by benchmark (200 questions each)", fontsize=13, fontweight="bold")
    ax.legend(
        ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=False
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "accuracy_by_benchmark.png")


def latency_vs_input(plt) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    failure_row = 40.0  # failed requests have no latency; mark them along the bottom instead
    for model, (label, color) in MODELS.items():
        path = RESULTS_DIR / "latency" / f"{_safe(model)}.jsonl"
        if not path.exists():
            continue
        by_size: dict[int, list[dict]] = defaultdict(list)
        for row in _read(path):
            if "size" in row:
                by_size[row["size"]].append(row)
        xs, ys, failed = [], [], []
        for size in sorted(by_size):
            ok = [row for row in by_size[size] if row["error"] is None]
            if ok:
                # Plot against the token count the backend itself reported.
                tokens = [row["input_tokens"] for row in ok if row["input_tokens"]]
                xs.append(statistics.median(tokens) if tokens else size)
                ys.append(statistics.median(row["latency_ms"] for row in ok))
            else:
                failed.append(size)
        ax.plot(xs, ys, "o-", color=color, label=label, linewidth=2, markersize=6)
        if failed:
            reason = "rejected (too long)" if model in ("jev", "vercel") else "connection dropped"
            ax.scatter(failed, [failure_row] * len(failed), marker="x", s=60, color=color, zorder=4)
            ax.annotate(
                f"{label}: {reason}",
                (failed[-1], failure_row),
                xytext=(8, -3),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
            failure_row *= 1.13
    ax.axvline(32768, color="#e8590c", linestyle=":", alpha=0.6)
    ax.annotate(
        "Jev limit ≈ 32k",
        (32768, 80),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9,
        color="#e8590c",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([200, 1000, 3000, 10000, 30000, 100000])
    ax.set_xticklabels(["200", "1k", "3k", "10k", "30k", "100k"])
    ax.set_ylim(34, 2600)
    ax.set_xlim(150, 250000)
    ax.set_yticks([50, 100, 200, 500, 1000, 2000])
    ax.set_yticklabels(["50", "100", "200", "500", "1,000", "2,000"])
    ax.set_xlabel(
        "Input tokens (as reported by each backend, log scale).  ✕ = failed request (no latency)"
    )
    ax.set_ylabel("Median latency (ms, log scale)")
    ax.set_title(
        "Latency vs. input size (needle-in-a-haystack, 5 runs per point)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    return _save(fig, "latency_vs_input.png")


def _save(fig, name: str) -> Path:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=140)
    return path


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for chart in (accuracy_vs_latency, accuracy_by_benchmark, latency_vs_input):
        print(chart(plt))


if __name__ == "__main__":
    main()

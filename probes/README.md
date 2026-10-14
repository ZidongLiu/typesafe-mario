# Model probes

Two experiments that put Jev, a local Jev-compatible server, and Claude through the same
inputs.

1. **Accuracy** (`probes.accuracy`): standard multiple-choice benchmarks. Every model gets the
   same JSON context, question, and lettered options. Jev and local servers answer with a
   System One `choice`. Claude answers with one letter through structured output.
2. **Latency vs. input size** (`probes.latency`): the input grows from ~100 to ~100k tokens.
   Each request is a needle-in-a-haystack question (real English prose with one fact hidden at a
   random depth), so the model has to read the whole input, and every timing also records
   whether the answer was right.

```powershell
uv sync --extra probe --extra claude   # pyarrow for datasets, anthropic for Claude

.venv\Scripts\python -m probes.accuracy --models jev local claude:claude-haiku-4-5 `
    --datasets arc mmlu hellaswag mmlu-pro --n 200
.venv\Scripts\python -m probes.latency --models jev local claude:claude-haiku-4-5 --reps 5 --find-limit
.venv\Scripts\python -m probes.charts  # PNG charts in probes/results/charts/
.venv\Scripts\python -m probes.report  # tables, also written to probes/results/report.md
```

## Model specs

| Spec | Model |
|---|---|
| `jev`, `jev:jev-preview` | TypeSafe's hosted Jev (`TYPESAFE_API_KEY`) |
| `vercel` | Jev through Vercel AI Gateway (`AI_GATEWAY_API_KEY`) |
| `local`, `local:http://host:port` | A local `/v1/systemone` server (`LOCAL_JEV_BASE_URL`, `SHIM_TOKEN`); default `http://192.168.1.161:3000` |
| `claude:<model>[@effort]` | A Claude model (`ANTHROPIC_API_KEY`); effort defaults to `low` |

## Datasets

Seeded samples (`--n`, `--seed`) downloaded once from Hugging Face into `probes/.cache/`:

| Name | Content | Options |
|---|---|---|
| `arc` | ARC-Challenge, grade-school science | 4 |
| `mmlu` | MMLU, 57 academic subjects | 4 |
| `hellaswag` | HellaSwag, commonsense sentence completion | 4 |
| `mmlu-pro` | MMLU-Pro, harder reasoning-heavy MMLU | up to 10 |

## Notes

- Accuracy runs resume: each answered item is appended to
  `probes/results/accuracy/<dataset>__<model>.jsonl`, and items that errored are retried on the
  next run.
- These public benchmarks may be in any model's training data, so treat scores as relative.
- Latency in the accuracy table is measured with `--workers` parallel requests. Use the latency
  experiment for clean timings: it sends one request at a time, makes one uncounted warm-up
  call, shuffles the size order on each rep, and uses a different slice of prose for every
  request so no prefix cache can help.
- Claude's latency includes writing its answer (plus thinking at Opus/Sonnet effort levels).
  The *first token* column is its time to start answering, which is mostly reading the input.

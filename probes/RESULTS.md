# Probing Jev: accuracy and latency vs. Claude and a local model

*Run on 2026-10-13. Code: [`probes/`](README.md). Raw per-request data: `probes/results/`.
Charts: `python -m probes.charts`.*

TypeSafe's Jev is a "System One" model: it returns a probability for each option of a typed
question in one pass, instead of writing an answer token by token. These two experiments
measure how smart and how fast that is, next to Claude and a self-hosted Jev-compatible model.

## Example questions

A few of the actual items, with each model's pick. ✅ marks the correct option; the percentage is the System One model's probability for its pick. The shortest item of each kind is shown, so these are easier than a typical question.

### ARC-Challenge

**Every model right**

> What is the pH level for pure water?

- A. 1.6
- B. 2.3
- **C. 7.0** ✅
- D. 11.2

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| C ✓ (100%) | C ✓ (100%) | C ✓ | C ✓ | C ✓ |

**Models split, Jev right**

> An exothermic reaction could best be demonstrated with a

- **A. car engine.** ✅
- B. refrigerator.
- C. frying pan.
- D. glass bottle.

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| A ✓ (99%) | A ✓ (99%) | C ✗ | A ✓ | A ✓ |

**Jev wrong**

> Which alternative energy resource is used most like coal?

- A. geothermal energy
- B. nuclear energy
- C. hydropower
- **D. biofuel energy** ✅

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| B ✗ (96%) | D ✓ (69%) | B ✗ | B ✗ | D ✓ |

### HellaSwag

**Every model right**

> *Running a marathon:* People are running on the street. a man …

- A. is wearing stilts.
- **B. takes a drink out of a water bottle.** ✅
- C. jumps into a pile of fire.
- D. takes a high jump in a bar competition.

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| B ✓ (100%) | B ✓ (97%) | B ✓ | B ✓ | B ✓ |

**Models split, Jev right**

> *Carving jack-o-lanterns:* We see a black opening logo. A man etches a pumpkin with a knife like tool. the man …

- A. peels the pumpkins one by one.
- **B. then carves a logo into the pumpkin.** ✅
- C. cuts it with a tool, then cleans the backing up.
- D. wraps the pumpkin in wrapping paper.

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| B ✓ (75%) | B ✓ (61%) | C ✗ | C ✗ | B ✓ |

**Jev wrong**

> *Disc dog:* A woman is seen standing in a field holding frisbees with a dog. the woman …

- A. rides the dog all through the field while the dog talks to the camera.
- **B. begins throwing around the frisbee performing tricks with the dog.** ✅
- C. throws the frisbees far and the dog catches them and runs around the woman.
- D. then tricks the dog around while he runs around in circles.

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| C ✗ (60%) | C ✗ (56%) | C ✗ | C ✗ | B ✓ |

### MMLU

**Every model right**

> *high school mathematics:* How many odd perfect squares are between 5 and 211?

- A. 2
- B. 25
- **C. 6** ✅
- D. 75

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| C ✓ (96%) | C ✓ (71%) | C ✓ | C ✓ | C ✓ |

**Models split, Jev right**

> *miscellaneous:* What is the traditional 20th wedding anniversary gift?

- A. paper
- B. leather
- **C. china** ✅
- D. silver

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| C ✓ (100%) | C ✓ (58%) | B ✗ | C ✓ | C ✓ |

**Jev wrong**

> *elementary mathematics:* Find the number that makes the statement 12 over 18 = ? over 9 true.

- **A. 4** ✅
- B. 6
- C. 8
- D. 12

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| B ✗ (99%) | B ✗ (100%) | B ✗ | B ✗ | B ✗ |

### MMLU-Pro

**Every model right**

> *math:* Find the product of 0.07 • 5.8.

- **A. 0.406** ✅
- B. 0.0406
- C. 4.06
- D. 40.6

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| A ✓ (99%) | A ✓ (99%) | A ✓ | A ✓ | A ✓ |

**Models split, Jev right**

> *math:* Compute 22 / 2 + 9.

- A. 24
- B. 23
- C. 21
- D. 22
- E. 10
- F. 25
- G. 2
- **H. 20** ✅
- I. 19
- J. 11

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| H ✓ (40%) | H ✓ (99%) | D ✗ | H ✓ | H ✓ |

**Jev wrong**

> *philosophy:* In his final work, Laws, Plato shifted from cosmology to which of the following issues?

- A. Epistemology
- **B. Morality** ✅
- C. Religion
- D. Aesthetics

| Jev | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|
| C ✗ (52%) | B ✓ (91%) | C ✗ | C ✗ | C ✗ |

## Summary

- **Jev scores about the same as Claude Sonnet 5 on multiple choice, at about 1/9 of the latency.** It averaged 92.2% across
  four benchmarks in ~110 ms per question. Sonnet 5 averaged 91.6% in ~950 ms, and Opus 5 averaged 94.4% in
  ~1.1 s (all without thinking).
- **Jev's confidence is trustworthy.** Its mean confidence (92.8%) almost equals its accuracy
  (92.2%). When it is ≥ 99% sure, it is right 98.4% of the time.
- **Jev barely slows down with input length.** From ~450 to ~27,600 tokens its latency goes from 116 ms
  to 310 ms. It rejects anything above **~32k tokens**.
- **The local model is fastest on short inputs (51 ms) but scales ~15× worse per token.** It loses to
  Jev beyond ~1k tokens and crashes at ~30k.
- **Claude's latency is a flat ~0.7–2 s, set by fixed time-to-first-token, not by input size.**
  It handled 128k-token inputs without error.

![Accuracy vs. latency](results/charts/accuracy_vs_latency.png)

## What was tested

### Models

| Label | What it is | How it answers |
|---|---|---|
| **Jev** | TypeSafe hosted `jev-latest` (served as `jev-1.13.0`), `api.typesafe.ai` | System One `choice` with probabilities |
| **Jev via Vercel** | Same model through Vercel AI Gateway (`typesafe-ai/jev`) | Same |
| **Local** | Self-hosted OpenJev shim (DiffusionGemma-based) at `192.168.1.161:3000` | Same API, its own probabilities |
| **Haiku 4.5** | `claude-haiku-4-5` | One letter through structured output |
| **Sonnet 5** | `claude-sonnet-5`, effort `low` | Same |
| **Opus 5** | `claude-opus-5`, effort `low` | Same |

Each Claude model wrote 9–11 output tokens per answer. At `low` effort none of them thought
before answering, so this compares instant answers on both sides.

### Experiment counts

| | Experiment 1: accuracy | Experiment 2: latency vs. input size |
|---|---|---|
| Model configurations | 6 | 6 |
| Test conditions | 4 benchmarks | 7 input sizes (~100 to ~100k tokens) |
| Samples per condition | 200 questions (seed 0, same set for every model) | 5 repetitions |
| Scored requests | **4,800** (4 × 200 × 6) | **206** of 210 planned (Jev skipped its last 4 over-limit requests) |
| Extra requests | 12 re-asks after a dataset ID fix | 6 warm-ups (not scored) + 6 limit-search probes |
| Failed requests | 0 | 16, all at sizes over a model's limit (see below) |
| Input tokens sent (as billed by each backend) | ~1.98M | ~3.03M |

**In total: ~5,030 API requests and ~5M input tokens.**

### Benchmarks (Experiment 1)

| Benchmark | Source split | Pool size | Sampled | Variety in the sample | Options | Chance |
|---|---|---|---|---|---|---|
| ARC-Challenge | test | 1,172 | 200 | grade-school science | 4 | 25% |
| MMLU | test | 14,042 | 200 | 49 of 57 subjects | 4 | 25% |
| HellaSwag | validation | 10,042 | 200 | 59 activity types | 4 | 25% |
| MMLU-Pro | test | 12,032 | 200 | 14 categories | 9.5 on average (up to 10) | ~11% |

## Experiment 1: accuracy

| Model | ARC | HellaSwag | MMLU | MMLU-Pro | **Average** | Median latency |
|---|---|---|---|---|---|---|
| **Jev** | 98.0% | 96.0% | 91.5% | 83.5% | **92.2%** | **~110 ms** |
| Jev via Vercel | 98.0% | 96.5% | 91.0% | 83.0% | 92.1% | ~190 ms |
| Local | 98.5% | 92.5% | 88.5% | 70.0% | 87.4% | **~50 ms** |
| Haiku 4.5 | 95.0% | 96.0% | 87.5% | 58.5% | 84.3% | ~720 ms |
| Sonnet 5 | 97.0% | 95.5% | 92.5% | 81.5% | 91.6% | ~950 ms |
| Opus 5 | 98.5% | 96.5% | 92.5% | **90.0%** | **94.4%** | ~1,100 ms |

With 200 questions per benchmark, the 95% confidence interval is about ±2–4 points near 95% accuracy
and about ±5–6 points near 80%. ARC, HellaSwag and MMLU don't separate the strong models. MMLU-Pro
does.

![Accuracy by benchmark](results/charts/accuracy_by_benchmark.png)

### MMLU-Pro by category (questions correct)

| Category | n | Jev | Opus 5 | Haiku 4.5 | Local |
|---|---|---|---|---|---|
| physics | 28 | 22 | 26 | 13 | 17 |
| law | 24 | 19 | 21 | 14 | 17 |
| engineering | 21 | 17 | 19 | 9 | 12 |
| chemistry | 20 | 14 | 18 | 8 | 11 |
| math | 20 | **18** | 20 | 10 | 15 |
| other | 16 | 12 | 10 | 9 | 11 |
| psychology | 13 | 9 | 10 | 8 | 7 |
| biology, health, business, economics, CS, history, philosophy | 58 | 56 | 56 | 46 | 50 |

Opus 5's lead over Jev comes mostly from physics, chemistry and engineering, where questions need
multi-step calculation.

### How often the models agree

| Benchmark | Jev = Opus 5 | Only Opus right | Only Jev right | Both wrong | Jev = Jev via Vercel | Jev = Local |
|---|---|---|---|---|---|---|
| ARC | 197/200 | 2 | 1 | 2 | 200/200 | 197/200 |
| HellaSwag | 188/200 | 6 | 5 | 2 | 199/200 | 187/200 |
| MMLU | 188/200 | 7 | 5 | 10 | 199/200 | 183/200 |
| MMLU-Pro | 168/200 | 21 | 8 | 12 | 198/200 | 141/200 |

Jev direct and Jev through Vercel agree on 796 of 800 questions, so the gateway serves the same
model and the leftover differences are nondeterminism.

### Calibration (System One models only)

Jev and the local model return a real probability for every option. Claude's answers here carry
no probability. Accuracy by Jev's confidence in its chosen answer, over all 800 questions:

| Confidence in chosen option | Jev: questions | Jev: accuracy | Local: questions | Local: accuracy |
|---|---|---|---|---|
| < 50% | 32 | 40.6% | 72 | 33.3% |
| 50–80% | 68 | 72.1% | 100 | 71.0% |
| 80–95% | 89 | 85.4% | 136 | 91.2% |
| 95–99% | 120 | 97.5% | 169 | 97.0% |
| ≥ 99% | 491 | 98.4% | 323 | 97.8% |
| **Mean** | | conf. 92.8% vs. acc. 92.2% | | conf. 87.3% vs. acc. 87.4% |

Both models are close to calibrated. Jev is more decisive: it is ≥ 99% sure on 61% of questions,
against 40% for the local model. So its confidence is a usable signal, for example to send
low-confidence answers to a slower model.

## Experiment 2: latency vs. input size

Each request is a needle-in-a-haystack question: real English prose (HellaSwag passages) padded
to the target size, with one fact (*"The archive access code is 1234."*) at a random depth and four
candidate codes as options. Each request used a different slice of prose, so no prefix cache could
help. Requests went one at a time, in shuffled size order, after one unscored warm-up per model.

**Median latency in ms (5 runs each).** Token counts are what each backend reported for the
request.

| Target size | Jev | Jev via Vercel | Local | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|---|---|---|
| ~100 | 116 (452 tok) | 207 | **51** (188 tok) | 787 (330 tok) | 1,266 | 1,245 |
| ~300 | 122 | 225 | **74** | 746 | 1,208 | 1,241 |
| ~1k | **129** | 191 | 146 | 757 | 855 | 1,278 |
| ~3k | **130** | 218 | 304 | 748 | 902 | 1,417 |
| ~10k | **184** (9.4k tok) | 282 | 959 (9.0k tok) | 768 | 1,265 | 1,187 |
| ~30k | **310** (27.6k tok) | 381 | crashed | 823 | 1,271 | 1,813 |
| ~100k | rejected | rejected | crashed | 1,169 (96k tok) | 1,285 (128k tok) | 1,951 (128k tok) |

**Needle found:** 190 of 190 answered requests. Every model found the fact at every size it
accepted.

![Latency vs. input size](results/charts/latency_vs_input.png)

### What the curves say

| Model | Fixed cost | Cost per 1,000 extra input tokens | Input limit |
|---|---|---|---|
| Jev | ~110 ms | ~7 ms | ~32k tokens (`max_tokens_exceeded` above ~35k target tokens, found by bisection) |
| Jev via Vercel | ~200 ms (gateway adds 70–100 ms) | ~7 ms | Same as Jev |
| Local | ~45 ms | ~100 ms | Drops the connection at ~30k after ~3 s, probably out of memory |
| Haiku 4.5 | ~650 ms before the first token | ~4 ms | ≥ 96k tested |
| Sonnet 5 / Opus 5 | ~0.9–1.2 s before the first token | small and noisy | ≥ 128k tested |

- **Jev** reads input about 14× faster than the local model per token. That fits a well-optimized
  single read-only pass on datacenter hardware.
- **The local model** has the lowest fixed overhead (on the LAN, a small shim), but its cost per token is
  steep. The two curves cross at about 1k tokens.
- **Claude** is the opposite of the local model: almost all of its latency is fixed. On a 100-token
  question, Haiku waits ~650 ms for its first token. Input size adds little, and at 9–11 output
  tokens neither does writing. In the Mario agent, Claude is slower because it writes four
  JSON fields there, and Opus thinks first.

### Failed requests (16, all expected)

| Model | Failures | Cause |
|---|---|---|
| Jev | 1 | ~100k request rejected with `400 max_tokens_exceeded`. The remaining 4 were skipped. |
| Jev via Vercel | 5 | Same rejection, passed through as a generic gateway 400, so the sweep didn't recognize it and sent all 5 |
| Local | 10 | All 5 at ~30k and all 5 at ~100k: the server dropped the connection after ~3–4 s |

## Caveats

- **Contamination.** All four benchmarks are public, and Jev's training data is unknown. 18/20 on
  MMLU-Pro math with no thinking is unusually high, so some of Jev's score may be memorization.
  Fresh, unpublished questions would separate recall from reasoning.
- **No thinking for Claude.** At `low` effort Claude answered directly. With more thinking Opus
  would likely gain on the harder benchmarks, at many times the latency.
- **Sample size.** 200 questions per benchmark gives ±2–6-point confidence intervals. Differences of a
  point or two between strong models are noise.
- **Latency in the accuracy table** was measured with 1 request at a time for Jev and Local, and 4
  at a time for Claude. Experiment 2 is the clean timing: one request at a time, from Windows on
  the same LAN as the local server.
- **Single run, single day.** Latency varies with server load. On another day the local server's Mario
  request latency ranged from ~200 to ~300 ms.

## Reproduce

```powershell
uv sync --extra probe --extra claude
$models = "jev", "vercel", "local", "claude:claude-haiku-4-5", "claude:claude-sonnet-5", "claude:claude-opus-5"
.venv\Scripts\python -m probes.latency  --models $models --reps 5 --find-limit
.venv\Scripts\python -m probes.accuracy --models $models --datasets arc mmlu hellaswag mmlu-pro --n 200
.venv\Scripts\python -m probes.charts
.venv\Scripts\python -m probes.report
```

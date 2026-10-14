# TypeSafe Mario

An experimental controller that lets TypeSafe's Jev model directly choose NES
controller inputs for the original Super Mario Bros.

The model does **not** receive screenshots. The harness translates emulator telemetry
and RAM into compact, object-centric JSON containing Mario's motion, jump trajectory,
upcoming enemies, terrain, measured response delay, recent-control results, and episode
progress. Jev chooses one of the legal controller actions, the emulator advances several
frames, and the loop repeats. The raw local tile grid remains available in debug logs
and the UI, but is not duplicated in the model input.

## Architecture

```text
NES emulator -> telemetry/RAM parser -> structured JSON -> Jev Choice -> controller input
```

The initial action set is intentionally small:

- `noop`
- `right`
- `right_jump`
- `right_run`
- `right_run_jump`
- `jump`
- `left`

## Requirements

- Python 3.13 or newer
- A TypeSafe API key in `TYPESAFE_API_KEY`, or a Vercel AI Gateway key in
  `AI_GATEWAY_API_KEY` (used when `TYPESAFE_API_KEY` is unset; override the model
  with `AI_GATEWAY_MODEL`, default `typesafe-ai/jev`)
- A legal local setup for Super Mario Bros.

This repository contains no Nintendo ROM or other copyrighted game data. You are
responsible for ensuring that your emulator and game files are obtained and used
lawfully.

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[mario,dev]"
$env:TYPESAFE_API_KEY = "your-key"
# or, to reach Jev through Vercel AI Gateway:
$env:AI_GATEWAY_API_KEY = "your-vercel-key"
```

With `uv`, `uv sync --extra mario --extra dev` creates the same environment from `uv.lock`.

Instead of exporting variables, you can put `KEY=VALUE` lines in a `.env` file in the
repository root. The CLI loads it at startup without overriding variables that are already
set.

Inspect the exact JSON and text that will be sent to TypeSafe without launching the
game or calling the API:

```powershell
.venv\Scripts\typesafe-mario state-demo
```

Run World 1-1 with Jev making a decision every eight emulator steps. The default
display is a single recordable window with the live game and model telemetry:

```powershell
.venv\Scripts\typesafe-mario play --env SuperMarioBros-1-1-v0 --frames-per-decision 8
```

The dashboard shows the selected action, full Choice probability distribution,
confidence, Jev latency, jump probability, danger score, reward, and parsed game
state. Press `R` or click **Restart** for a fresh episode; the dashboard remains open
after death or level completion. Press `Esc` or `Q` to quit. Use `--display game` for
only the emulator window (a 3x pygame window paced at 60 fps, since nes-py's own pyglet
viewer crashes on Python 3.14) or `--display none` for a headless benchmark.

Each decision is written to `artifacts/run-<timestamp>.jsonl`. These records include
latency, action probabilities, confidence, canonical model state, raw debug state, and
game outcome, providing the data for a live overlay or rendered social clip.

## Decision backends

`play --backend` picks the service that answers each decision. All backends get the same
state and the same questions, and each logged decision records its `source`.

| Backend | Service | Keys / settings |
|---|---|---|
| `auto` (default) | `typesafe` if `TYPESAFE_API_KEY` is set, else `vercel` | |
| `typesafe` | TypeSafe API directly | `TYPESAFE_API_KEY` |
| `vercel` | Jev through Vercel AI Gateway | `AI_GATEWAY_API_KEY` or `VERCEL_API_KEY`; `--model` defaults to `typesafe-ai/jev` |
| `local` | Your own server speaking Jev's `POST /v1/systemone` API | `--base-url` (default `http://127.0.0.1:8080`, or `LOCAL_JEV_BASE_URL`), `--model` (or `LOCAL_JEV_MODEL`), optional `LOCAL_JEV_API_KEY` (falls back to `SHIM_TOKEN`) |
| `claude` | Anthropic Messages API | `ANTHROPIC_API_KEY`; install with `pip install -e ".[claude]"`; `--model` defaults to `claude-opus-5`, `--claude-effort` defaults to `low` |

```powershell
.venv\Scripts\typesafe-mario play --backend typesafe --display game
.venv\Scripts\typesafe-mario play --backend claude --model claude-haiku-4-5 --display game
.venv\Scripts\typesafe-mario play --backend local --base-url http://192.168.1.161:3000 --display game
```

If a backend errors or times out, that decision falls back to a safe default action and the
run continues.

Measured latency per decision with the full Mario request:

| Backend | Latency |
|---|---|
| `typesafe` (Jev direct) | ~140 ms |
| `vercel` (Jev via gateway) | ~250 ms |
| `local` (OpenJev on a LAN GPU) | ~200 ms |
| `claude` Haiku 4.5 / Sonnet 5 / Opus 5 | ~1.1 s / ~1.8 s / ~2.7 s |

Jev-style servers read answer probabilities from one forward pass instead of generating
text, so their latency stays low. Claude writes its answer as JSON token by token, which
takes seconds per decision.

Claude has no native choice or score distributions, so its confidence, jump probability, and
danger score are prompted estimates, not calibrated probabilities.

## What the parser produces

TypeSafe accepts JSON directly, so there is no need to flatten telemetry into prose.
The model-facing object groups observations by meaning:

- `player`: position, velocity, grounded state, jump phase, and power-up
- `trajectory`: airtime, distance since takeoff, and committed gap crossing
- `hazard`: up to three enemies, projected positions, contact timing, and takeoff deadline
- `terrain`: obstacle/gap geometry (including distance to the gap edge and a
  `gap_takeoff_now` flag), observation reliability, and last grounded preview
- `reaction_timing`: action duration and measured observation-to-action delay
- `recent_control`: chosen action, duration, progress gained, and observed outcome
- `episode`: lives, clock, progress, stalls, death, and level completion

For humans, the fuller debug snapshot can still be rendered as compact text:

```text
Goal: Reach the flag in World 1-1 without dying.
Mario: x=172 y=79, moving right, airborne=False, status=small
Progress: 172 (best 172), time=387, lives=2
Nearby enemies: goomba 42px ahead
Local grid (# solid, . empty, E enemy, M Mario):
...........
...........
...........
..M..E.....
###########
```

The structured object is canonical; the text view is only for debugging and UI.

## TypeSafe judgments

Each request evaluates three independent judgments over the same state:

- a `Choice` selects the next controller macro;
- a `Noul` estimates whether a forward jump is useful now;
- a `Score` measures immediate danger for the live visualization.

Exact timing arithmetic stays in code. For example, the parser combines measured
response age, enemy motion, action cadence, and jump-clearance time into a typed
`jump_must_start_this_decision` fact. Jev interprets those facts and still owns the
controller choice—there is no scripted recovery-action override.

The harness does handle button timing within a chosen action. It releases A for one frame
before a new jump so the press registers, holds A while rising when a tall obstacle or an
enemy ahead needs a full-height jump, and releases A early for a short hop onto a step just
before a gap.

## Development

```powershell
.venv\Scripts\ruff format --check src tests
.venv\Scripts\ruff check src tests
.venv\Scripts\python -m pytest -q
```

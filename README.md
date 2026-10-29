# Agent Run Debugger

**Debug failed agent runs with evidence, not stories.**

Agent Run Debugger is a local-first evidence debugger for failed, stalled or expensive agent runs. Import one explicitly selected trace and get a deterministic timeline, observed failure points, ranked hypotheses that cite their evidence, the gaps you still need to fill, and an inert replay of the recorded run.

**Status:** v1 debugger, offline-first. The end-to-end offline journey is implemented: import, timeline, diagnosis, replay, export and status. Live capture and model narration are separate, opt-in lanes.

## At a glance

| Area | What it gives you |
|---|---|
| Input | Selected JSONL traces from Hermes, Claude Code or the harness-neutral v1 format |
| Timeline | Deterministic ordering from the recorded events |
| Diagnosis | Observed failure points and ranked hypotheses with evidence |
| Gaps | Missing evidence and falsification checks you can run |
| Replay | Inert playback of the recorded timeline; nothing is executed |
| Output | A private Markdown report, with redacted exports |

## Why it exists

Agent failures are usually buried in long traces. Reading the raw log gives you events, but not a clear account of what failed, what is unknown, or which explanation is actually supported.

Agent Run Debugger turns one selected trace into an evidence trail:

- what happened, in order;
- where the run first diverged or failed;
- which hypotheses are supported by the recorded evidence;
- what evidence is missing;
- what to check next;
- what can be replayed safely without touching the live system.

## Design principles

- **Evidence first.** Observed facts, hypotheses and missing evidence are kept separate.
- **Inert replay.** Replay re-reads the recorded timeline. It runs no subprocess, opens no network connection and touches no live database.
- **Local by default.** Import a trace explicitly selected by you or the model through `rundebug_import`. No telemetry collector and no background capture.
- **Host-neutral core.** The domain core does not import Hermes. A thin adapter handles the host integration.
- **Explicit opt-ins.** Live reads, capture, model narration and real re-execution are separate lanes with their own gates.
- **Unknown stays unknown.** Missing logs, unknown billing and unsupported signals are reported as gaps, not guessed or rounded to zero.

## Harness support

| Format | What it takes |
|---|---|
| `hermes` | `hermes sessions export --format jsonl` — the host’s own session envelope |
| `v1` | The harness-neutral trace in [`schemas/trace-v1.json`](schemas/trace-v1.json) |
| `claude` | Claude Code JSONL session transcripts, via a tested adapter |

```sh
hermes sessions export --format jsonl --session-id <id> ./run.jsonl
hermes run-debug import --file ./run.jsonl --format hermes
```

Adding another harness means adding one adapter that maps its records onto `event-v1`. The store, timeline, diagnosis, replay and report layers stay the same.

### Debug agent actions and harness boundaries

`explain` follows failed actions through their later results. It distinguishes a repeated
unchanged request, a changed request and a later reported success of the same operation.
Different operations behind `tool_call` are not treated as a single retry loop. The
structured diagnosis includes `action_chains` and separately reported attempt outcomes.

Use `hermes run-debug timeline --run <id> --event <event-id>` (or the timeline tool's
`event` argument) to inspect the event, its recorded parents/children and same-request
results. Source references point back to the recording; they do not authenticate the
durable business effect. The explain tool accepts `json: true` for machine-readable output.

Cauldron harness receipts can be selected explicitly with `--format cauldron`.
This imports recorded stage and lifecycle evidence, not a live run or a second scorer.
Keep agent operation recovery, harness setup/cleanup and the UAT business score separate.
Existing imports remain readable; re-import selected original exports for richer request
identity. The debugger does not fill missing history with guesses.

## Surfaces

| Surface | Route |
|---|---|
| CLI | `hermes run-debug import\|timeline\|explain\|replay\|export\|status` |
| Slash | `/rundebug <subcommand>` |
| Agent tools | `rundebug_import`, `rundebug_timeline`, `rundebug_explain`, `rundebug_replay`, `rundebug_status` |

`export` is operator-CLI-only. Tool, slash and gateway routes do not expose it, so viewing a private report never becomes an implicit sharing permission.

## Privacy

- Minimal metadata by default; no raw prompt or result retention.
- Redaction runs before the store, before the journal, before error logs and again before export.
- Raw input digests stay local. Exports carry redacted content digests and stable opaque references.
- The store lives under `plugin-data/agent-run-debugger/` for the active profile.
- Deletion only targets owned record IDs and never follows a source-path reference.

## Quick start

```sh
hermes run-debug import  --file ./selected-run.jsonl --format v1
hermes run-debug timeline --run <run-id>
hermes run-debug explain  --run <run-id>
hermes run-debug replay   --run <run-id>
hermes run-debug export   --run <run-id> --out ~/private/rundebug/report.md
hermes run-debug status
```

## Roadmap

- Native capture lane with an explicit owner gate.
- Selected-session live reads.
- Optional model narration, clearly labelled as hypothesis.
- Sandboxed experiments and comparative replays.

These are separate product lanes. v1 is useful without them: it makes the recorded evidence legible and keeps the unknown visible.

## Development

```sh
uv sync --locked --extra dev
uv run python -m pytest tests/unit tests/adversarial --strict-markers -ra
uv run python -m pytest tests/integration --strict-markers -ra
uv run ruff check .
```

Nothing here installs into a live Hermes profile. The host test loads the plugin from a checkout into a disposable `HERMES_HOME`.

Licensed under Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Repository: https://github.com/Sahil-SS9/agent-run-debugger

---
name: rundebug
description: Use when a run failed, stalled or cost too much. Import a selected trace, read its timeline, hypotheses and missing evidence, then replay it inertly.
version: 0.1.0
---

# Rundebug — debugging an agent run from evidence

## When this applies

A run failed, stalled, looped, or cost more than expected, and the question is
*what the recording actually shows* — not what a summary sounds like.

## Workflow

1. **Select a trace deliberately.** One file (or a set of files) you choose. The
   plugin never scans `~/.claude`, a session database or your home directory.
   `hermes run-debug import --file <path> [--format v1|claude|hermes]`.
2. **Read the timeline**, not just the error: `/rundebug timeline <run-id>`. Check
   the *gaps* section first — a missing parent or a truncated tail limits every
   conclusion below it.
3. **Read the diagnosis.** `/rundebug explain <run-id>`. Take each hypothesis with
   its alternatives and its falsifier; treat `confidence` as a category derived from
   the recorded evidence, not a probability.
   Start with action follow-through: unchanged failed requests, changed requests,
   and a later success of the same operation are different observations. A shared
   wrapper name such as `tool_call` does not make unrelated requests a retry loop.
   `rundebug_explain` accepts `json: true` for structured action chains.
   Drill into a cited event with `rundebug_timeline` using `run` and `event`, or
   `hermes run-debug timeline --run <run-id> --event <event-id>`.
4. **Falsify before believing.** Falsifiers are read-only checks you run yourself.
   The debugger does not re-execute anything.
5. **Replay to reconstruct the sequence** when you need to walk the run in order:
   `/rundebug replay <run-id>`. Replay reads the recording; it executes nothing.
6. **Export privately if you need to keep or share it deliberately**:
   `hermes run-debug export --run <run-id> --out <private-path>` — operator CLI
   only, mode 0600.

## Reading the output honestly

- `reported` means the source said it. `observed` means this plugin computed it.
  `inferred` means it was derived by comparison. `unknown` means nobody said.
- A run reported as `completed` is **not** a passed task: absence of an error is not
  evidence of correctness.
- Token counts are reported usage, not money. Unknown billing stays null.
- Concurrent events appear as ambiguous pairs. Wall-clock order alone is not
  causation.
- Diagnose the actions and their handoffs first; the consuming agent can supply
  the UAT score. Import explicitly selected Cauldron harness JSON receipts with
  `--format cauldron` to inspect stage failures and recorded lifecycle outcomes.
  Keep setup, cleanup, process completion and business correctness separate.
- For old imports, missing request identity remains unknown. Re-import the
  explicitly selected original export to obtain richer call/result linkage;
  never reconstruct missing arguments from a diagnosis or overwrite old evidence.
- There are no causal findings in this build. A causal claim needs a controlled
  intervention, which is a separately gated lane.

## What this tool will not do

Re-run commands, repair a repository, read credentials, scan whole home
directories, upload anything, or produce a root cause from an incomplete log. If
you enabled a deferred lane (capture, live read, narration, experiments) it refuses
by name — read the refusal rather than assuming it worked.

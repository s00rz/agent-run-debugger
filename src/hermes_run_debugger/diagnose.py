"""Diagnosis: observed failures, ranked hypotheses, falsifiers, unknown gaps.

The separation this module enforces:

* **Observed** — what the imported evidence literally reports (a failed event, an
  interrupt, a completed-outcome-with-failures contradiction, an unusual reported
  usage figure). These are facts *about the log*, with provenance attached.
* **Hypothesis** — an explanation, ranked by the evidence behind it, carrying its
  own alternatives, the check that would falsify it, and whether a blocking gap
  caps its confidence.
* **Causal finding** — empty in this build. A causal claim requires a controlled
  intervention, which v1 deliberately does not perform.

Output/cost anomalies, repeated retries and missing logs are never promoted into a
cause: repeated correlation is evidence to investigate, not proof.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .event_v1 import Event
from .redact import safe_text
from .report import summarise_cost
from .timeline import Timeline

__all__ = ["CONFIDENCE_CATEGORIES", "RULES", "Diagnosis", "Hypothesis", "diagnose"]

CONFIDENCE_CATEGORIES = ("high", "medium", "low")
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}

#: Reported error text shapes that indicate a probable argument/validation fault.
_ARGUMENT_FAULT = re.compile(
    r"(invalid (?:argument|argument[s]?)|missing required|unexpected keyword|"
    r"usage: |schema validation|not a valid json|unknown option)",
    re.IGNORECASE,
)
#: Text that suggests a "success" which did not verify anything.
_ZERO_TEST = re.compile(r"(\b0 tests?\b|\bno tests? (?:ran|found|collected)\b|tests?=0\b)", re.IGNORECASE)
_WRAPPER_SUCCESS = re.compile(r"(exit code 0|exit 0|\bcompleted successfully\b)", re.IGNORECASE)
_RATE_LIMIT = re.compile(r"(429|rate limit|overloaded|too many requests)", re.IGNORECASE)
_USAGE_ANOMALY_TOKENS = 100_000

RULES: tuple[tuple[str, str], ...] = (
    ("R001", "repeated provider errors consistent with retry exhaustion"),
    ("R002", "unrecovered tool operation with no later success of the same operation"),
    ("R003", "repeated identical operations consistent with a loop"),
    ("R004", "wrapper success and child failure describe different scopes"),
    ("R005", "evidence gaps limit what can be concluded"),
    ("R006", "long wall-clock gap without monotonic evidence"),
    ("R007", "reported argument/validation fault"),
    ("R008", "provider error the source did not classify"),
    ("R009", "run stopped through an interrupt or cancellation"),
    ("R010", "reported usage anomaly (not a cost conclusion)"),
    ("R011", "reported success that verified nothing"),
    ("R012", "identical reported errors repeated"),
)


@dataclass(frozen=True)
class Hypothesis:
    rule_id: str
    statement: str
    confidence: str
    evidence_refs: tuple[str, ...]
    alternatives: tuple[str, ...]
    falsifiers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "statement": self.statement,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
            "alternatives": list(self.alternatives),
            "falsifiers": list(self.falsifiers),
        }


@dataclass
class Diagnosis:
    run_id: str
    observed: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    next_diagnostics: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    #: Correctness verdicts. v1 mints none: a completed run is not a passed task.
    correctness_claims: list[dict[str, Any]] = field(default_factory=list)
    #: Causal findings. Always empty here; a controlled intervention is required.
    causal_findings: list[dict[str, Any]] = field(default_factory=list)
    action_chains: list[dict[str, Any]] = field(default_factory=list)
    attempt_outcomes: list[dict[str, Any]] = field(default_factory=list)
    harness_events: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "observed": self.observed,
            "hypotheses": [hypothesis.as_dict() for hypothesis in self.hypotheses],
            "missing_evidence": list(self.missing_evidence),
            "next_diagnostics": list(self.next_diagnostics),
            "limitations": list(self.limitations),
            "cost": self.cost,
            "correctness_claims": self.correctness_claims,
            "causal_findings": self.causal_findings,
            "action_chains": self.action_chains,
            "attempt_outcomes": self.attempt_outcomes,
            "harness_events": self.harness_events,
        }

    def text(self) -> str:
        lines = [f"diagnosis for run {safe_text(self.run_id)}"]
        lines.append("action follow-through (recorded sequence, not proof of task correctness):")
        for chain in self.action_chains:
            lines.append(
                f"  - {safe_text(chain['operation'])} at {chain['failure_ref']}: "
                f"{chain['follow_through']}"
                + (f"; next result={chain['next_result_ref']}" if chain['next_result_ref'] else "")
                + (f"; source={safe_text(chain['evidence_ref'])}" if chain['evidence_ref'] else "")
            )
            lines.append(f"      inspect: {chain['next_check']}")
        if not self.action_chains:
            lines.append("  - no failed tool actions recorded")
        for attempt in self.attempt_outcomes:
            lines.append(f"attempt {attempt['attempt_id']}: reported outcome={attempt['outcome']}")
        if self.harness_events:
            lines.append("harness boundaries and reported outcomes:")
            for event in self.harness_events:
                lines.append(
                    f"  - {event['event_ref']} {safe_text(event['name'])}: {event['status']}"
                    + (f"; {safe_text(event['error'])}" if event['error'] else "")
                    + (f"; source={safe_text(event['evidence_ref'])}" if event['evidence_ref'] else "")
                )
        lines.append(f"observed (what the imported evidence reports): {len(self.observed)}")
        for item in self.observed:
            lines.append(
                f"  - {item['kind']} at {safe_text(str(item.get('event_ref') or 'n/a'))}: "
                f"{safe_text(item['detail'])} [provenance={item.get('provenance', 'reported')}]"
            )
        lines.append(f"hypotheses (evidence-linked, ranked): {len(self.hypotheses)}")
        for hypothesis in self.hypotheses:
            lines.append(
                f"  - [{hypothesis.rule_id}] {safe_text(hypothesis.statement)} "
                f"(confidence: {hypothesis.confidence}; evidence: "
                f"{', '.join(safe_text(ref) for ref in hypothesis.evidence_refs)})"
            )
            for alternative in hypothesis.alternatives:
                lines.append(f"      alternative reading: {safe_text(alternative)}")
            for falsifier in hypothesis.falsifiers:
                lines.append(f"      falsify by: {safe_text(falsifier)}")
        lines.append("missing evidence:")
        for item in self.missing_evidence or ["(none recorded)"]:
            lines.append(f"  - {safe_text(item)}")
        lines.append("next checks (read-only; nothing is executed here):")
        for step in self.next_diagnostics:
            lines.append(f"  - {safe_text(step)}")
        lines.append("cost (reported usage only):")
        lines.append(
            f"  - reported total: {self.cost.get('reported_total')} "
            f"(kind: {self.cost.get('cost_kind')}); events with unknown cost: "
            f"{self.cost.get('events_with_unknown_cost')}"
        )
        lines.append(
            "causal findings: none — a causal finding requires a controlled intervention, "
            "which this build does not perform"
        )
        lines.append("limitations:")
        for item in self.limitations:
            lines.append(f"  - {safe_text(item)}")
        return "\n".join(lines) + "\n"


_LIMITATIONS = (
    "An evidence reference is a pointer into the selected source, not a receipt: it is not "
    "checked, authenticated or signed.",
    "Imported logs are testimony about a run, not proof of what happened; every item above keeps "
    "its provenance and none is upgraded.",
    "Cost figures are the source's reported usage. Unknown billing stays unknown and reported "
    "tokens are never converted into money without a named price source.",
    "Absence of an error in the log is not evidence of correctness: a missing terminal event "
    "leaves the outcome unknown.",
)

_NEXT_STEP_FAILING_EVIDENCE = (
    "read-only: inspect the referenced source lines for the failing events and confirm the "
    "reported status against the raw text; nothing is executed"
)

_NEXT_STEPS = (
    "read-only: re-run `timeline` with a wider `--max-events` and compare the gap list against "
    "the selected source; no execution is triggered here",
    _NEXT_STEP_FAILING_EVIDENCE,
    "read-only: for a retry chain, export the run and compare attempt order; no execution",
)

#: Names "the failing events", so it is only offered when one was actually observed; on a clean
#: run it reads as though a failure existed.
_NEXT_STEPS_FAILURE_ONLY = frozenset({_NEXT_STEP_FAILING_EVIDENCE})


def diagnose(timeline: Timeline, *, cost: Mapping[str, Any] | None = None) -> Diagnosis:
    """Produce a Diagnosis v1 view of a built timeline."""
    events = [entry.event for entry in timeline.entries]
    for event in timeline.quarantined:
        events.append(event)

    diagnosis = Diagnosis(
        run_id=timeline.run_id,
        limitations=list(_LIMITATIONS),
        cost=dict(cost) if cost is not None else summarise_cost(events),
    )

    _observe(diagnosis, events, timeline)
    diagnosis.action_chains = _action_chains(events)
    diagnosis.attempt_outcomes = [
        {"attempt_id": str(attempt.get("attempt_id") or ""),
         "outcome": str(attempt.get("outcome") or "unknown")}
        for attempt in timeline.attempts
    ]
    harness_attempts = {attempt.get("attempt_id") for attempt in timeline.attempts
                        if attempt.get("role") == "harness"}
    diagnosis.harness_events = [
        {"event_ref": event.event_id, "attempt_ref": event.attempt_id,
         "name": event.name or event.event_kind, "status": event.reported_status,
         "error": event.reported_error, "evidence_ref": event.evidence_ref}
        for event in events if event.attempt_id in harness_attempts
    ]

    # A next check that names "the failing events" only makes sense when the evidence reports
    # one. The failure-independent checks stay on every run.
    observed_failure = _observed_failure(diagnosis)
    diagnosis.next_diagnostics = [
        step
        for step in _NEXT_STEPS
        if step not in _NEXT_STEPS_FAILURE_ONLY or observed_failure
    ]

    _hypothesise(diagnosis, events, timeline)
    _collect_missing_evidence(diagnosis, timeline)

    blocking_gap = _blocking_gap(timeline)
    if blocking_gap:
        diagnosis.hypotheses = [
            Hypothesis(
                rule_id=hypothesis.rule_id,
                statement=hypothesis.statement,
                confidence="low",
                evidence_refs=hypothesis.evidence_refs,
                alternatives=hypothesis.alternatives,
                falsifiers=hypothesis.falsifiers,
            )
            for hypothesis in diagnosis.hypotheses
        ]

    diagnosis.hypotheses.sort(key=lambda h: (_CONFIDENCE_RANK.get(h.confidence, 3), h.rule_id))
    return diagnosis


def _observe(diagnosis: Diagnosis, events: Sequence[Event], timeline: Timeline) -> None:
    for event in events:
        if event.reported_status == "failed":
            diagnosis.observed.append({
                "kind": f"failure:{event.event_kind}",
                "event_ref": event.event_id,
                "detail": (
                    f"source reports a failed {event.event_kind}"
                    + (f" ({event.name})" if event.name else "")
                    + (f": {safe_text(event.reported_error)}" if event.reported_error else "")
                ),
                "provenance": "reported",
            })
        elif event.reported_status == "interrupted":
            diagnosis.observed.append({
                "kind": "interrupted:event",
                "event_ref": event.event_id,
                "detail": f"source reports an interrupted {event.event_kind}"
                          + (f": {safe_text(event.reported_error)}" if event.reported_error else ""),
                "provenance": "reported",
            })

        usage = event.usage
        total = usage.total_tokens()
        if total is not None and total >= _USAGE_ANOMALY_TOKENS:
            diagnosis.observed.append({
                "kind": "anomaly:reported_usage",
                "event_ref": event.event_id,
                "detail": (
                    f"reported token usage of {total} for one {event.event_kind} is unusual; "
                    "this is a reported figure, not a cost conclusion"
                ),
                "provenance": "reported",
            })
        if event.reported_status == "ok" and event.reported_error \
                and _ZERO_TEST.search(event.reported_error):
            diagnosis.observed.append({
                "kind": "false_success:suspected",
                "event_ref": event.event_id,
                "detail": (
                    "the source reports a successful step while its own message says nothing was "
                    f"verified ({safe_text(event.reported_error)}); a success flag is not a test result"
                ),
                "provenance": "reported",
            })

    failed = [event for event in events if event.reported_status == "failed"]
    outcomes = {str(attempt.get("outcome")) for attempt in timeline.attempts}
    if failed and outcomes == {"completed"} and any(not _later_success(events, event) for event in failed):
        diagnosis.observed.append({
            "kind": "contradiction:outcome_vs_events",
            "event_ref": failed[0].event_id,
            "detail": (
                f"the attempt is reported completed while {len(failed)} event(s) report failure "
                f"({', '.join(event.event_id for event in failed[:5])}); "
                "but completion does not establish whether those operations recovered or the task passed"
            ),
            "provenance": "inferred",
        })
    if not events:
        diagnosis.observed.append({
            "kind": "empty:no_events",
            "event_ref": None,
            "detail": "no events were imported for this run, so nothing can be concluded",
            "provenance": "unknown",
        })
    for event in timeline.quarantined:
        diagnosis.observed.append({
            "kind": "quarantine:contradictory_parents",
            "event_ref": event.event_id,
            "detail": "the event belongs to a cyclic parent graph and was not ordered",
            "provenance": "inferred",
        })


def _blocking_gap(timeline: Timeline) -> bool:
    return any(
        gap.kind in {"missing_parent", "partial_source", "truncation", "unsupported_kind"}
        for gap in timeline.gaps
    )


def _hypothesise(diagnosis: Diagnosis, events: Sequence[Event], timeline: Timeline) -> None:
    by_kind: dict[str, list[Event]] = {}
    for event in events:
        by_kind.setdefault(event.event_kind, []).append(event)

    provider_errors = by_kind.get("provider_error", [])
    if provider_errors:
        confidence = "high" if len(provider_errors) >= 3 else "medium" if len(provider_errors) == 2 else "low"
        rate_limited = [event for event in provider_errors
                        if event.reported_error and _RATE_LIMIT.search(event.reported_error)]
        diagnosis.hypotheses.append(Hypothesis(
            rule_id="R001",
            statement=(
                "the run is consistent with provider-side failure or throttling exhausting the "
                f"retry budget: {len(provider_errors)} provider error(s) reported"
                + (f", {len(rate_limited)} of them rate-limit or overload shaped" if rate_limited else "")
            ),
            confidence=confidence,
            evidence_refs=tuple(event.event_id for event in provider_errors),
            alternatives=(
                "the provider errors are unrelated transients and the run failed for another reason",
                "a local network or credential problem is presented by the provider as a server error",
                "the retry budget was adequate and the last error was the only fatal one",
            ),
            falsifiers=(
                "run one bounded request against the same provider from outside this plugin and "
                "compare the reported error",
                "check the source's timestamps against the provider's own status history for that window",
                "read the referenced error text in the raw source and compare the status codes reported",
            ),
        ))
        unclassified = [event for event in provider_errors
                        if not event.reported_error or not _RATE_LIMIT.search(event.reported_error)]
        if unclassified:
            diagnosis.hypotheses.append(Hypothesis(
                rule_id="R008",
                statement=(
                    f"{len(unclassified)} provider error(s) were not classified by the source, so the "
                    "cause is not identifiable from this log alone"
                ),
                confidence="low",
                evidence_refs=tuple(event.event_id for event in unclassified),
                alternatives=(
                    "the provider returned a documented error the source did not record",
                    "the error text was truncated or redacted before import",
                ),
                falsifiers=(
                    "obtain the provider response body from the source system and compare it with the "
                    "recorded status",
                ),
            ))

    tool_events = by_kind.get("tool_call", [])
    # A failed tool *result* counts exactly like a failed tool start: the source
    # recorded a tool step that reported failure.
    failed_tools = [event for event in tool_events if event.reported_status == "failed"]
    # An operation is unrecovered only when the source records no later success of that SAME
    # operation. Recovery is never inferred from unrelated later activity, and an unrelated
    # later success never licenses a claim that the run stopped here.
    unrecovered = [event for event in failed_tools if not _later_success(tool_events, event)]
    if unrecovered:
        continued = any(_later_run_activity(events, event) for event in unrecovered)
        # A run holds one attempt per try: retry chains and subagent branches are separate
        # attempts. Quoting an outcome from an attempt that did not record these failures
        # describes evidence the source does not have, so only the owning attempts count.
        own_outcomes = _own_attempt_outcomes(timeline, unrecovered)
        # Step-scoped, so the ceiling is medium. A blocking gap weakens it further.
        statement = (
            "this evidence is consistent with an unrecovered tool failure: "
            f"{len(unrecovered)} tool call(s) report failure with no recorded later success of "
            "the same operation"
            + (
                f" ({', '.join(event.name for event in unrecovered if event.name)})"
                if any(event.name for event in unrecovered) else ""
            )
        )
        if continued:
            statement += (
                "; the source records later successful activity after that failure, so the run "
                "continued past it and the failed operation is unrecovered"
            )
        else:
            statement += "; the source records no later successful activity after that failure"
        if own_outcomes == {"failed"}:
            statement += (
                "; the attempt outcome is reported failed, which is the source's own terminal "
                "report, not a terminal event this build observed"
            )
        elif own_outcomes == {"interrupted"}:
            statement += (
                "; the attempt outcome is reported interrupted, which is another reported value, "
                "not an observed terminal event"
            )
        elif own_outcomes == {"completed"}:
            statement += (
                "; the attempt outcome is reported completed, which records a finished run and "
                "not a passed task"
            )
        elif len(own_outcomes) > 1:
            statement += (
                "; the affected operations sit in attempts whose reported outcomes differ "
                f"({', '.join(sorted(own_outcomes))}), so no single terminal outcome is quoted here"
            )
        else:
            statement += (
                "; the run's terminal outcome is not recorded in this evidence, so whether the run "
                "ended at this step is unknown"
            )
        recovered = len(failed_tools) - len(unrecovered)
        if recovered:
            statement += (
                f"; {recovered} further failed operation(s) have a recorded later success of the "
                "same operation and are not counted here"
            )
        diagnosis.hypotheses.append(Hypothesis(
            rule_id="R002",
            statement=statement,
            confidence="low" if _blocking_gap(timeline) else "medium",
            evidence_refs=tuple(event.event_id for event in unrecovered),
            alternatives=(
                "the failed operation was retried or handled in a way this log does not record",
                "the tool result text describes a partial success the status field does not capture",
                "the failing step is downstream of an earlier, unrecorded failure",
            ),
            falsifiers=(
                "read the referenced source lines for the failing events and check whether a later "
                "attempt of the same operation succeeded",
                "obtain the source's own terminal record for this run and compare it with the "
                "reported attempt outcome",
            ),
        ))

    argument_faults = [event for event in events
                       if event.reported_error and _ARGUMENT_FAULT.search(event.reported_error)]
    if argument_faults:
        diagnosis.hypotheses.append(Hypothesis(
            rule_id="R007",
            statement=(
                f"{len(argument_faults)} event(s) report an argument or validation fault, which is "
                "consistent with a malformed request rather than a service failure"
            ),
            confidence="medium",
            evidence_refs=tuple(event.event_id for event in argument_faults),
            alternatives=(
                "the arguments were correct and the receiving tool rejected them for an unrelated reason",
                "the fault message is a wrapper's paraphrase of a deeper failure",
            ),
            falsifiers=(
                "re-run the same operation with the argument reference inspected manually",
                "compare the reported arguments reference against the tool's documented schema",
            ),
        ))

    signature_groups: dict[tuple[str, str, str], list[Event]] = {}
    for event in events:
        if event.reported_status == "failed" and event.reported_error:
            signature_groups.setdefault(
                (event.attempt_id, event.name or "", event.reported_error.strip()), []
            ).append(event)
    for (_attempt, _name, signature), group in sorted(signature_groups.items()):
        if len(group) >= 2:
            diagnosis.hypotheses.append(Hypothesis(
                rule_id="R012",
                statement=(
                    f"the same reported error appears {len(group)} times ({safe_text(signature[:120])}), "
                    "which is consistent with a deterministic fault rather than transient noise"
                ),
                confidence="medium" if len(group) >= 3 else "low",
                evidence_refs=tuple(event.event_id for event in group),
                alternatives=(
                    "the repeated text is a generic wrapper message covering different underlying faults",
                    "the retry loop never changed its inputs",
                ),
                falsifiers=(
                    "compare the argument references across the repeated events; identical references "
                    "would support a deterministic fault",
                ),
            ))

    repeated_tool = [event for event in tool_events if event.reported_status == "failed" and event.name]
    groups: dict[tuple[str, str, str], list[Event]] = {}
    for event in repeated_tool:
        if event.arguments_ref and event.arguments_ref.startswith("lossy:"):
            continue
        if event.arguments_ref:
            identity = event.arguments_ref
        elif event.reported_error and event.name not in {"tool_call", "execute_code", "terminal"}:
            identity = "error-only:" + event.reported_error
        else:
            continue
        groups.setdefault((event.attempt_id, str(event.name), identity), []).append(event)
    looping = {key: group for key, group in groups.items() if len(group) >= 3}
    if looping:
        worst = max(looping, key=lambda key: len(looping[key]))
        count = len(looping[worst])
        evidence = tuple(event.event_id for event in looping[worst])
        diagnosis.hypotheses.append(Hypothesis(
            rule_id="R003",
            statement=(
                f"the run is consistent with a loop: {worst[1]!r} was attempted {count} times "
                "with a failure status each time"
                + (" and the same argument reference" if not worst[2].startswith("error-only:")
                   else "; argument identity is unavailable, only the same reported error is known")
            ),
            confidence="medium" if count >= 4 and not worst[2].startswith("error-only:") else "low",
            evidence_refs=evidence,
            alternatives=(
                "the repeated attempts were deliberate retries with changing inputs",
                "the tool name is shared by different operations that each failed once",
            ),
            falsifiers=(
                "compare the recorded argument references across the attempts; different references "
                "would falsify a loop",
                "re-run the same operation manually outside this plugin to see whether the failure is "
                "deterministic",
            ),
        ))

    ok_wrappers = [event for event in events
                   if event.reported_status == "ok" and event.reported_error
                   and _WRAPPER_SUCCESS.search(event.reported_error)]
    if ok_wrappers and failed_tools:
        evidence = tuple(event.event_id for event in ok_wrappers) + tuple(
            event.event_id for event in failed_tools
        )
        diagnosis.hypotheses.append(Hypothesis(
            rule_id="R004",
            statement=(
                "a wrapper reported its own success while a child step reported failure; the two "
                "describe different scopes and the log does not resolve which one describes the task"
            ),
            confidence="medium",
            evidence_refs=evidence,
            alternatives=(
                "the wrapper's success describes its own process, while the child failure is the "
                "real outcome of the task",
                "the child failure was non-fatal and the wrapper's success is the accurate summary",
                "the wrapper aggregated several children and reported the aggregate, hiding this one",
            ),
            falsifiers=(
                "locate the wrapper's own exit evidence in the source and compare it with the child's "
                "reported error",
                "re-run only the child step manually outside this plugin and compare the outcome",
            ),
        ))

    interrupts = [event for event in events if event.reported_status == "interrupted"
                  or event.event_kind == "interrupt"]
    if interrupts:
        diagnosis.hypotheses.append(Hypothesis(
            rule_id="R009",
            statement=(
                "the run stopped through an interrupt or cancellation rather than reaching a "
                "conclusion of its own"
            ),
            confidence="high" if len(interrupts) >= 2 else "medium",
            evidence_refs=tuple(event.event_id for event in interrupts),
            alternatives=(
                "an operator stopped the run deliberately",
                "the host cancelled the run under resource pressure or a timeout policy",
                "the interrupt marker is a downstream artefact of a provider failure",
            ),
            falsifiers=(
                "check the host's own logs for a cancellation decision in the same window",
                "compare the last recorded event before the interrupt with the recorded budget or timeout",
            ),
        ))

    if timeline.suspect_windows:
        evidence = tuple(
            dict.fromkeys(
                event_id
                for window in timeline.suspect_windows
                for event_id in (window.get("after"), window.get("before"))
                if event_id
            )
        )
        observed_windows = [window for window in timeline.suspect_windows
                            if window.get("status") == "observed"]
        diagnosis.hypotheses.append(Hypothesis(
            rule_id="R006",
            statement=(
                f"{len(timeline.suspect_windows)} long wall-clock gap(s) appear in the timeline; "
                + (
                    "monotonic evidence supports the elapsed time, but not a cause"
                    if observed_windows else
                    "without monotonic evidence a stall cannot be distinguished from a suspended "
                    "process or clock drift"
                )
            ),
            confidence="medium" if observed_windows else "low",
            evidence_refs=evidence,
            alternatives=(
                "the process was suspended while the wall clock advanced",
                "the wall clock drifted or was stepped during the run",
                "a genuinely slow step consumed the interval",
                "the events were written out of order by the source",
            ),
            falsifiers=(
                "obtain a source that records a monotonic clock for the same window",
                "compare the host's own process accounting for the same interval",
            ),
        ))

    gap_events = tuple(
        gap.event_id for gap in timeline.gaps if gap.event_id
    )
    if gap_events:
        diagnosis.hypotheses.append(Hypothesis(
            rule_id="R005",
            statement=(
                f"{len(timeline.gaps)} evidence gap(s) are recorded for this run, so conclusions "
                "drawn from this timeline are limited"
            ),
            confidence="low",
            evidence_refs=gap_events,
            alternatives=(
                "the missing evidence is unrelated to the failure",
                "the gap is an artefact of the export rather than of the run",
            ),
            falsifiers=(
                "re-import a complete export of the same run and compare the gap list",
            ),
        ))


def _same_operation(event: Event, failed: Event) -> bool:
    """True when ``event`` records the same operation as ``failed``.

    The attempt is part of the identity: a retry chain and a subagent branch record their
    own tool calls, so a same-named success there is a different operation.

    Where the source gives an operation a span, the span is that identity and a later
    success of the same span is the same operation. Where it does not, the tool name is
    all that is left — and a name is not an identity: two ``Read`` calls are different
    operations. In that case the later event must record a causal link to the failed one
    before it can count as recovery, because a name match alone would silently erase a
    reported failure.
    """
    if event.attempt_id != failed.attempt_id:
        return False
    if event.name != failed.name:
        return False
    if failed.span_id is not None and event.span_id == failed.span_id:
        return True
    if ((failed.arguments_ref or "").startswith("lossy:")
            or (event.arguments_ref or "").startswith("lossy:")):
        return False
    if (failed.arguments_ref and event.arguments_ref
            and not failed.arguments_ref.startswith("lossy:")
            and not event.arguments_ref.startswith("lossy:")):
        return event.arguments_ref == failed.arguments_ref
    if failed.span_id is not None or event.span_id is not None:
        return False
    return failed.event_id in event.causal_parents


def _action_chains(events: Sequence[Event]) -> list[dict[str, Any]]:
    """Follow each failed action without equating different requests with recovery."""
    tools = sorted(
        (event for event in events if event.event_kind == "tool_call"),
        key=lambda event: event.source_sequence,
    )
    chains: list[dict[str, Any]] = []
    for index, failed in enumerate(tools):
        if failed.reported_status != "failed":
            continue
        later = [event for event in tools[index + 1:]
                 if event.attempt_id == failed.attempt_id and event.name == failed.name
                 and event.reported_status in {"ok", "failed", "interrupted"}]
        recovered = next((event for event in later
                          if event.reported_status == "ok" and _same_operation(event, failed)), None)
        successor = recovered or next(iter(later), None)
        if recovered:
            follow_through = "same operation has a later reported success; task correctness remains separate"
        elif (successor and failed.arguments_ref and successor.arguments_ref
              and not failed.arguments_ref.startswith("lossy:")
              and not successor.arguments_ref.startswith("lossy:")):
            if failed.arguments_ref == successor.arguments_ref:
                follow_through = "unchanged request failed again; no later matching success recorded"
            else:
                follow_through = "different request followed; not proof that the original work recovered"
        elif successor:
            follow_through = "same tool used later, but request identity is missing; recovery unknown"
        else:
            follow_through = "no later result for this operation recorded"
        chains.append({
            "failure_ref": failed.event_id,
            "attempt_ref": failed.attempt_id,
            "operation": failed.name or "unknown tool",
            "arguments_ref": failed.arguments_ref,
            "evidence_ref": failed.evidence_ref,
            "follow_through": follow_through,
            "next_result_ref": successor.event_id if successor else None,
            "next_result_source": successor.evidence_ref if successor else None,
            "recovery": "reported_success" if recovered else "not_established",
            "next_check": _action_check(failed),
        })
    return chains


def _action_check(event: Event) -> str:
    """Suggest a check from recorded error text, never claim it is the cause."""
    error = (event.reported_error or "").lower()
    if any(marker in error for marker in ("validation", "requires a 'name'", "unexpected", "record_ids")):
        return "compare this request with the tool schema and returned manifest; inspect changed-request follow-through"
    if any(marker in error for marker in ("permission", "access is denied", "modulenotfounderror", "not found")):
        return "check the exact execution principal, interpreter, installed dependency and pathname at this boundary"
    if any(marker in error for marker in ("cannot authenticate", "refus", "duplicate target", "stale")):
        return ("inspect the refusing owner's evidence and identity preconditions; "
                "compare the next request before retrying")
    if any(marker in error for marker in ("timeout", "timed out", "deadline")):
        return "inspect child execution and stage receipts during this interval; distinguish waiting from active work"
    return "read this action's source result and the next same-operation result; verify the intended durable effect"


def _later_success(events: Sequence[Event], failed: Event) -> bool:
    """True when the SAME operation succeeded later in the same attempt.

    This is the only shape that counts as recovery of the failed operation. It says
    nothing about the run as a whole: a matching success elsewhere is not a retry, and
    the absence of one is not evidence that the run terminated at this step.
    """
    for event in events:
        if event.source_sequence <= failed.source_sequence:
            continue
        if event.reported_status == "ok" and _same_operation(event, failed):
            return True
    return False


def _later_run_activity(events: Sequence[Event], failed: Event) -> bool:
    """True when the source records ANY later successful event after this failure.

    This is activity, never recovery. It exists so a step-scoped failure statement cannot
    imply that the whole run stopped: an unrelated later success does not repair the failed
    operation, but it does prove the run carried on past it.
    """
    for event in events:
        if event.source_sequence <= failed.source_sequence:
            continue
        if event.reported_status == "ok":
            return True
    return False


def _observed_failure(diagnosis: Diagnosis) -> bool:
    """True when the imported evidence actually reports a failure or an interruption.

    The retention note explains why a *mechanism* cannot be named, so it only makes sense
    when there is something whose mechanism is in question.
    """
    return any(
        str(item.get("kind", "")).startswith(("failure:", "interrupted:"))
        for item in diagnosis.observed
    )


def _own_attempt_outcomes(timeline: Timeline, events: Sequence[Event]) -> set[str]:
    """Reported outcomes of the attempts that actually recorded these events.

    A run can carry several attempts: every retry in a chain and every subagent branch is
    its own attempt with its own reported outcome. Quoting an outcome belonging to an
    attempt that did not record the failure describes evidence the source does not have,
    which is the same scope error as quoting a run-level outcome for a step.
    """
    by_id = {
        str(attempt.get("attempt_id")): str(attempt.get("outcome") or "unknown")
        for attempt in timeline.attempts
    }
    return {by_id.get(str(event.attempt_id), "unknown") for event in events}


def _collect_missing_evidence(diagnosis: Diagnosis, timeline: Timeline) -> None:
    if timeline.text_dropped_by_policy and _observed_failure(diagnosis):
        # Say it plainly: the reason no mechanism is offered is the retention policy, not a
        # lack of care. Silence here would look like an unexplained gap. Gated on an observed
        # failure: on a run that recorded none there is no mechanism to name, and the note
        # would read as though one had been withheld by policy.
        diagnosis.missing_evidence.append(
            "error and tool-result text was not retained under the active retention policy "
            "(metadata-only), so the mechanism behind a failure cannot be named from this "
            "evidence — re-import with a retention mode that keeps redacted summaries if the "
            "mechanism matters more than minimisation"
        )
    for gap in timeline.gaps:
        diagnosis.missing_evidence.append(
            f"gap ({gap.kind.replace('_', ' ')}): {safe_text(gap.detail)}"
        )
    if any(window.get("monotonic_delta_ms") is None for window in timeline.suspect_windows):
        diagnosis.missing_evidence.append(
            "no monotonic clock evidence for at least one long gap, so elapsed time cannot be "
            "separated from suspend or drift"
        )
    if not timeline.attempts or any(attempt.get("outcome", "unknown") == "unknown" for attempt in timeline.attempts):
        diagnosis.missing_evidence.append(
            "the source's own terminal decision is not represented for at least one attempt; "
            "later successful activity alone cannot establish the task outcome"
        )
    for note in timeline.clock_notes:
        diagnosis.missing_evidence.append(f"clock limitation: {safe_text(note)}")
    # de-duplicate while preserving order
    seen: set[str] = set()
    diagnosis.missing_evidence = [
        item for item in diagnosis.missing_evidence
        if not (item in seen or seen.add(item))
    ]


def iter_rule_ids() -> Iterable[str]:
    return (rule_id for rule_id, _description in RULES)

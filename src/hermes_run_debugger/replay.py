"""Inert replay: re-read the recorded timeline, execute nothing.

This module imports no execution primitive — no ``subprocess``, no ``socket``, no
network client, no database driver. It is a pure function over already-loaded
evidence, which is what makes "replay" safe to offer to someone debugging a
failure they do not fully understand yet.

Imported command and error text is *displayed as data*. It is never executed, and
it is never accepted as an instruction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .event_v1 import Event
from .redact import safe_text
from .timeline import Timeline, _display_summary

__all__ = ["REPLAY_SCHEMA_ID", "ReplayLog", "ReplayStep", "replay_events", "replay_text"]

REPLAY_SCHEMA_ID = "hermes-run-debugger/replay"

_NOTES = (
    "replay is inert: it re-reads the recorded timeline and executes nothing "
    "(no subprocess, no network, no live database, no writes)",
    "imported command/error text is displayed as data and is not executed",
    "ordering preserves the supplied event order (Timeline input uses causal parents with source tie-breaks); "
    "display order alone is not causation",
)


@dataclass(frozen=True)
class ReplayStep:
    index: int
    kind: str
    status: str
    provenance: str
    at: str | None
    name: str | None
    parent_ref: str | None
    text: str
    event_ref: str | None = None
    attempt_ref: str | None = None
    source_sequence: int | None = None
    span_ref: str | None = None
    evidence_ref: str | None = None
    arguments_ref: str | None = None
    causal_parents: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "status": self.status,
            "provenance": self.provenance,
            "at": self.at,
            "name": self.name,
            "parent_ref": self.parent_ref,
            "text": self.text,
            "event_ref": self.event_ref,
            "attempt_ref": self.attempt_ref,
            "source_sequence": self.source_sequence,
            "span_ref": self.span_ref,
            "evidence_ref": self.evidence_ref,
            "arguments_ref": self.arguments_ref,
            "causal_parents": list(self.causal_parents),
        }


@dataclass
class ReplayLog:
    run_id: str
    steps: list[ReplayStep] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_pairs: Sequence[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    schema: str = REPLAY_SCHEMA_ID

    def text(self, *, expand_ambiguity: bool = False, full_text: bool = False) -> str:
        lines = [f"inert replay of run {safe_text(self.run_id)} (nothing is executed)"]
        for note in self.notes:
            lines.append(f"note: {safe_text(note)}")
        if not self.steps:
            lines.append("(no events to replay)")
        for step in self.steps:
            parents = step.causal_parents or ((step.parent_ref,) if step.parent_ref else ())
            parent = f" parents={','.join(parents)}" if parents else ""
            name = f" {_display_summary(step.name)}" if step.name else ""
            lines.append(
                f"{step.index:04d} event={safe_text(step.event_ref)} attempt={safe_text(step.attempt_ref)} "
                f"seq={step.source_sequence} {safe_text(step.kind)}{name} status={safe_text(step.status)} "
                f"at={safe_text(step.at or 'unknown')} [{safe_text(step.provenance)}]{safe_text(parent)} "
                f":: {_display_summary(step.text, full=full_text)}"
            )
            refs = {"span": step.span_ref, "evidence": step.evidence_ref, "arguments": step.arguments_ref}
            if any(refs.values()):
                lines.append("     " + " ".join(f"{key}={safe_text(value)}" for key, value in refs.items() if value))
        lines.append(
            f"ambiguous order: {len(self.ambiguous_pairs)} pair(s) have no causal path; "
            "this does not prove concurrency. Details: ambiguous_pairs or expand_ambiguity=True."
        )
        for pair in self.ambiguous_pairs if expand_ambiguity else ():
            lines.append(f"ambiguous order: {safe_text(pair[0])} <-> {safe_text(pair[1])} (causal order unknown)")
        for gap in self.gaps:
            lines.append(
                f"gap: {str(gap.get('kind', '')).replace('_', ' ')} — {safe_text(gap.get('detail', ''))}"
            )
        return "\n".join(lines) + "\n"

    def as_dict(self, *, expand_ambiguity: bool = False) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "steps": [step.as_dict() for step in self.steps],
            "gaps": self.gaps,
            "ambiguous_pairs": [list(pair) for pair in self.ambiguous_pairs] if expand_ambiguity else [],
            "ambiguity": {
                "pair_count": len(self.ambiguous_pairs),
                "expanded": expand_ambiguity,
                "detail": "iterate ambiguous_pairs or as_dict(expand_ambiguity=True)",
                "rule": "two timeline steps are ambiguous iff neither reaches the other via causal_parents",
            },
            "notes": list(self.notes),
        }


def replay_events(
    events: Iterable[Event],
    *,
    run_id: str,
    gaps: Sequence[Mapping[str, Any]] | None = None,
    ambiguous_pairs: Sequence[tuple[str, str]] | None = None,
) -> ReplayLog:
    """Walk recorded events in their stored order. Pure; no side effects."""
    log = ReplayLog(run_id=run_id, notes=list(_NOTES))
    for index, event in enumerate(events):
        text = event.reported_error or ""
        parent = event.causal_parents[0] if event.causal_parents else None
        log.steps.append(
            ReplayStep(
                index=index,
                kind=event.event_kind,
                status=event.reported_status,
                provenance=event.semantic_provenance,
                at=event.wall_time,
                name=event.name,
                parent_ref=parent,
                # Text is data. It is escaped for display and never interpreted.
                text=text,
                event_ref=event.event_id,
                attempt_ref=event.attempt_id,
                source_sequence=event.source_sequence,
                span_ref=event.span_id,
                evidence_ref=event.evidence_ref,
                arguments_ref=event.arguments_ref,
                causal_parents=event.causal_parents,
            )
        )
    log.gaps = [dict(gap) for gap in (gaps or [])]
    log.ambiguous_pairs = ambiguous_pairs if ambiguous_pairs is not None else []
    return log


def replay_text(
    events: Iterable[Event] | Timeline, *, expand_ambiguity: bool = False, full_text: bool = False, **kwargs: Any
) -> str:
    """Convenience: replay either a Timeline or an event iterable and return text."""
    if isinstance(events, Timeline):
        log = replay_events(
            (entry.event for entry in events.entries),
            run_id=events.run_id,
            gaps=[gap.as_dict() for gap in events.gaps],
            ambiguous_pairs=events.ambiguous_pairs,
        )
        return log.text(expand_ambiguity=expand_ambiguity, full_text=full_text)
    return replay_events(events, **kwargs).text(expand_ambiguity=expand_ambiguity, full_text=full_text)

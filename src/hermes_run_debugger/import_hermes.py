"""Hermes session-export adapter — the native lane for this host.

Consumes ``hermes sessions export --format jsonl``: one JSON record per session, where
each record carries the session's own fields (``id``, ``model``, ``started_at``,
``ended_at``, ``end_reason``, ``message_count``, …) plus a ``messages`` array of
``user`` / ``assistant`` / ``tool`` rows.

Documented source semantics this adapter implements:

* **One record per session, not per event.** The export is a session envelope, not a
  flat event stream, so the adapter walks ``messages`` in stored order and uses that
  order as ``source_sequence``. Wall-clock order is never used to invent causality.
* **``tool_calls`` is serialised with ``str()``, not ``json.dumps()``.** A real export
  contains single-quoted Python literals, so the adapter tries JSON first and falls
  back to ``ast.literal_eval``. A value that parses as neither is reported as an
  unsupported record, never guessed at.
* **``tool_message.content`` is a JSON object string** carrying ``exit_code`` and
  ``error`` for command-shaped tools. Status is derived only from explicit fields:
  failure status, truthy ``error`` or boolean ``success=False`` takes precedence;
  integer exit codes, boolean ``success`` and known status enums are mapped next.
  A payload without a status signal stays ``unknown``. Prose is never a verdict.
* **Ordinary exports report no task outcome.** ``end_reason`` describes why the session
  process stopped (``user_exit``, ``startup_orphan_reap``, …), which is not a task
  result. The attempt outcome stays ``unknown`` unless an explicit typed ``outcome``
  is present; tool success or assistant prose never supplies a task outcome.

What is deliberately NOT retained: prompt text and tool arguments (opaque references
only), and tool-result text under ``retention_mode="metadata-only"``.

Deliberately not implemented here: reading the live session database. A live read is a
separately gated lane (``live_read_enabled``); this adapter consumes an explicit,
already-exported file.
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .event_v1 import EVENT_KINDS, OUTCOMES, Gap, validate_trace_records
from .identity import content_ref, redacted_content_ref, sha256_hex, stable_id
from .import_v1 import ImportRefused, TraceImport, guard_source
from .jsonio import InputLimitExceeded, MalformedRecord, ScanState, StreamLimits, scan_jsonl
from .redact import redact_record, sanitise_message

__all__ = ["MAX_RESULT_SUMMARY", "RETENTION_MODES", "import_hermes_trace"]

#: Bound on retained tool-result summary text (redacted again before persistence).
MAX_RESULT_SUMMARY = 500

RETENTION_MODES = ("redacted-summary", "metadata-only")

#: Message roles the export uses. Anything else is counted as unsupported, not inferred.
_KNOWN_ROLES = frozenset({"user", "assistant", "tool"})

#: Tool-result fields that name a failure. Checked in order; the first signal wins.
_FAILURE_STATUS_VALUES = frozenset({"failed", "error", "blocked"})


@dataclass
class _Source:
    """A selected file and the session records inside it."""

    path: Path
    state: ScanState | None
    sessions: list[Mapping[str, Any]]
    partial: bool = False


def import_hermes_trace(
    paths: str | Path | Sequence[str | Path],
    *,
    limits: StreamLimits | None = None,
    allow_partial: bool = False,
    retention_mode: str = "redacted-summary",
    run_id: str | None = None,
) -> TraceImport:
    """Import one or more explicitly selected Hermes session exports."""
    if retention_mode not in RETENTION_MODES:
        raise ValueError(f"retention_mode must be one of {RETENTION_MODES}")
    limits = limits or StreamLimits()
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not paths:
        raise ImportRefused("empty-selection", ["no source files were selected"])

    sources: list[_Source] = []
    for raw_path in paths:
        path = Path(raw_path)
        guard_source(path)
        state = ScanState(path=str(path))
        try:
            records, state = scan_jsonl(
                path, limits=limits, allow_partial=allow_partial, state=state
            )
        except InputLimitExceeded as exc:
            raise ImportRefused("limits", [str(exc)]) from exc
        except MalformedRecord as exc:
            raise ImportRefused("malformed", [str(exc)]) from exc

        sessions: list[Mapping[str, Any]] = [
            record
            for record in records
            if isinstance(record, Mapping) and isinstance(record.get("messages"), list)
        ]
        if not sessions:
            raise ImportRefused(
                "format",
                [
                    f"{path.name} does not look like a Hermes session export: no record carried a "
                    "`messages` array. Export with `hermes sessions export --format jsonl`."
                ],
            )
        sources.append(_Source(path=path, state=state, sessions=sessions, partial=state.partial))

    return _build_trace(sources, limits=limits, retention_mode=retention_mode, run_id=run_id)


def _iso(value: Any) -> str | None:
    """Epoch seconds (int/float) or an ISO-8601 string; anything else is None.

    The export stores message and session times as epoch floats. Trace v1 requires
    ISO-8601 with an offset, so a numeric time is converted to UTC and a value that
    cannot be read is dropped rather than approximated.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        moment = _dt.datetime.fromtimestamp(float(value), tz=_dt.UTC)
    except (OverflowError, OSError, ValueError):
        return None
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _tool_calls(raw: Any) -> list[Mapping[str, Any]]:
    """Parse a message's ``tool_calls`` into a list of call dicts.

    A real export serialises this field with ``str()``, producing single-quoted Python
    literals. JSON is tried first because that is the documented shape; ``literal_eval``
    covers the observed one. A value that is neither is an unsupported record, not a
    reason to invent calls.
    """
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, Mapping)]
    if not isinstance(raw, str) or not raw.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, Mapping)]
    return []


def _call_name(call: Mapping[str, Any]) -> str | None:
    """The reported tool name, from either the flat or the nested shape."""
    function = call.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        if isinstance(name, str) and name:
            return name
    name = call.get("name")
    return name if isinstance(name, str) and name else None


def _call_arguments(call: Mapping[str, Any]) -> Any:
    function = call.get("function")
    if isinstance(function, Mapping):
        return function.get("arguments")
    return call.get("arguments")


def _argument_value(raw: Any) -> Any:
    """Compare JSON arguments by value, not formatting; never evaluate code."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            pass
    return raw


def _operation(call: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Unwrap a recorded single dispatch, without guessing batch result ownership.

    Hermes returns the inner payload directly for a single ``tool_call`` dispatch.
    Multi-call batches retain their wrapper name and whole request reference: an
    aggregate failure cannot safely be attributed to every member. Arguments are
    redacted before computing the reference and are never retained themselves.
    """
    name = _call_name(call)
    arguments = _argument_value(_call_arguments(call))
    if name in {"tool_call", "functions.tool_call"} and isinstance(arguments, Mapping):
        calls = arguments.get("calls")
        if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], Mapping):
            inner = calls[0]
            if _call_name(inner):
                name = _call_name(inner)
                arguments = _argument_value(_call_arguments(inner))
    if arguments is None:
        return name, None
    redacted, _ = redact_record(arguments, stage="pre-store")
    return name, redacted_content_ref(redacted)


def _reported_outcome(record: Mapping[str, Any]) -> str:
    """Only a typed task outcome is a task outcome; end_reason is not one."""
    value = record.get("outcome")
    return value if isinstance(value, str) and value in OUTCOMES else "unknown"


def _call_id(call: Mapping[str, Any], fallback: str) -> str:
    for key in ("id", "call_id"):
        value = call.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def _recorded_call_ids(call: Mapping[str, Any]) -> set[str]:
    """Both source ID aliases are usable; synthesized display IDs are not."""
    return {
        value for key in ("id", "call_id")
        if isinstance((value := call.get(key)), str) and value
    }


def _payload(message: Mapping[str, Any]) -> Any:
    """A tool message's content as a parsed payload where possible."""
    raw = message.get("content")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            return None
    if isinstance(raw, list):
        parts = [
            str(item.get("text", ""))
            for item in raw
            if isinstance(item, Mapping) and item.get("text")
        ]
        if not parts:
            return None
        text = "\n".join(parts)
        try:
            return json.loads(text)
        except ValueError:
            return text
    return raw


def _tool_status(payload: Any) -> str:
    """Reported status from an explicit signal only; otherwise ``unknown``.

    A payload with no status field is **not** read as success: absence of an error in a
    tool result is not evidence that the call worked.
    """
    if not isinstance(payload, Mapping):
        return "unknown"
    reported = payload.get("status")
    if isinstance(reported, str) and reported.strip().lower() in _FAILURE_STATUS_VALUES:
        return "failed"
    if payload.get("error") or payload.get("success") is False:
        return "failed"
    code = payload.get("exit_code")
    if isinstance(code, int) and not isinstance(code, bool):
        return "ok" if code == 0 else "failed"
    if payload.get("success") is True:
        return "ok"
    if isinstance(reported, str) and reported in {"ok", "success", "completed"}:
        return "ok"
    if isinstance(reported, str) and reported in {"interrupted", "cancelled", "canceled"}:
        return "interrupted"
    return "unknown"


def _tool_summary(payload: Any, retention_mode: str) -> str | None:
    """The bounded, redacted-later summary kept under ``redacted-summary``."""
    if retention_mode != "redacted-summary":
        return None
    if isinstance(payload, Mapping):
        for key in ("error", "output", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:MAX_RESULT_SUMMARY]
        return None
    if isinstance(payload, str) and payload.strip():
        return payload.strip()[:MAX_RESULT_SUMMARY]
    return None


def _build_trace(
    sources: Sequence[_Source], *, limits: StreamLimits, retention_mode: str, run_id: str | None
) -> TraceImport:
    stats: dict[str, Any] = {
        "sessions": 0,
        "messages_read": 0,
        "model_calls": 0,
        "tool_calls_started": 0,
        "tool_results": 0,
        "human_messages": 0,
        "unsupported_roles": 0,
        "unparsed_tool_calls": 0,
        "unparsed_tool_results": 0,
        "records_in_sources": sum(len(source.sessions) for source in sources),
    }
    notes: list[str] = []
    warnings: list[str] = []
    partial = any(source.partial for source in sources)

    digest_input = "|".join(
        f"{source.path.name}:{source.state.digest if source.state else ''}" for source in sources
    )
    session_ids: list[str] = []
    for source in sources:
        for record in source.sessions:
            session_id = record.get("id")
            if isinstance(session_id, str) and session_id and session_id not in session_ids:
                session_ids.append(session_id)

    if run_id is None:
        run_id = f"run_hermes_{sha256_hex(digest_input + '|' + ','.join(session_ids))[:12]}"
    trace_id = stable_id("trace", run_id)
    root_attempt_id = stable_id("att", run_id, "primary")

    attempts: list[dict[str, Any]] = []
    attempt_by_session: dict[str, str] = {}
    session_records = [record for source in sources for record in source.sessions]
    for index, record in enumerate(session_records):
        session_id = str(record.get("id") or f"session-{index}")
        parent_session = record.get("parent_session_id")
        is_subagent = isinstance(parent_session, str) and bool(parent_session)
        attempt_id = root_attempt_id if index == 0 else stable_id("att", run_id, session_id)
        attempt_by_session[session_id] = attempt_id
        attempts.append({
            "kind": "attempt",
            "attempt_id": attempt_id,
            "run_id": run_id,
            "parent_attempt_id": None,
            "model": record.get("model") if isinstance(record.get("model"), str) else None,
            "provider": None,
            "role": "subagent" if is_subagent else "primary",
            "branch": str(record.get("source") or "main"),
            "sequence": index,
            "started_at": _iso(record.get("started_at")),
            "ended_at": _iso(record.get("ended_at")),
            # ``end_reason`` says why the process stopped, not how the task went.
            "outcome": _reported_outcome(record),
            "unknowns": ["outcome"] if _reported_outcome(record) == "unknown" else [],
        })
    # A child session whose parent was exported alongside it hangs off that attempt.
    for record in (record for source in sources for record in source.sessions):
        parent_session = record.get("parent_session_id")
        session_id = str(record.get("id") or "")
        if isinstance(parent_session, str) and parent_session in attempt_by_session:
            attempt_id = attempt_by_session.get(session_id)
            for attempt in attempts:
                if attempt["attempt_id"] == attempt_id:
                    attempt["parent_attempt_id"] = attempt_by_session[parent_session]

    events: list[dict[str, Any]] = []
    sequence = 0

    def emit(payload: dict[str, Any]) -> str:
        nonlocal sequence
        payload["source_sequence"] = sequence
        sequence += 1
        events.append(payload)
        return str(payload["event_id"])

    for source in sources:
        for record in source.sessions:
            session_id = str(record.get("id") or "")
            attempt_id = attempt_by_session.get(session_id, root_attempt_id)
            model = record.get("model") if isinstance(record.get("model"), str) else None
            messages = [m for m in record.get("messages", []) if isinstance(m, Mapping)]
            stats["sessions"] += 1
            stats["messages_read"] += len(messages)

            tool_event_by_call: dict[str, str] = {}
            operation_by_call: dict[str, tuple[str | None, str | None]] = {}
            span_by_call: dict[str, str] = {}
            ambiguous_calls: set[str] = set()
            seen_call_ids: set[str] = set()
            for message in messages:
                if message.get("role") != "assistant":
                    continue
                for call in _tool_calls(message.get("tool_calls")):
                    ids = _recorded_call_ids(call)
                    ambiguous_calls.update(ids & seen_call_ids)
                    seen_call_ids.update(ids)
            last_event: str | None = None

            for message_index, message in enumerate(messages):
                role = str(message.get("role") or "")
                message_id = str(message.get("id") or f"m{message_index}")
                wall = _iso(message.get("timestamp"))
                if role not in _KNOWN_ROLES:
                    stats["unsupported_roles"] += 1
                    continue

                if role == "assistant":
                    model_event = emit({
                        "kind": "event",
                        "event_id": stable_id(
                            "evt", run_id, attempt_id, "model_call", message_id, sequence
                        ),
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "span_id": None,
                        "causal_parents": [last_event] if last_event else [],
                        "wall_time": wall,
                        "monotonic_ms": None,
                        "clock": {"source": "wall", "quality": "medium"},
                        "event_kind": "model_call",
                        "reported_status": "ok",
                        "reported_error": None,
                        "usage": {},
                        "evidence_ref": f"hermes:{session_id}:message:{message_id}",
                        "delivered_via": "import",
                        "semantic_provenance": "reported",
                        "name": model,
                        "gaps": [],
                        "contradictions": [],
                    })
                    stats["model_calls"] += 1
                    last_event = model_event

                    calls = _tool_calls(message.get("tool_calls"))
                    if message.get("tool_calls") and not calls:
                        stats["unparsed_tool_calls"] += 1
                    for call_index, call in enumerate(calls):
                        call_id = _call_id(call, f"{message_id}-call{call_index}")
                        name, arguments_ref = _operation(call)
                        tool_event = emit({
                            "kind": "event",
                            "event_id": stable_id(
                                "evt", run_id, attempt_id, "tool_start", call_id, sequence
                            ),
                            "run_id": run_id,
                            "attempt_id": attempt_id,
                            "span_id": call_id,
                            "causal_parents": [model_event],
                            "wall_time": wall,
                            "monotonic_ms": None,
                            "clock": {"source": "wall", "quality": "medium"},
                            "event_kind": "tool_call",
                            "reported_status": "unknown",
                            "reported_error": None,
                            "usage": {},
                            "evidence_ref": f"hermes:{session_id}:tool_call:{call_id}",
                            "delivered_via": "import",
                            "semantic_provenance": "reported",
                            "name": name,
                            # The arguments themselves are never retained: a ref only.
                            "arguments_ref": arguments_ref,
                            "gaps": [],
                            "contradictions": [],
                        })
                        # A generated display span is not a recorded linkage key.
                        for recorded_id in _recorded_call_ids(call):
                            tool_event_by_call[recorded_id] = tool_event
                            operation_by_call[recorded_id] = (name, arguments_ref)
                            span_by_call[recorded_id] = call_id
                        last_event = tool_event
                        stats["tool_calls_started"] += 1
                    continue

                if role == "tool":
                    call_id = message.get("tool_call_id")
                    call_key = str(call_id) if isinstance(call_id, str) and call_id else ""
                    payload = _payload(message)
                    if payload is None:
                        stats["unparsed_tool_results"] += 1
                    parent = (
                        tool_event_by_call.get(call_key)
                        if call_key not in ambiguous_calls else None
                    )
                    operation = operation_by_call.get(call_key) if parent else None
                    event_id = emit({
                        "kind": "event",
                        "event_id": stable_id(
                            "evt", run_id, attempt_id, "tool_result",
                            call_key or message_id, sequence,
                        ),
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        # The source's own tool-call id links a result to its start.
                        "span_id": span_by_call[call_key] if parent else None,
                        "causal_parents": [parent] if parent else [],
                        "wall_time": wall,
                        "monotonic_ms": None,
                        "clock": {"source": "wall", "quality": "medium"},
                        "event_kind": "tool_call",
                        "reported_status": _tool_status(payload),
                        "reported_error": _tool_summary(payload, retention_mode),
                        "usage": {},
                        "evidence_ref": f"hermes:{session_id}:tool_result:{call_key or message_id}",
                        "delivered_via": "import",
                        "semantic_provenance": "reported",
                        "name": operation[0] if operation else (
                            message.get("tool_name")
                            if isinstance(message.get("tool_name"), str) else None
                        ),
                        "arguments_ref": operation[1] if operation else None,
                        "gaps": [] if parent else [
                            "tool result has no unique recorded call-id match; no causal parent inferred"
                        ],
                        "contradictions": [],
                    })
                    stats["tool_results"] += 1
                    last_event = event_id
                    continue

                # user turn: a reference, never the prompt text
                stats["human_messages"] += 1
                event_id = emit({
                    "kind": "event",
                    "event_id": stable_id(
                        "evt", run_id, attempt_id, "note", message_id, sequence
                    ),
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "span_id": None,
                    "causal_parents": [last_event] if last_event else [],
                    "wall_time": wall,
                    "monotonic_ms": None,
                    "clock": {"source": "wall", "quality": "medium"},
                    "event_kind": "note",
                    "reported_status": "ok",
                    "reported_error": None,
                    "usage": {},
                    "evidence_ref": f"hermes:{session_id}:message:{message_id}",
                    "delivered_via": "import",
                    "semantic_provenance": "reported",
                    "name": None,
                    "gaps": [],
                    "contradictions": [],
                })
                last_event = event_id

    if stats["unsupported_roles"]:
        notes.append(
            f"{stats['unsupported_roles']} message(s) used a role this adapter does not map "
            "(reported, not inferred)"
        )
    if stats["unparsed_tool_calls"]:
        notes.append(
            f"{stats['unparsed_tool_calls']} message(s) carried tool_calls that parsed as neither "
            "JSON nor a Python literal; no calls were invented for them"
        )
    if stats["unparsed_tool_results"]:
        notes.append(
            f"{stats['unparsed_tool_results']} tool result(s) did not parse as an object; their "
            "status stays unknown"
        )
    if retention_mode == "metadata-only":
        notes.append(
            "retention_mode=metadata-only: tool-result text was dropped at import, so a failure's "
            "mechanism may not be nameable from this evidence"
        )
    notes.append(
        "adapter semantics: prompt text and tool arguments are never retained (opaque references "
        "only); only an explicit typed outcome is a task outcome; end_reason is not one"
    )

    covered = sorted({str(event["event_kind"]) for event in events} & EVENT_KINDS)
    envelope = {
        "kind": "trace",
        "schema": "hermes-run-debugger/trace",
        "version": 1,
        "trace_id": trace_id,
        "run_id": run_id,
        "importer": "import_hermes",
        "importer_version": "0.1.1",
        "source_format": "hermes-session-jsonl",
        "source_scope": "explicit-selection",
        "created_at": _iso(sources[0].sessions[0].get("started_at")) if sources else None,
        "source": {
            "kind": "file",
            "ref": content_ref(digest_input),
            "name": sources[0].path.name if sources else "",
        },
        "redaction": {
            "policy": "minimal-metadata",
            "retention_mode": retention_mode,
            "applied_count": 0,
            "limitations": (
                ["tool-result text is not retained under metadata-only retention"]
                if retention_mode == "metadata-only" else []
            ),
        },
        "covered_kinds": covered or ["note"],
        "unsupported_kinds": [],
        "completeness": {
            "status": "partial" if partial else "complete",
            "missing_kinds": [],
            "notes": notes,
            "source_limits": {
                "max_bytes": limits.max_bytes,
                "max_events": limits.max_events,
            },
        },
        "private_digests": {
            "content": sha256_hex(digest_input),
            "files": [
                {"name": source.path.name, "sha256": source.state.digest if source.state else ""}
                for source in sources
            ],
        },
    }
    if envelope["created_at"] is None:
        envelope["created_at"] = "1970-01-01T00:00:00Z"

    records: list[dict[str, Any]] = [envelope, *attempts, *events]
    result = validate_trace_records(records)
    if partial:
        result.gaps.append(
            Gap("truncation", "at least one selected export ended mid-record; PARTIAL import")
        )
        warnings.append("import is PARTIAL: at least one source ended mid-record")

    snapshot = {
        "path": str(sources[0].path) if sources else "",
        "sha256": sha256_hex(digest_input),
        "records": stats["records_in_sources"],
        "partial": partial,
        "files": [
            {
                "path": str(source.path),
                "kind": "hermes-session-export",
                "records": len(source.sessions),
                "sha256": source.state.digest if source.state else "",
            }
            for source in sources
        ],
    }
    return TraceImport(
        format="hermes",
        result=result,
        snapshot=snapshot,
        partial=partial,
        warnings=warnings + list(result.warnings),
        notes=[sanitise_message(note) for note in notes],
        stats=stats,
        paths=[str(source.path) for source in sources],
    )

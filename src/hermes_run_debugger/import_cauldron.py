"""Import explicitly selected, retained Cauldron UAT JSON receipts.

Supported producers: UAT/release.py ReleaseReport and numbered adapter evidence,
UAT/golden.py verdict/score, release deploy checks and supervised_execution.py.
No path embedded in a receipt is followed. Selection order is not causality;
receipts are independent attempts, even when their reported run labels match.

A completed harness/command is not a product PASS. Product verdicts and scores
are retained as REPORTED note subjects and summaries, never correctness claims.
Arbitrary JSON, configuration, transcripts and historical alternate shapes are
refused. A partial JSON object has no trustworthy record boundary: allow_partial
never salvages one. Well-formed receipts can themselves report partial execution.
"""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .event_v1 import Gap, TraceValidationError, validate_trace_records
from .identity import canonical_json, content_ref, redacted_content_ref, sha256_hex, stable_id
from .import_v1 import ImportRefused, TraceImport, guard_source
from .jsonio import CHUNK_BYTES, InputError, ScanState, StreamLimits, loads_strict
from .redact import minimal_policy, sanitise_message, scrub_guard_payload

__all__ = ["import_cauldron_trace"]

_RELEASE = "cauldron-uat-release/v1"
_EXECUTION = "cauldron-uat-execution/v1"
_VERDICTS = {"PASS", "MAIN FAIL", "UAT SETUP FAIL"}
_COMPLETIONS = {"done", "timeout", "stopped", "interrupted", "preserved-regrade"}
_NUMBERED = re.compile(r"[0-9]{2}-([A-Za-z0-9_-]+)\.json\Z")
_SCORE = re.compile(r"[0-9]+/[0-9]+\Z")


def _refuse(kind: str, message: str) -> None:
    raise ImportRefused(kind, [message])


def _text(value: Any, *, maximum: int = 500) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _refuse("shape", "receipt text field must be a string or null")
    # Redact BEFORE bounding: truncation must not split a recognised secret.
    return sanitise_message(value)[:maximum]


def _label(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        _refuse("shape", "stage/step label must be a nonempty string of at most 128 characters")
    return sanitise_message(value)[:128]


def _boolean(record: dict[str, Any], key: str) -> None:
    if key in record and record[key] is not None and type(record[key]) is not bool:
        _refuse("shape", "receipt boolean field has an unsupported type")


def _member(value: Any, allowed: set[str]) -> bool:
    return isinstance(value, str) and value in allowed


def _outcome(record: dict[str, Any], kind: str) -> str:
    if kind == "execution":
        return {"completed": "completed", "launch_failed": "failed"}.get(record["state"], "unknown")
    if record.get("manual_required") or record.get("execution_uncertain"):
        return "unknown"
    if record.get("stage") == "complete" or record.get("completion") == "done":
        return "completed"
    if record.get("completion") == "interrupted":
        return "interrupted"
    # A product failure, low score or ok=False does NOT establish execution failure.
    if kind == "release" and record.get("failure"):
        return "failed"
    return "unknown"


def _read(path: Path, limits: StreamLimits, remaining: int) -> tuple[dict[str, Any], ScanState]:
    guard_source(path)
    if path.suffix.lower() != ".json":
        _refuse("format", "Cauldron receipt import accepts selected JSON files only")
    state = ScanState(path=str(path))
    state.begin()
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                _refuse("format", "selected receipt must be a regular file")
            cap = min(limits.max_bytes, remaining)
            if before.st_size > cap:
                _refuse("limits", "selected receipt bytes exceed the import byte budget")
            chunks: list[bytes] = []
            line_bytes = 0
            while True:
                chunk = stream.read(min(CHUNK_BYTES, cap - state.bytes_read + 1))
                if not chunk:
                    break
                state.feed(chunk)
                if state.bytes_read > cap:
                    _refuse("limits", "selected receipt bytes exceed the import byte budget")
                for index, part in enumerate(chunk.split(b"\n")):
                    line_bytes = line_bytes + len(part) if index == 0 else len(part)
                    if line_bytes > limits.max_line_bytes:
                        _refuse("limits", "receipt line exceeds max_line_bytes")
                chunks.append(chunk)
            after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            _refuse("changed-source", "selected receipt changed while being read; select a stable retained copy")
        raw = b"".join(chunks)
        state.size, state.mtime_ns = before.st_size, before.st_mtime_ns
        state.sha256 = state.digest
        value = loads_strict(raw.decode("utf-8"), max_depth=limits.max_depth, where="selected receipt")
    except ImportRefused:
        raise
    except (OSError, UnicodeError, InputError, RecursionError):
        _refuse("malformed", "selected receipt is unreadable, malformed, non-finite or too deeply nested; "
                "no payload retained")
    if not isinstance(value, dict):
        _refuse("shape", "selected receipt must contain one JSON object")
    # JSON's exponent notation can overflow even without NaN/Infinity constants.
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, float) and not math.isfinite(item):
            _refuse("malformed", "non-finite receipt number is refused")
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    state.records = 1
    return value, state


def _shape(path: Path, record: dict[str, Any]) -> str:
    schema = record.get("schema")
    for key in ("ok", "scored", "partial", "cleanup_ok", "manual_required", "execution_uncertain"):
        _boolean(record, key)
    if schema == _RELEASE:
        if type(record.get("ok")) is not bool or not isinstance(record.get("stages"), list):
            _refuse("shape", "release receipt requires boolean ok and a structured stages list")
        if not isinstance(record.get("release_commit"), str):
            _refuse("shape", "release receipt requires its reported release_commit")
        for stage in record["stages"]:
            _label(stage)
        return "release"
    if schema == _EXECUTION:
        if not _member(record.get("state"), {"launch_intent", "running", "completed", "uncertain", "launch_failed"}):
            _refuse("shape", "unsupported execution receipt state")
        _label(record.get("label"))
        if record["state"] == "completed" and type(record.get("exit")) is not int:
            _refuse("shape", "completed execution receipt requires an integer exit")
        return "execution"
    # Numbered adapter receipts have their own maintained schema (e.g. clone).
    adapter_schemas = {
        "cauldron-uat-clone/v1", "cauldron-uat-clone-readback/v1", "cauldron-uat-operational/v1",
    }
    if schema is not None and (schema not in adapter_schemas or not _NUMBERED.fullmatch(path.name)):
        _refuse("format", "unsupported Cauldron receipt schema/version")
    if path.name in {"verdict.json", "golden-verdict.json"}:
        if not _member(record.get("verdict"), _VERDICTS) or not isinstance(record.get("run_id"), str):
            _refuse("shape", "golden verdict requires a known verdict and reported run_id")
        if not _member(record.get("completion"), _COMPLETIONS) or not isinstance(record.get("failures"), list):
            _refuse("shape", "golden verdict requires structured completion and failures")
        return "verdict"
    if path.name == "score.json":
        if (not _member(record.get("verdict"), _VERDICTS) or not isinstance(record.get("points"), list)
                or type(record.get("passed")) is not int or type(record.get("total")) is not int):
            _refuse("shape", "score receipt requires explicit verdict, passed, total and structured points")
        return "score"
    if path.name.startswith("deploy-checks-") and path.suffix == ".json":
        if (type(record.get("ok")) is not bool or not isinstance(record.get("checks"), list)
                or not isinstance(record.get("failed"), list)):
            _refuse("shape", "deploy-check receipt requires boolean ok and structured checks/failed lists")
        return "checks"
    if _NUMBERED.fullmatch(path.name) and type(record.get("ok")) is bool:
        return "stage"
    _refuse("format", "unsupported receipt shape; select a release report, golden verdict/score, "
            "deploy checks, execution receipt or numbered boolean-ok adapter evidence")
    return ""


def import_cauldron_trace(
    paths: str | Path | Sequence[str | Path],
    *,
    limits: StreamLimits | None = None,
    allow_partial: bool = False,
    retention_mode: str = "redacted-summary",
) -> TraceImport:
    """Return TraceImport(format='cauldron') from selected retained JSON files.

    Bounds apply to the entire selection as well as each file; max_events bounds
    normalized records (envelope + attempts + events). Malformed objects are
    refused even with allow_partial, never scraped for a plausible valid prefix.
    Unknown fields/payloads are omitted deliberately; coverage always stays unknown.
    """
    if retention_mode not in {"redacted-summary", "metadata-only"}:
        raise ValueError("retention_mode must be redacted-summary or metadata-only")
    limits = limits or StreamLimits()
    selection = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not selection:
        _refuse("empty-selection", "no receipt files selected")
    if len(selection) > limits.max_events:
        _refuse("limits", "selected receipt count exceeds max_events")
    sources: list[tuple[Path, dict[str, Any], ScanState, str]] = []
    seen: set[str] = set()
    total_bytes = 0
    for raw_path in selection:
        path = Path(raw_path)
        key = os.path.normcase(str(path.resolve()))
        if key in seen:
            _refuse("duplicate-selection", "the same receipt file was selected more than once")
        seen.add(key)
        record, state = _read(path, limits, limits.max_bytes - total_bytes)
        total_bytes += state.bytes_read
        sources.append((path, record, state, _shape(path, record)))
    digest = sha256_hex(canonical_json([state.digest for _, _, state, _ in sources]))
    run_id = stable_id("run_cauldron", digest)
    attempts: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    partial = False

    def emit(attempt_id: str, ref: str, name: str, *, status: str = "unknown",
             error: Any = None, kind: str = "session_lifecycle") -> None:
        if 1 + len(attempts) + len(events) >= limits.max_events:
            _refuse("limits", "normalized receipt records exceed max_events")
        events.append({
            "kind": "event", "event_id": stable_id("evt", run_id, len(events)),
            "run_id": run_id, "attempt_id": attempt_id, "source_sequence": len(events),
            "event_kind": kind, "name": sanitise_message(name)[:128],
            "reported_status": status,
            "reported_error": _text(error) if retention_mode == "redacted-summary" else None,
            "evidence_ref": ref, "semantic_provenance": "reported", "delivered_via": "import",
        })

    for index, (path, record, _state, shape) in enumerate(sources):
        attempt_id = stable_id("att", run_id, index)
        outcome = _outcome(record, shape)
        attempts.append({"kind": "attempt", "attempt_id": attempt_id, "run_id": run_id,
                         "outcome": outcome, "sequence": index, "role": "harness",
                         "unknowns": ["causal_relationships", "agent_actions"]})
        # Only redacted metadata enters exported references, never raw source hashes.
        ref = redacted_content_ref({"selection_index": index, "receipt_kind": shape})
        summary: dict[str, Any] = {"receipt_kind": shape, "evidence_ref": ref,
                                  "execution_outcome": outcome, "semantic_provenance": "reported"}
        for key in ("scored", "cleanup_ok", "ok", "manual_required", "execution_uncertain", "partial"):
            if key in record:
                summary[key] = record[key]
        for key in ("completion", "stage", "stopped_at_step"):
            if key in record:
                summary[key] = _text(record[key], maximum=128)
        if record.get("verdict") is not None:
            if not _member(record["verdict"], _VERDICTS):
                _refuse("shape", "unsupported reported product verdict")
            summary["product_verdict"] = record["verdict"]
            emit(attempt_id, ref + ":verdict", "product verdict: " + record["verdict"], kind="note")
        if record.get("score") is not None:
            if not isinstance(record["score"], str) or not _SCORE.fullmatch(record["score"]):
                _refuse("shape", "unsupported reported score representation")
            summary["score"] = record["score"]
            emit(attempt_id, ref + ":score", "reported score: " + record["score"], kind="note")
        partial |= record.get("partial") is True or _member(
            record.get("completion"), {"timeout", "stopped", "interrupted"})
        if shape == "release":
            for stage_index, stage in enumerate(record["stages"]):
                emit(attempt_id, ref + f":stages:{stage_index}", "stage recorded: " + _label(stage))
            if record.get("stage") is not None or record.get("failure") is not None:
                failure = _text(record.get("failure"))
                reason = _text(record.get("reason"))
                error = ": ".join(part for part in (failure, reason) if part) if failure else None
                emit(attempt_id, ref + ":stage", "stage: " + _label(record.get("stage", "unknown")),
                     status="failed" if failure else "unknown", error=error)
            cleanup = record.get("cleanup_errors", [])
            if not isinstance(cleanup, list):
                _refuse("shape", "cleanup_errors must be structured records")
            for cleanup_index, item in enumerate(cleanup):
                if not isinstance(item, dict):
                    _refuse("shape", "cleanup error must be an object")
                emit(attempt_id, ref + f":cleanup_errors:{cleanup_index}", "cleanup: " + _label(item.get("stage")),
                     status="failed", error=item.get("failure"))
        elif shape == "verdict":
            for failure_index, failure in enumerate(record["failures"]):
                if not isinstance(failure, dict) or not isinstance(failure.get("error"), str):
                    _refuse("shape", "golden failures require structured step/class/error records")
                step, failure_class = _label(failure.get("step")), _label(failure.get("class"))
                emit(attempt_id, ref + f":failures:{failure_index}", f"{failure_class}: {step}",
                     status="failed", error=failure["error"])
        elif shape == "execution":
            summary["state"] = record["state"]
            code = record.get("exit")
            if code is not None and type(code) is not int:
                _refuse("shape", "execution exit must be an integer")
            summary["exit"] = code
            status = "unknown" if code is None else "ok" if code == 0 else "failed"
            emit(attempt_id, ref + ":execution", "command: " + _label(record["label"]),
                 status=status, error=record.get("failure"))
        elif shape == "checks":
            for check_index, check in enumerate(record["checks"]):
                if not isinstance(check, dict) or not _member(check.get("result"), {"PASS", "FAIL"}):
                    _refuse("shape", "deploy checks require structured check/result records")
                emit(attempt_id, ref + f":checks:{check_index}", "check: " + _label(check.get("check")),
                     status="ok" if check["result"] == "PASS" else "failed",
                     error=check.get("reason") if check["result"] == "FAIL" else None)
        elif shape == "score":
            summary.update(passed=record["passed"], total=record["total"])
            for point_index, point in enumerate(record["points"]):
                if not isinstance(point, dict):
                    _refuse("shape", "score points must be structured records")
                label, status = _label(point.get("id")), _label(point.get("status"))
                emit(attempt_id, ref + f":points:{point_index}", f"point {label}: {status}", kind="note")
        else:
            stage = _NUMBERED.fullmatch(path.name)
            assert stage is not None
            emit(attempt_id, ref + ":stage", "stage: " + _label(stage.group(1)),
                 status="ok" if record["ok"] else "failed", error=record.get("reason"))
        emit(attempt_id, ref + ":lifecycle", "execution outcome: " + outcome)
        summaries.append(summary)

    notes = [
        "Only selected structured receipts were imported; embedded paths were not followed. "
        "Agent actions, capture windows and cross-receipt causality are unknown.",
        "Reported verdict/score is distinct from lifecycle completion; no correctness claim is minted. "
        "Release stages record reached/returned commands, not successful qualification.",
        "Unmapped payload fields, command arguments, output and tracebacks were not retained.",
        "Receipt timestamps are not projected; created_at uses an epoch sentinel, not an observed start time.",
        *["reported receipt: " + canonical_json(summary) for summary in summaries],
    ]
    if allow_partial:
        notes.append("allow_partial does not salvage malformed JSON receipt objects")
    envelope = {
        "kind": "trace", "schema": "hermes-run-debugger/trace", "version": 1,
        "trace_id": stable_id("trace", run_id), "run_id": run_id,
        "importer": "import_cauldron", "importer_version": "0.1.0",
        "source_format": "cauldron-uat-receipts-json", "source_scope": "explicit-selection",
        "created_at": "1970-01-01T00:00:00Z",
        "source": {"kind": "file", "ref": content_ref(digest), "name": "selected Cauldron receipts"},
        "redaction": {"policy": "minimal-metadata", "retention_mode": retention_mode,
                      "limitations": minimal_policy().limitations()},
        "covered_kinds": sorted({event["event_kind"] for event in events}), "unsupported_kinds": [],
        "completeness": {"status": "partial" if partial else "unknown", "notes": notes,
                         "source_limits": {"max_bytes": limits.max_bytes, "max_events": limits.max_events,
                                           "max_line_bytes": limits.max_line_bytes, "max_depth": limits.max_depth}},
        "private_digests": {"content": digest, "files": [state.snapshot() for _, _, state, _ in sources]},
    }
    try:
        scrub_guard_payload({"notes": notes, "events": events})
        result = validate_trace_records([envelope, *attempts, *events], max_records=limits.max_events)
    except TraceValidationError:
        _refuse("schema", "normalized receipt evidence did not satisfy Trace v1")
    result.gaps.append(Gap("unknown_capture_window",
                           "Selected receipts do not establish complete agent-action coverage"))
    if partial:
        result.gaps.append(Gap("partial_source", "Selected receipt explicitly reports partial execution"))
    return TraceImport(
        format="cauldron", result=result,
        snapshot={"sha256": digest, "size": total_bytes, "records": len(sources), "partial": partial,
                  "files": [state.snapshot() for _, _, state, _ in sources]},
        partial=partial, warnings=list(result.warnings), notes=notes,
        stats={"receipts": len(sources), "events": len(events), "reported_receipts": summaries},
        paths=[str(path) for path, _, _, _ in sources],
    )

"""Harness-neutral Trace v1 import, plus the source-shape guards.

This module owns three things:

1. **What counts as a supported source.** JSONL only in v1. Archives, archives
   with traversal/bomb shapes, and raw databases are refused *before* any
   extraction code could run — and the refusal states the bar a future adapter
   must clear (member count, size, ratio, paths, symlinks, hardlinks).
2. **Normalisation.** Parse with bounded IO, then validate against the strict
   Trace v1 contract. A refusal names its reason; nothing is "best-effort"
   repaired.
3. **Snapshot identity.** The digest of the bytes actually read, so a retry can
   never blend a grown file's new suffix into earlier counts.

Format detection never guesses: a source whose shape is not recognised is refused
with the reason, not parsed as something it might be.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .event_v1 import SCHEMA_ID, TraceValidationError, ValidationResult, validate_trace_records
from .identity import content_ref
from .jsonio import (
    InputError,
    InputLimitExceeded,
    MalformedRecord,
    ScanState,
    StreamLimits,
    scan_jsonl,
)
from .redact import sanitise_message

__all__ = [
    "SUPPORTED_FORMATS",
    "ImportRefused",
    "TraceImport",
    "detect_format",
    "import_trace",
    "limits_from_settings",
]

SUPPORTED_FORMATS = ("v1", "claude", "hermes", "cauldron")

ARCHIVE_SUFFIXES = (
    ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".gz", ".bz2",
    ".xz", ".7z", ".rar", ".zst",
)
DATABASE_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".parquet", ".arrow", ".pkl", ".pickle")

_ARCHIVE_REFUSAL = (
    "unsupported source: archive input ({path}) — this build imports JSONL only and never extracts "
    "an archive. A future archive adapter must first validate member count, member size, "
    "compression ratio, member paths (no absolute paths, no traversal), symlinks and hardlinks, "
    "and a refusal must happen before any extraction."
)
_DATABASE_REFUSAL = (
    "unsupported source: raw database/file input ({path}) — this build imports JSONL only. A live "
    "session read or a dedicated export adapter is a separate, separately gated lane."
)
_UNKNOWN_REFUSAL = (
    "cannot determine the source format of {path}: neither an event-v1 trace envelope nor a "
    "recognised foreign transcript shape was found. Nothing was imported."
)


class ImportRefused(InputError):
    """The source was refused. ``kind`` names why; ``reasons`` are the details.

    Reasons are sanitised before they leave this object: a refusal must never become a
    vector for republishing the value that caused it — in a terminal, a slash reply, a
    tool result, a caller's log or an exception cause chain.
    """

    def __init__(self, kind: str, reasons: Sequence[str]):
        self.kind = kind
        self.reasons = [sanitise_message(str(reason)) for reason in reasons]
        InputError.__init__(self, "; ".join(self.reasons))


@dataclass
class TraceImport:
    """One imported source, with the evidence needed to trust or reject a claim."""

    format: str
    result: ValidationResult
    snapshot: dict[str, Any]
    partial: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)

    @property
    def may_claim_complete(self) -> bool:
        if self.partial:
            return False
        status = str((self.result.trace.completeness or {}).get("status", "unknown")) \
            if self.result.trace else "unknown"
        if status != "complete":
            return False
        blocking = {"missing_parent", "partial_source", "truncation", "unsupported_kind"}
        return not any(gap.kind in blocking for gap in self.result.gaps)

    def summary(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "run_id": self.result.trace.run_id if self.result.trace else None,
            "events": len(self.result.events),
            "attempts": len(self.result.attempts),
            "gaps": len(self.result.gaps),
            "partial": self.partial,
            "may_claim_complete": self.may_claim_complete,
        }


def limits_from_settings(settings: Mapping[str, Any] | None = None) -> StreamLimits:
    """Build stream limits from plugin settings, ignoring unknown keys."""
    settings = dict(settings or {})
    return StreamLimits(
        max_bytes=int(settings.get("import_max_bytes", 64 * 1024 * 1024)),
        max_line_bytes=int(settings.get("import_max_line_bytes", 8 * 1024 * 1024)),
        max_events=int(settings.get("import_max_events", 200_000)),
        max_depth=int(settings.get("import_max_depth", 24)),
    )


def _suffix_of(path: Path) -> str:
    name = path.name.lower()
    for suffix in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    for suffix in DATABASE_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return ""


def guard_source(path: Path) -> None:
    """Refuse archives/databases and anything that is not readable text JSONL."""
    if path.is_dir():
        raise ImportRefused(
            "directory",
            [
                f"unsupported source: {path} is a directory. v1 imports explicitly selected FILES; "
                "directory scanning is not performed."
            ],
        )
    if not path.exists():
        raise ImportRefused("missing", [f"source does not exist: {path}"])
    suffix = _suffix_of(path)
    if suffix and suffix in ARCHIVE_SUFFIXES:
        raise ImportRefused("archive", [_ARCHIVE_REFUSAL.format(path=path)])
    if suffix and suffix in DATABASE_SUFFIXES:
        raise ImportRefused("database", [_DATABASE_REFUSAL.format(path=path)])


def detect_format(path: str | Path, explicit: str | None = None) -> str:
    """Return the source format (``v1``, ``claude`` or ``hermes``); refuse anything unrecognised."""
    p = Path(path)
    guard_source(p)
    if explicit:
        wanted = explicit.strip().lower()
        if wanted not in SUPPORTED_FORMATS:
            raise ImportRefused(
                "format", [f"unsupported format {explicit!r}; supported: {', '.join(SUPPORTED_FORMATS)}"]
            )
        return wanted

    sample = _first_records(p, limit=3)
    if not sample:
        raise ImportRefused("empty", [f"cannot determine the source format of {p}: the file is empty"])
    head = sample[0]
    if isinstance(head, Mapping):
        if head.get("schema") == SCHEMA_ID or head.get("kind") == "trace":
            return "v1"
        # A Hermes session export is a session envelope: an id, a model and a
        # ``messages`` array. Checked before the Claude shape because both carry a
        # ``type`` key, and this one is the stronger signature.
        if isinstance(head.get("messages"), list) and isinstance(head.get("id"), str):
            return "hermes"
        record_type = head.get("type")
        if record_type in {"assistant", "user", "system", "summary"}:
            return "claude"
    raise ImportRefused("format", [_UNKNOWN_REFUSAL.format(path=p)])


def _first_records(
    path: Path,
    *,
    limit: int,
    max_line_bytes: int = 8 * 1024 * 1024,
    max_bytes: int = 512 * 1024,
) -> list[Any]:
    """Best-effort peek at the first records. Bounded, and never raises on content.

    A record is read whole even when it is larger than the total sample budget: a Hermes
    session export is one very large JSON line per session, so a budget that stopped
    before the first line was parsed would report a valid export as empty. One record
    over ``max_line_bytes`` (the same bound the scan applies) ends the peek instead,
    so the read stays bounded.
    """
    records: list[Any] = []
    total = 0
    try:
        with path.open("rb") as handle:
            for raw in handle:
                if len(raw) > max_line_bytes:
                    break
                total += len(raw)
                text = raw.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    records.append(json.loads(text))
                except ValueError:
                    records.append(text)
                if len(records) >= limit or total >= max_bytes:
                    break
    except OSError as exc:
        raise ImportRefused("unreadable", [f"cannot read {path}: {exc}"]) from exc
    return records


def import_trace(
    path: str | Path,
    *,
    fmt: str | None = None,
    limits: StreamLimits | None = None,
    allow_partial: bool = False,
    unknown_field_policy: str = "reject",
    retention_mode: str = "redacted-summary",
) -> TraceImport:
    """Import one explicitly selected source as Trace v1 evidence."""
    p = Path(path)
    limits = limits or StreamLimits()
    guard_source(p)
    resolved = detect_format(p, explicit=fmt)
    if resolved == "cauldron":
        from .import_cauldron import import_cauldron_trace

        return import_cauldron_trace(
            [p], limits=limits, allow_partial=allow_partial, retention_mode=retention_mode
        )
    if resolved == "claude":
        from .import_claude import import_claude_trace

        return import_claude_trace(
            [p], limits=limits, allow_partial=allow_partial, retention_mode=retention_mode
        )
    if resolved == "hermes":
        from .import_hermes import import_hermes_trace

        return import_hermes_trace(
            [p], limits=limits, allow_partial=allow_partial, retention_mode=retention_mode
        )

    state = ScanState(path=str(p))
    try:
        records, state = scan_jsonl(
            p, limits=limits, allow_partial=allow_partial, state=state
        )
    except InputLimitExceeded as exc:
        raise ImportRefused("limits", [str(exc)]) from exc
    except MalformedRecord as exc:
        raise ImportRefused("malformed", [str(exc)]) from exc

    notes: list[str] = []
    digest = state.digest
    if records and isinstance(records[0], Mapping):
        source = records[0].get("source")
        if isinstance(source, Mapping) and not source.get("ref"):
            source["ref"] = content_ref(digest)
            notes.append("source digest computed from the selected file (the trace did not carry one)")
        if isinstance(records[0], dict):
            private = records[0].setdefault("private_digests", {})
            if isinstance(private, dict):
                private.setdefault("content", digest)
                private.setdefault("size", state.size)

    try:
        result = validate_trace_records(
            records,
            max_records=limits.max_events,
            unknown_field_policy=unknown_field_policy,
        )
    except TraceValidationError as exc:
        raise ImportRefused("schema", [str(exc)]) from exc

    if state.partial:
        from .event_v1 import Gap

        result.gaps.append(
            Gap(
                "truncation",
                f"final line {state.truncated_line_no} was truncated; only the valid prefix was kept",
            )
        )
        result.warnings.append("import is PARTIAL: the source ended mid-record")
        notes.append("PARTIAL import: valid prefix retained, no completeness claim")

    return TraceImport(
        format="v1",
        result=result,
        snapshot=state.snapshot(),
        partial=state.partial,
        warnings=list(result.warnings),
        notes=notes,
        stats={"records": state.records, "lines": state.lines, "blank_lines": state.skipped_blank},
        paths=[str(p)],
    )

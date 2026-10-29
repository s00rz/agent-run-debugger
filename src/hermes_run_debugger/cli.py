"""Services and surfaces: one analysis path behind CLI, slash command and tools.

Every surface calls the same services, so the CLI cannot do something the agent
tools cannot — with exactly one deliberate exception: **export is operator-CLI
only**. There is no export tool, no slash export and no gateway export route. Those
routes refuse, write nothing and upload nothing, because viewing a report locally
is not permission to share it.

The services layer is host-neutral: no Hermes import appears here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .diagnose import diagnose
from .import_v1 import ImportRefused, import_trace
from .jsonio import InputLimitExceeded
from .redact import ScrubFailure, sanitise_message
from .replay import replay_events
from .settings import Settings
from .store import RunDebugStore
from .timeline import TimelineContradiction, build_timeline

__all__ = [
    "EXPORT_REFUSAL",
    "Services",
    "handle_slash",
    "run_cli",
    "tool_explain",
    "tool_import",
    "tool_replay",
    "tool_status",
    "tool_timeline",
]

EXPORT_REFUSAL = (
    "refused: export is operator-CLI-only in this build. Run "
    "`hermes run-debug export --run <run-id> --out <private-path>` yourself. This route wrote no "
    "file and has no upload path."
)

_USAGE = (
    "usage: /rundebug import <file> [--format v1|claude|hermes] | timeline <run-id> | explain <run-id> | "
    "replay <run-id> | status"
)


@dataclass
class Services:
    """Analysis services over one profile-scoped store.

    ``data_dir`` may be a path or a resolver callable. The host adapter passes a
    resolver so the plugin data directory is (re)resolved against the *active*
    profile on every operation, and nothing is created at registration time.
    """

    data_dir: Path | Callable[[], Path]
    settings: Settings = field(default_factory=Settings)

    def __post_init__(self) -> None:
        self._store: RunDebugStore | None = None

    def resolved_data_dir(self) -> Path:
        target = self.data_dir() if callable(self.data_dir) else self.data_dir
        return Path(target)

    # -- lifecycle ---------------------------------------------------------
    def store(self) -> RunDebugStore:
        if self._store is None:
            self._store = RunDebugStore(
                self.resolved_data_dir(), policy=self.settings.policy()
            ).open()
            self._store.recover_incomplete_imports()
        return self._store

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def __enter__(self) -> Services:
        self.store()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _refuse_lanes(self) -> None:
        refusals = self.settings.refusals()
        if refusals:
            raise ImportRefused("settings", refusals)

    # -- lanes -------------------------------------------------------------
    def import_sources(
        self,
        paths: Sequence[str | Path],
        *,
        fmt: str | None = None,
        allow_partial: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Import explicitly selected sources and store them."""
        self._refuse_lanes()
        if not paths:
            raise ImportRefused("empty-selection", ["no source file was selected"])
        partial = self.settings.allow_partial if allow_partial is None else allow_partial

        try:
            return self._import_sources(paths, fmt=fmt, partial=partial)
        except ScrubFailure as exc:
            # A privacy refusal is a first-class refusal, not a crash: the surfaces
            # report it with a named kind and nothing was persisted.
            raise ImportRefused("privacy", [str(exc)]) from exc

    def _import_sources(
        self, paths: Sequence[str | Path], *, fmt: str | None, partial: bool
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        if fmt == "cauldron":
            from .import_cauldron import import_cauldron_trace

            combined = import_cauldron_trace(
                paths, limits=self.settings.limits(), allow_partial=partial,
                retention_mode=self.settings.retention_mode,
            )
            imported_list = [combined]
        elif fmt == "hermes":
            from .import_hermes import import_hermes_trace

            combined = import_hermes_trace(
                paths,
                limits=self.settings.limits(),
                allow_partial=partial,
                retention_mode=self.settings.retention_mode,
            )
            imported_list = [combined]
        elif fmt == "claude" or (fmt is None and len(paths) > 1):
            from .import_claude import import_claude_trace

            combined = import_claude_trace(
                paths,
                limits=self.settings.limits(),
                allow_partial=partial,
                retention_mode=self.settings.retention_mode,
            )
            imported_list = [combined]
        else:
            imported_list = [
                import_trace(
                    path,
                    fmt=fmt,
                    limits=self.settings.limits(),
                    allow_partial=partial,
                    retention_mode=self.settings.retention_mode,
                )
                for path in paths
            ]

        for imported in imported_list:
            outcome = self.store().import_trace(
                imported.result,
                source_summary={
                    "path": str(imported.paths[0] if imported.paths else ""),
                    "sha256": imported.snapshot.get("sha256", ""),
                },
                expected_records=imported.snapshot.get("records"),
            )
            summary = imported.summary()
            summary.update({
                "revision": outcome.revision,
                "deduplicated": outcome.deduplicated,
                "redaction_applied": outcome.redaction_applied,
                "notes": imported.notes,
                "warnings": imported.warnings,
                "stats": imported.stats,
            })
            results.append(summary)
        return results

    def load_timeline(self, run_id: str):
        run = self.store().load_run(run_id)
        return build_timeline(_as_result(run))

    def timeline_text(self, run_id: str, *, event_ref: str | None = None) -> str:
        timeline = self.load_timeline(run_id)
        if event_ref is None:
            return timeline.text()
        selected = timeline.entry(event_ref)
        if selected is None:
            raise ValueError(f"event {event_ref!r} is not in run {run_id!r}")
        event = selected.event
        related = [entry.event for entry in timeline.entries if (
            entry.event_id == event_ref
            or entry.event_id in event.causal_parents
            or event_ref in entry.causal_parents
            or (event.span_id is not None and entry.event.span_id == event.span_id
                and entry.event.attempt_id == event.attempt_id)
            or (event.arguments_ref is not None and entry.event.arguments_ref == event.arguments_ref
                and entry.event.name == event.name and entry.event.attempt_id == event.attempt_id)
        )]
        return json.dumps({
            "run_id": run_id, "selected_event": event_ref,
            "relationship": "recorded parents/children, same call span, or same request in the same attempt",
            "events": [item.as_dict() for item in related],
            "notice": "same request is not proof of the same durable outcome; source references are pointers",
        }, indent=2, ensure_ascii=False) + "\n"

    def explain_text(self, run_id: str, *, as_json: bool = False) -> str:
        timeline = self.load_timeline(run_id)
        diagnosis = diagnose(timeline)
        if as_json:
            return json.dumps(diagnosis.as_dict(), indent=2, ensure_ascii=False) + "\n"
        return diagnosis.text()

    def replay_text(self, run_id: str) -> str:
        run = self.store().load_run(run_id)
        log = replay_events(
            run.events,
            run_id=run.run_id,
            gaps=run.gaps,
            ambiguous_pairs=build_timeline(_as_result(run)).ambiguous_pairs,
        )
        return log.text()

    def export_report(self, run_id: str, out: str | Path, *, fmt: str = "markdown") -> Path:
        """Private export. Operator CLI only; never reachable from a tool or slash route."""
        self._refuse_lanes()
        return self.store().export_run(run_id, out, fmt=fmt)

    def status(self) -> dict[str, Any]:
        store = self.store()
        runs = store.list_runs()
        return {
            "schema": "hermes-run-debugger/status",
            "version": __version__,
            "data_dir": str(self.resolved_data_dir()),
            "store_schema_version": store.schema_version(),
            "runs": len(runs),
            "latest_runs": [row["run_id"] for row in runs[-5:]],
            "lanes": self.settings.lane_status(),
            "refusals": self.settings.refusals(),
            "limits": {
                "import_max_bytes": self.settings.import_max_bytes,
                "import_max_line_bytes": self.settings.import_max_line_bytes,
                "import_max_events": self.settings.import_max_events,
                "import_max_depth": self.settings.import_max_depth,
                "allow_partial": self.settings.allow_partial,
                "retention_days": self.settings.retention_days,
                "retention_mode": self.settings.retention_mode,
            },
            "honesty": [
                "imported logs are reported data; diagnosis is inferred and never a causal finding",
                "replay executes nothing",
                "unknown billing stays null",
                "export has no tool, slash or gateway route",
            ],
        }


def _as_result(run: Any) -> Any:
    """Wrap stored rows back into the ValidationResult shape the analysers take."""
    from .event_v1 import Gap, ValidationResult

    trace = _trace_like(run)
    return ValidationResult(
        trace=trace,
        attempts=[_attempt_like(attempt) for attempt in run.attempts],
        events=list(run.events),
        gaps=[Gap(
            kind=str(gap.get("kind") or "unknown"),
            detail=str(gap.get("detail") or ""),
            event_id=gap.get("event_id"),
            parent_id=gap.get("parent_id"),
        ) for gap in run.gaps],
        refused=[],
    )


def _trace_like(run: Any):
    from .event_v1 import TraceEnvelope

    trace = run.trace or {}
    return TraceEnvelope(
        trace_id=str(trace.get("trace_id") or ""),
        run_id=str(trace.get("run_id") or run.run_id),
        importer=str(trace.get("importer") or ""),
        importer_version=str(trace.get("importer_version") or ""),
        source_format=str(trace.get("source_format") or ""),
        source_scope=str(trace.get("source_scope") or ""),
        created_at=str(trace.get("created_at") or ""),
        source=dict(trace.get("source") or {}),
        redaction=dict(trace.get("redaction") or {}),
        covered_kinds=tuple(trace.get("covered_kinds") or ()),
        unsupported_kinds=tuple(trace.get("unsupported_kinds") or ()),
        completeness=dict(trace.get("completeness") or {}),
    )


def _attempt_like(payload: Mapping[str, Any]):
    from .event_v1 import Attempt

    return Attempt(
        attempt_id=str(payload.get("attempt_id") or ""),
        run_id=str(payload.get("run_id") or ""),
        outcome=str(payload.get("outcome") or "unknown"),
        parent_attempt_id=payload.get("parent_attempt_id"),
        model=payload.get("model"),
        provider=payload.get("provider"),
        role=payload.get("role"),
        branch=payload.get("branch"),
        sequence=int(payload.get("sequence") or 0),
        started_at=payload.get("started_at"),
        ended_at=payload.get("ended_at"),
        unknowns=tuple(payload.get("unknowns") or ()),
    )


def run_cli(argv: Sequence[str], services: Services) -> int:
    """Dispatch one CLI invocation. Returns an exit code; never raises for user error."""
    args = list(argv)
    if not args:
        print(_USAGE)
        return 2
    command, rest = args[0], args[1:]
    try:
        if command == "import":
            files = _flag_values(rest, "--file")
            if not files:
                print("import requires at least one --file <path>")
                return 2
            fmt = _flag_value(rest, "--format")
            allow_partial = "--allow-partial" in rest
            out = services.import_sources(files, fmt=fmt, allow_partial=allow_partial or None)
            for item in out:
                print(json.dumps(item, indent=2, ensure_ascii=False))
            return 0
        if command == "timeline":
            run_id = _positional(rest) or _flag_value(rest, "--run")
            print(services.timeline_text(_require_run(run_id), event_ref=_flag_value(rest, "--event")))
            return 0
        if command == "explain":
            run_id = _positional(rest) or _flag_value(rest, "--run")
            print(services.explain_text(_require_run(run_id), as_json="--json" in rest))
            return 0
        if command == "replay":
            run_id = _positional(rest) or _flag_value(rest, "--run")
            print(services.replay_text(_require_run(run_id)))
            return 0
        if command == "export":
            run_id = _flag_value(rest, "--run")
            out = _flag_value(rest, "--out")
            if not run_id or not out:
                print("export requires --run <run-id> and --out <private-path>")
                return 2
            fmt = _flag_value(rest, "--format") or (
                "markdown" if str(out).endswith((".md", ".markdown")) else "json"
            )
            path = services.export_report(run_id, out, fmt=fmt)
            print(f"wrote private report: {path} (format {fmt}; share it only if you choose to)")
            return 0
        if command == "status":
            print(json.dumps(services.status(), indent=2, ensure_ascii=False))
            return 0
        print(f"unknown command {command!r}\n{_USAGE}")
        return 2
    except ImportRefused as exc:
        print(f"refused ({exc.kind}): {'; '.join(exc.reasons)}")
        return 3
    except InputLimitExceeded as exc:
        print(f"refused (limits): {exc}")
        return 3
    except (TimelineContradiction, FileNotFoundError, ValueError) as exc:
        print(f"error: {sanitise_message(str(exc))}")
        return 1


def _require_run(run_id: str | None) -> str:
    if not run_id:
        raise ValueError("a run id is required (use `status` to list imported runs)")
    return run_id


def _positional(args: Sequence[str]) -> str | None:
    for token in args:
        if not token.startswith("-"):
            return token
    return None


def _flag_value(args: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(args):
        if token == flag and index + 1 < len(args):
            return args[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _flag_values(args: Sequence[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(args):
        if token == flag and index + 1 < len(args):
            values.append(args[index + 1])
        elif token.startswith(f"{flag}="):
            values.append(token.split("=", 1)[1])
    return values


def handle_slash(raw_args: str, services: Services) -> str:
    """Slash-command surface. Never raises into the host; export always refuses."""
    tokens = [token for token in (raw_args or "").split() if token]
    if not tokens:
        return _USAGE
    command, rest = tokens[0].lower(), tokens[1:]
    if command in {"export", "share", "upload"}:
        return EXPORT_REFUSAL
    if command not in {"import", "timeline", "explain", "replay", "status"}:
        return f"unknown subcommand {command!r}\n{_USAGE}"
    try:
        if command == "import":
            files = [token for token in rest if not token.startswith("-")]
            if not files:
                return "import needs at least one file path: /rundebug import <file>"
            fmt = _flag_value(rest, "--format")
            out = services.import_sources(files, fmt=fmt, allow_partial="--allow-partial" in rest or None)
            lines = [
                f"{item['run_id']}: {item['events']} event(s), {item['attempts']} attempt(s), "
                f"{item['gaps']} gap(s), completeness={item['may_claim_complete']}"
                for item in out
            ]
            return "imported\n" + "\n".join(lines) + f"\nnext: /rundebug explain {out[0]['run_id']}"
        if command == "status":
            status = services.status()
            lanes = ", ".join(f"{lane}={state}" for lane, state in status["lanes"].items())
            return (
                f"runs: {status['runs']}\nlanes: {lanes}\n"
                f"export: operator CLI only (`hermes run-debug export`)\n"
                "nothing is uploaded from this route"
            )
        run_id = _positional(rest)
        if not run_id:
            return f"{command} needs a run id\n{_USAGE}"
        if command == "timeline":
            return services.timeline_text(run_id, event_ref=_flag_value(rest, "--event"))
        if command == "explain":
            return services.explain_text(run_id)
        return services.replay_text(run_id)
    except ImportRefused as exc:
        return f"refused ({exc.kind}): {'; '.join(exc.reasons)}"
    except (InputLimitExceeded, TimelineContradiction, FileNotFoundError, ValueError) as exc:
        return f"error: {sanitise_message(str(exc))}"


# -- agent tools ----------------------------------------------------------
def tool_import(args: Mapping[str, Any] | None, services: Services) -> str:
    payload = dict(args or {})
    files = payload.get("files") or []
    if isinstance(files, str):
        files = [files]
    fmt = payload.get("format")
    try:
        out = services.import_sources(files, fmt=fmt, allow_partial=payload.get("allow_partial") or None)
    except ImportRefused as exc:
        return json.dumps({"ok": False, "refused": exc.reasons}, ensure_ascii=False)
    return json.dumps({"ok": True, "imports": out}, ensure_ascii=False)


def tool_timeline(args: Mapping[str, Any] | None, services: Services) -> str:
    payload = dict(args or {})
    run_id = str(payload.get("run") or payload.get("run_id") or "")
    if not run_id:
        return json.dumps({"ok": False, "error": "a run id is required"})
    try:
        return services.timeline_text(run_id, event_ref=payload.get("event"))
    except (FileNotFoundError, ValueError, TimelineContradiction) as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def tool_explain(args: Mapping[str, Any] | None, services: Services) -> str:
    payload = dict(args or {})
    run_id = str(payload.get("run") or payload.get("run_id") or "")
    if not run_id:
        return json.dumps({"ok": False, "error": "a run id is required"})
    try:
        return services.explain_text(run_id, as_json=bool(payload.get("json", False)))
    except (FileNotFoundError, ValueError, TimelineContradiction) as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def tool_replay(args: Mapping[str, Any] | None, services: Services) -> str:
    payload = dict(args or {})
    run_id = str(payload.get("run") or payload.get("run_id") or "")
    if not run_id:
        return json.dumps({"ok": False, "error": "a run id is required"})
    try:
        return services.replay_text(run_id)
    except (FileNotFoundError, ValueError, TimelineContradiction) as exc:
        return json.dumps({"ok": False, "error": str(exc)})


def tool_status(args: Mapping[str, Any] | None, services: Services) -> str:
    del args
    return json.dumps(services.status(), indent=2, ensure_ascii=False)

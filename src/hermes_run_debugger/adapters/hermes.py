"""Hermes host mapping: the only module in this plugin that touches the host.

What this adapter does and does not do:

* Registers **ordinary** surfaces: one CLI subcommand, one slash command, five
  tools. No privileged capability is declared or needed
  (``tools.override``, ``llm.*``, ``gateway.platform_actions`` are all untouched).
* Registers **no hooks**. This build has no capture lane, and the host's
  ``pre_tool_call`` is fail-closed, so an observer there could block the user's
  agent. When a capture lane ships it will use validated non-policy result/API/
  session hooks and be advertised for the routes actually tested.
* Performs **no I/O at registration time**. The store is opened on first use, and
  the data directory is resolved per operation against the active profile rather
  than cached, so switching profiles cannot write into the wrong profile's data.
* Never reads credentials, config secrets, ``.env`` files or ``auth.json``.
* Never patches core behaviour, monkeypatches a registry, or re-implements a
  built-in tool.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from ..cli import (
    Services,
    handle_slash,
    run_cli,
    tool_explain,
    tool_import,
    tool_replay,
    tool_status,
    tool_timeline,
)
from ..import_v1 import SUPPORTED_FORMATS
from ..settings import Settings

__all__ = ["PLUGIN_ID", "TOOL_NAMES", "build_services", "register"]

PLUGIN_ID = "agent-run-debugger"

#: Plugin-relative settings keys read through ``ctx.get_config``.
SETTING_KEYS = (
    "capture_enabled",
    "live_read_enabled",
    "llm_enabled",
    "experiment_enabled",
    "import_max_bytes",
    "import_max_line_bytes",
    "import_max_events",
    "import_max_depth",
    "allow_partial",
    "retention_days",
    "retention_mode",
)

TOOL_NAMES = (
    "rundebug_import",
    "rundebug_timeline",
    "rundebug_explain",
    "rundebug_replay",
    "rundebug_status",
)

_HELP = "Import a selected agent trace, then show its timeline, diagnosis or inert replay."

_IMPORT_SCHEMA = {
    "name": "rundebug_import",
    "description": (
        "Import one or more explicitly selected trace files (agent-run-debugger format 'v1' or a "
        "Claude/Hermes JSONL transcript, or explicitly selected Cauldron JSON receipts). "
        "Reads only the files named here; directories and archives are "
        "refused. Nothing is executed and nothing leaves the machine."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Absolute paths of the selected trace files.",
            },
            "format": {
                "type": "string",
                "enum": list(SUPPORTED_FORMATS),
                "description": "Source format. Omit to auto-detect the supported shapes.",
            },
            "allow_partial": {
                "type": "boolean",
                "description": (
                    "Keep the valid prefix of a truncated final line, marked PARTIAL with an import "
                    "gap. Default false: a truncated source is refused."
                ),
            },
        },
        "required": ["files"],
    },
}

_RUN_SCHEMA = {
    "name": "{name}",
    "description": "{description}",
    "parameters": {
        "type": "object",
        "properties": {
            "run": {"type": "string", "description": "Run id returned by an earlier import."},
        },
        "required": ["run"],
    },
}

_STATUS_SCHEMA = {
    "name": "rundebug_status",
    "description": (
        "Report the debugger's own state: imported runs, which lanes are implemented, which are "
        "refused, and the active bounds. No export route and no upload path exists here."
    ),
    "parameters": {"type": "object", "properties": {}},
}


def _settings_from(ctx: Any) -> Settings:
    raw: dict[str, Any] = {}
    for key in SETTING_KEYS:
        value = ctx.get_config(key, None)
        if value is not None:
            raw[key] = value
    return Settings.from_mapping(raw)


def _lazy_data_dir(ctx: Any) -> Callable[[], Path]:
    def resolve() -> Path:
        try:
            from plugins.plugin_storage import plugin_data_dir

            return plugin_data_dir(PLUGIN_ID)
        except Exception:
            # Documented fallback: the host storage helper is unavailable (for example
            # a non-Hermes harness). This is a location choice, not a security
            # boundary, and it is reported in `status`.
            home = Path(os.environ.get("HERMES_HOME") or "~/.hermes").expanduser()
            return home / "plugin-data" / PLUGIN_ID

    return resolve


def build_services(ctx: Any) -> Services:
    return Services(data_dir=_lazy_data_dir(ctx), settings=_settings_from(ctx))


def register(ctx: Any) -> None:
    """Register the CLI subcommand, the slash command and the five tools."""
    services = build_services(ctx)

    ctx.register_cli_command(
        "run-debug",
        help=_HELP,
        setup_fn=_setup_cli_command,
        handler_fn=partial(_handle_cli_command, services=services),
        description="Local-first evidence debugger for agent runs (offline import; inert replay).",
    )
    ctx.register_command(
        "rundebug",
        handler=lambda raw_args="": handle_slash(raw_args, services),
        description=_HELP,
        args_hint="<import|timeline|explain|replay|status>",
        argument_mode="text",
    )

    import_handler = _wrap(services, tool_import)
    ctx.register_tool(
        name="rundebug_import",
        toolset="rundebug",
        schema=_IMPORT_SCHEMA,
        handler=import_handler,
        description=_IMPORT_SCHEMA["description"],
    )
    for name, description, handler in (
        ("rundebug_timeline", "Show the recorded timeline: order, provenance, gaps, ambiguity.", tool_timeline),
        ("rundebug_explain", "Explain a run: observed failures, ranked hypotheses, missing evidence.", tool_explain),
        ("rundebug_replay", "Inert replay of the recorded timeline. Executes nothing.", tool_replay),
        ("rundebug_status", "Debugger status: runs, implemented lanes, refusals, active bounds.", tool_status),
    ):
        schema = (
            _STATUS_SCHEMA if name == "rundebug_status"
            else dict(_RUN_SCHEMA, name=name, description=description)
        )
        if name in {"rundebug_timeline", "rundebug_explain"}:
            parameters = dict(schema["parameters"])
            properties = dict(parameters["properties"])
            if name == "rundebug_timeline":
                properties["event"] = {
                    "type": "string",
                    "description": "Inspect this exact event plus recorded links and same-request follow-through.",
                }
            else:
                properties["json"] = {
                    "type": "boolean", "description": "Return structured action chains and hypotheses.",
                }
            parameters["properties"] = properties
            schema = dict(schema, parameters=parameters)
        ctx.register_tool(
            name=name,
            toolset="rundebug",
            schema=schema,
            handler=_wrap(services, handler),
            description=description,
        )

    on_unload = getattr(ctx, "on_unload", None)
    if callable(on_unload):
        on_unload(services.close)


def _wrap(services: Services, handler: Callable[[Any, Services], str]) -> Callable[..., str]:
    def inner(args: Any = None, **_kwargs: Any) -> str:
        return handler(args, services)

    return inner


def _setup_cli_command(parser: Any) -> None:
    sub = parser.add_subparsers(dest="rundebug_command")
    import_cmd = sub.add_parser("import", help="Import explicitly selected trace files.")
    import_cmd.add_argument("--file", action="append", required=True, help="Path to a selected trace file.")
    import_cmd.add_argument("--format", choices=list(SUPPORTED_FORMATS), default=None)
    import_cmd.add_argument("--allow-partial", action="store_true")
    for name, help_text in (
        ("timeline", "Show the recorded timeline."),
        ("explain", "Show observed failures, hypotheses and missing evidence."),
        ("replay", "Inert replay of the recorded timeline."),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--run", required=True)
        if name == "timeline":
            command.add_argument("--event", help="Inspect one event and its recorded action relationships.")
        if name == "explain":
            command.add_argument("--json", action="store_true")
    export_cmd = sub.add_parser("export", help="Write a private report (operator CLI only).")
    export_cmd.add_argument("--run", required=True)
    export_cmd.add_argument("--out", required=True)
    export_cmd.add_argument("--format", choices=["markdown", "json", "html"], default=None)
    sub.add_parser("status", help="Show debugger status.")
    parser.set_defaults(rundebug_command=None)


def _handle_cli_command(args: Any, *, services: Services) -> int:
    # Hermes passes an argparse Namespace here; use the services bound at registration.
    command = getattr(args, "rundebug_command", None)
    if not command:
        print(_HELP)
        return 0
    argv = [command]
    if command == "import":
        for path in args.file or []:
            argv += ["--file", path]
        if args.format:
            argv += ["--format", args.format]
        if args.allow_partial:
            argv.append("--allow-partial")
    elif command == "export":
        argv += ["--run", args.run, "--out", args.out]
        if args.format:
            argv += ["--format", args.format]
    elif command in {"timeline", "explain", "replay"}:
        argv += ["--run", args.run]
        if getattr(args, "json", False):
            argv.append("--json")
        if getattr(args, "event", None):
            argv += ["--event", args.event]
    try:
        return run_cli(argv, services)
    finally:
        services.close()

#!/usr/bin/env python3
"""
Hierarchical view of Taskwarrior moves by dependency.
"""

import importlib
import json
import os
import re
import shlex
import tempfile
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .taskwarrior import (
    apply_case_insensitive_overrides,
    apply_taskrc_overrides,
    describe_missing_udas,
    filter_modified_zero_lines,
    get_tasks_from_taskwarrior,
    missing_udas,
    normalize_dependency_value,
    parse_dependencies,
    read_tasks_from_json,
)

def enable_line_editing(is_interactive: Optional[bool] = None) -> bool:
    """
    Enable readline-style line editing for interactive input prompts.

    Parameters
    ----------
    is_interactive : Optional[bool], optional
        Override for stdin TTY detection (default: sys.stdin.isatty()).

    Returns
    -------
    bool
        True when a readline-compatible module is available.

    Examples
    --------
    >>> enable_line_editing(is_interactive=False)
    False
    """
    if is_interactive is None:
        is_interactive = sys.stdin.isatty()
    if not is_interactive:
        return False

    for module_name in ("readline", "pyreadline3", "pyreadline"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
        return True
    return False


def build_tree_prefix(ancestor_has_more: List[bool]) -> str:
    parts = []
    for has_more in ancestor_has_more:
        parts.append("|  " if has_more else "   ")
    parts.append("+- ")
    return "".join(parts)


def get_taskwarrior_setting(key: str) -> Optional[str]:
    try:
        task_args = apply_taskrc_overrides(["_get", key])
        result = subprocess.run(
            ["task", *task_args],
            capture_output=True,
            text=True,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    value = result.stdout.strip()
    return value if value else None


def normalize_context_name(context_name: Optional[str]) -> Optional[str]:
    """
    Normalize Taskwarrior context names, treating "none" as inactive.

    Parameters
    ----------
    context_name : Optional[str]
        Raw context name value from Taskwarrior.

    Returns
    -------
    Optional[str]
        Normalized context name, or None when no context is active.

    Examples
    --------
    >>> normalize_context_name("work")
    'work'
    >>> normalize_context_name("none") is None
    True
    >>> normalize_context_name("  ") is None
    True
    """
    if context_name is None:
        return None
    name = context_name.strip()
    if not name or name.lower() == "none":
        return None
    return name


def get_active_context_name() -> Optional[str]:
    """
    Return the active Taskwarrior context name, if any.

    Returns
    -------
    Optional[str]
        Active context name or None when no context is active.
    """
    for key in ("rc.context", "context"):
        value = get_taskwarrior_setting(key)
        if value is None:
            continue
        return normalize_context_name(value)
    return None


def get_context_definition(context_name: str) -> Optional[str]:
    """
    Resolve the Taskwarrior context definition for a given context.

    Parameters
    ----------
    context_name : str
        Name of the Taskwarrior context.

    Returns
    -------
    Optional[str]
        Context definition string, if available.
    """
    for key in (f"context.{context_name}", f"rc.context.{context_name}"):
        value = get_taskwarrior_setting(key)
        if value:
            return value.strip()
    return None


def parse_context_filters(context_definition: str) -> Tuple[Optional[str], List[str]]:
    """
    Extract project and positive tags from a context definition.

    Parameters
    ----------
    context_definition : str
        Taskwarrior context definition string.

    Returns
    -------
    Tuple[Optional[str], List[str]]
        Project name (if defined) and a list of tag names.

    Examples
    --------
    >>> parse_context_filters("project:work +alpha +beta")
    ('work', ['alpha', 'beta'])
    >>> parse_context_filters("+home -waiting")
    (None, ['home'])
    >>> parse_context_filters("tag:alpha project:ops")
    ('ops', ['alpha'])
    """
    project = None
    tags: List[str] = []
    seen_tags = set()

    for token in shlex.split(context_definition):
        if token.startswith("project:"):
            value = token[len("project:"):]
            if value:
                project = value
            continue
        if token.startswith("+") and len(token) > 1:
            tag = token[1:]
            if tag not in seen_tags:
                tags.append(tag)
                seen_tags.add(tag)
            continue
        if token.startswith("tag:") or token.startswith("tags:"):
            _, value = token.split(":", 1)
            if value and value not in seen_tags:
                tags.append(value)
                seen_tags.add(value)
    return project, tags


def parse_add_attributes(args: List[str]) -> Tuple[Optional[str], Set[str]]:
    """
    Parse task add arguments for existing project and tags.

    Parameters
    ----------
    args : List[str]
        Arguments passed after the ``add`` command.

    Returns
    -------
    Tuple[Optional[str], Set[str]]
        Project name and set of tags already provided.
    """
    project = None
    tags: Set[str] = set()

    for arg in args:
        if arg == "--":
            break
        if arg.startswith("project:"):
            value = arg[len("project:"):]
            if value:
                project = value
            continue
        if arg.startswith("+") and len(arg) > 1:
            # Taskwarrior treats +tag tokens as tag filters/assignments.
            tags.add(arg[1:])
            continue
        if arg.startswith("tag:") or arg.startswith("tags:"):
            _, value = arg.split(":", 1)
            if value:
                tags.add(value)
    return project, tags


def insert_additions_before_double_dash(argv: List[str], additions: List[str]) -> List[str]:
    """
    Insert new arguments before a ``--`` delimiter, if present.

    Parameters
    ----------
    argv : List[str]
        Command arguments to modify.
    additions : List[str]
        Arguments to insert.

    Returns
    -------
    List[str]
        Updated argument list with additions inserted safely.
    """
    if "--" in argv:
        index = argv.index("--")
        return argv[:index] + additions + argv[index:]
    return argv + additions


def format_context_message(
    context_name: str,
    project: Optional[str],
    tags: List[str],
) -> Optional[str]:
    """
    Build an informational message describing context-based additions.

    Parameters
    ----------
    context_name : str
        Active context name.
    project : Optional[str]
        Project name added from the context.
    tags : List[str]
        Tag names added from the context.

    Returns
    -------
    Optional[str]
        Message to display, or None when nothing was added.
    """
    if not project and not tags:
        return None

    parts: List[str] = []
    if project:
        parts.append(f"project set to {project}")
    if tags:
        label = "tag" if len(tags) == 1 else "tags"
        parts.append(f"{label} set to {', '.join(tags)}")
    return f"twh: {'; '.join(parts)} because context is {context_name}"


def apply_context_to_add_args(argv: List[str]) -> Tuple[List[str], Optional[str]]:
    """
    Append context-derived project or tags to ``task add`` arguments.

    Parameters
    ----------
    argv : List[str]
        Command-line arguments excluding the program name.

    Returns
    -------
    Tuple[List[str], Optional[str]]
        Updated arguments and an informational message, if applicable.
    """
    if not argv or argv[0] != "add":
        return argv, None

    context_name = get_active_context_name()
    if not context_name:
        return argv, None

    context_definition = get_context_definition(context_name)
    if not context_definition:
        return argv, None

    project, tags = parse_context_filters(context_definition)
    if not project and not tags:
        return argv, None

    existing_project, existing_tags = parse_add_attributes(argv[1:])
    additions: List[str] = []
    added_project = None
    added_tags: List[str] = []

    if project and not existing_project:
        additions.append(f"project:{project}")
        added_project = project

    for tag in tags:
        if tag not in existing_tags:
            additions.append(f"+{tag}")
            added_tags.append(tag)

    if not additions:
        return argv, None

    updated_args = insert_additions_before_double_dash(argv, additions)
    message = format_context_message(context_name, added_project, added_tags)
    return updated_args, message


@dataclass(frozen=True)
class AddMoveInput:
    """
    Interactive add inputs for a new move.

    Attributes
    ----------
    description : str
        Move description.
    project : Optional[str]
        Project name, if provided.
    tags : List[str]
        Tags to apply to the move.
    due : Optional[str]
        Due date string.
    blocks : List[str]
        Move IDs blocked by this move.
    metadata : Dict[str, str]
        Metadata fields keyed by UDA name (imp/urg/opt_human/diff/mode).
    """

    description: str
    project: Optional[str]
    tags: List[str]
    due: Optional[str]
    blocks: List[str]
    metadata: Dict[str, str]


def normalize_description_default(args: Sequence[str]) -> Optional[str]:
    """
    Collapse add arguments into a default description.

    Parameters
    ----------
    args : Sequence[str]
        Extra tokens provided after ``add``.

    Returns
    -------
    Optional[str]
        Joined description default, or None when empty.

    Examples
    --------
    >>> normalize_description_default(["Write", "notes"])
    'Write notes'
    >>> normalize_description_default([]) is None
    True
    >>> normalize_description_default([" "]) is None
    True
    """
    joined = " ".join(str(arg).strip() for arg in args).strip()
    return joined or None


def parse_tag_input(value: Optional[str]) -> List[str]:
    """
    Parse comma-separated tag input into unique tag names.

    Parameters
    ----------
    value : Optional[str]
        Raw tag input.

    Returns
    -------
    List[str]
        Parsed tag names.

    Examples
    --------
    >>> parse_tag_input("alpha, +beta, alpha")
    ['alpha', 'beta']
    >>> parse_tag_input("") == []
    True
    """
    tags: List[str] = []
    seen = set()
    for raw in split_csv(value):
        tag = raw.strip()
        if tag.startswith("+"):
            tag = tag[1:]
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def parse_blocks_input(value: Optional[str]) -> List[str]:
    """
    Parse comma-separated block IDs into a unique list.

    Parameters
    ----------
    value : Optional[str]
        Raw blocks input.

    Returns
    -------
    List[str]
        Parsed block IDs.

    Examples
    --------
    >>> parse_blocks_input("12, 34, 12")
    ['12', '34']
    >>> parse_blocks_input(None)
    []
    """
    blocks: List[str] = []
    seen = set()
    for raw in split_csv(value):
        block = raw.strip()
        if not block or block in seen:
            continue
        blocks.append(block)
        seen.add(block)
    return blocks


def prompt_add_metadata(input_func: Callable[[str], str] = input) -> Dict[str, str]:
    """
    Prompt for add metadata fields using the ondeck wizard prompts.

    Parameters
    ----------
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).

    Returns
    -------
    Dict[str, str]
        Metadata updates keyed by UDA name.
    """
    from .review import ReviewTask, interactive_fill_missing

    placeholder = ReviewTask(
        uuid="new",
        id=None,
        description="",
        project=None,
        depends=[],
        imp=None,
        urg=None,
        opt=None,
        diff=None,
        mode=None,
        raw={},
    )
    return interactive_fill_missing(placeholder, input_func=input_func)


def prompt_add_input(
    input_func: Callable[[str], str] = input,
    description_default: Optional[str] = None,
) -> AddMoveInput:
    """
    Prompt the user for new move details in the required order.

    Parameters
    ----------
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).
    description_default : Optional[str], optional
        Default description to offer (default: None).

    Returns
    -------
    AddMoveInput
        Collected add input data.
    """
    while True:
        if description_default:
            prompt = f"Move description [{description_default}]: "
        else:
            prompt = "Move description: "
        description = input_func(prompt).strip()
        if description:
            break
        if description_default:
            description = description_default
            break
        print("Move description is required.")

    project_value = input_func("Project: ").strip()
    project = project_value if project_value else None

    tags = parse_tag_input(input_func("Tags (comma-separated): ").strip())

    due_value = input_func("Due date: ").strip()
    due = due_value if due_value else None

    blocks = parse_blocks_input(
        input_func("Blocks (move IDs blocked by this move, comma-separated): ").strip()
    )

    metadata = prompt_add_metadata(input_func=input_func)

    return AddMoveInput(
        description=description,
        project=project,
        tags=tags,
        due=due,
        blocks=blocks,
        metadata=metadata,
    )


def build_add_args(add_input: AddMoveInput) -> List[str]:
    """
    Build Taskwarrior add arguments from interactive inputs.

    Parameters
    ----------
    add_input : AddMoveInput
        Collected add inputs.

    Returns
    -------
    List[str]
        Taskwarrior arguments for ``task add``.

    Examples
    --------
    >>> add_input = AddMoveInput(
    ...     description="Write notes",
    ...     project="work",
    ...     tags=["alpha"],
    ...     due="2024-02-01",
    ...     blocks=[],
    ...     metadata={"imp": "10", "mode": "analysis"},
    ... )
    >>> build_add_args(add_input)
    ['add', 'Write notes', 'project:work', '+alpha', 'due:2024-02-01', 'imp:10', 'mode:analysis']
    """
    args = ["add", add_input.description]
    if add_input.project:
        args.append(f"project:{add_input.project}")
    for tag in add_input.tags:
        args.append(f"+{tag}")
    if add_input.due:
        args.append(f"due:{add_input.due}")
    for field in ("imp", "urg", "opt_human", "diff", "mode"):
        value = add_input.metadata.get(field)
        if value:
            args.append(f"{field}:{value}")
    return args


def apply_blocks_to_targets(
    created_uuid: str,
    blocks: Iterable[str],
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    quiet: bool = False,
) -> int:
    """
    Apply dependency updates for blocks targets.

    Parameters
    ----------
    created_uuid : str
        Newly created move UUID.
    blocks : Iterable[str]
        IDs of blocked moves.
    runner : Callable[..., subprocess.CompletedProcess], optional
        Runner for Taskwarrior commands (default: run_task_command).
    quiet : bool, optional
        Suppress Taskwarrior stdout when True (default: False).

    Returns
    -------
    int
        Exit code from the blocks updates.
    """
    if runner is None:
        runner = run_task_command

    return apply_dependency_updates(blocks, [created_uuid], runner=runner, quiet=quiet)


def run_interactive_add(
    argv: Sequence[str],
    input_func: Callable[[str], str] = input,
) -> int:
    """
    Run the interactive ``twh add`` workflow.

    Parameters
    ----------
    argv : Sequence[str]
        Extra command-line arguments after ``add``.
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).

    Returns
    -------
    int
        Exit code for the add workflow.
    """
    description_default = normalize_description_default(argv)
    add_input = prompt_add_input(
        input_func=input_func,
        description_default=description_default,
    )
    add_args = build_add_args(add_input)
    add_args, message = apply_context_to_add_args(add_args)
    if message:
        print(message)
        sys.stdout.flush()

    if add_input.metadata:
        missing = missing_udas(
            add_input.metadata.keys(),
            allow_taskrc_fallback=False,
        )
        if missing:
            print(
                f"twh: add failed: {describe_missing_udas(missing)}",
                file=sys.stderr,
            )
            return 1

    add_result = run_task_command(add_args, capture_output=True)
    if add_result.stdout:
        for line in filter_modified_zero_lines(add_result.stdout):
            print(line)
    if add_result.stderr:
        for line in filter_modified_zero_lines(add_result.stderr):
            print(line, file=sys.stderr)
    if add_result.returncode != 0:
        return add_result.returncode

    created_id = parse_created_task_id(add_result.stdout or "")
    exit_code = 0
    if add_input.blocks:
        if not created_id:
            print("twh: unable to parse created move id for blocks.", file=sys.stderr)
            return 1
        created_uuid = resolve_task_uuid(created_id)
        if not created_uuid:
            print("twh: unable to resolve created move uuid for blocks.", file=sys.stderr)
            return 1
        exit_code = apply_blocks_to_targets(
            created_uuid,
            add_input.blocks,
            quiet=True,
        )

    missing_dominance = missing_udas(
        ["dominates", "dominated_by"],
        allow_taskrc_fallback=False,
    )
    if missing_dominance:
        print(
            f"twh: add failed: {describe_missing_udas(missing_dominance)}",
            file=sys.stderr,
        )
        return exit_code or 1

    from . import dominance as dominance_module

    dominance_exit = dominance_module.run_dominance(
        input_func=input_func,
        quiet=True,
    )
    if exit_code == 0:
        exit_code = dominance_exit
    return exit_code


def parse_default_command(command: Optional[str]) -> str:
    """
    Parse the report name from the default Taskwarrior command.

    Parameters
    ----------
    command : Optional[str]
        Raw ``default.command`` value.

    Returns
    -------
    str
        Report name, defaulting to "next".

    Examples
    --------
    >>> parse_default_command("next")
    'next'
    >>> parse_default_command("next +work")
    'next'
    >>> parse_default_command(None)
    'next'
    """
    report_name, _ = parse_default_command_tokens(command)
    return report_name


def parse_default_command_tokens(command: Optional[str]) -> Tuple[str, List[str]]:
    """
    Parse the default report name and filters from ``default.command``.

    Parameters
    ----------
    command : Optional[str]
        Raw ``default.command`` value.

    Returns
    -------
    Tuple[str, List[str]]
        Report name and remaining default filters.

    Examples
    --------
    >>> parse_default_command_tokens("next +work")
    ('next', ['+work'])
    >>> parse_default_command_tokens("next")
    ('next', [])
    >>> parse_default_command_tokens(None)
    ('next', [])
    """
    if not command:
        return "next", []
    tokens = shlex.split(command)
    if not tokens:
        return "next", []
    return tokens[0], tokens[1:]


def get_default_report_name() -> str:
    """
    Return the default Taskwarrior report name.

    Returns
    -------
    str
        Report name derived from ``default.command``.
    """
    if should_disable_simple_pager():
        return parse_default_command(get_taskwarrior_setting_simple("default.command"))
    return parse_default_command(get_taskwarrior_setting("default.command"))


def get_default_command_tokens() -> Tuple[str, List[str]]:
    """
    Return the default report name and filters from Taskwarrior.

    Returns
    -------
    Tuple[str, List[str]]
        Report name and default filters.
    """
    if should_disable_simple_pager():
        return parse_default_command_tokens(
            get_taskwarrior_setting_simple("default.command")
        )
    return parse_default_command_tokens(get_taskwarrior_setting("default.command"))


def is_wsl() -> bool:
    """
    Return True when running inside WSL.

    Returns
    -------
    bool
        True if WSL environment variables or kernel markers are present.
    """
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def should_disable_simple_pager() -> bool:
    """
    Determine whether to disable the pager for ``twh simple``.

    Returns
    -------
    bool
        True when the pager should be disabled.
    """
    override = os.environ.get("TWH_SIMPLE_PAGER", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return False
    return is_wsl()


def simple_task_overrides() -> List[str]:
    """
    Build Taskwarrior overrides for ``twh simple``.

    Returns
    -------
    List[str]
        List of ``rc.`` overrides.
    """
    if not should_disable_simple_pager():
        return []
    return [
        "rc.pager=cat",
        "rc.confirmation=off",
        "rc.hooks=off",
    ]


def replace_description_column(columns: str) -> str:
    """
    Replace description columns with ``description.count``.

    Parameters
    ----------
    columns : str
        Comma-separated column list.

    Returns
    -------
    str
        Updated column list with annotation counts.

    Examples
    --------
    >>> replace_description_column("id,description")
    'id,description.count'
    >>> replace_description_column("description.truncated,project")
    'description.count,project'
    """
    updated: List[str] = []
    for column in split_csv(columns):
        if column.lower().startswith("description"):
            updated.append("description.count")
        else:
            updated.append(column)
    return ",".join(updated)


def strip_limit_page(value: str) -> str:
    """
    Remove ``limit:page`` tokens from a report filter string.

    Parameters
    ----------
    value : str
        Report filter string.

    Returns
    -------
    str
        Filter string without ``limit:page``.

    Examples
    --------
    >>> strip_limit_page("status:pending limit:page")
    'status:pending'
    >>> strip_limit_page("limit:page")
    ''
    """
    tokens = [token for token in shlex.split(value) if token != "limit:page"]
    return " ".join(tokens)


def strip_limit_page_tokens(tokens: List[str]) -> List[str]:
    """
    Remove ``limit:page`` from a list of tokens.

    Parameters
    ----------
    tokens : List[str]
        Tokens to filter.

    Returns
    -------
    List[str]
        Tokens without ``limit:page``.

    Examples
    --------
    >>> strip_limit_page_tokens(["status:pending", "limit:page"])
    ['status:pending']
    """
    return [token for token in tokens if token != "limit:page"]


def run_simple_task_command(
    args: List[str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run Taskwarrior for ``twh simple`` with non-interactive safeguards.

    Parameters
    ----------
    args : List[str]
        Taskwarrior arguments excluding the executable.
    capture_output : bool, optional
        Whether to capture stdout/stderr (default: False).

    Returns
    -------
    subprocess.CompletedProcess
        Completed Taskwarrior process.
    """
    task_args = [*simple_task_overrides(), *args]
    return run_task_command(
        task_args,
        capture_output=capture_output,
        stdin=subprocess.DEVNULL,
    )


def parse_report_settings(report_name: str, output: str) -> Dict[str, str]:
    """
    Parse ``task show report.<name>`` output into a settings map.

    Parameters
    ----------
    report_name : str
        Report name to extract.
    output : str
        Output from ``task show report.<name>``.

    Returns
    -------
    Dict[str, str]
        Mapping of setting suffix to value.

    Examples
    --------
    >>> output = "report.next.columns id,description\\nreport.next.sort urgency-"
    >>> parse_report_settings("next", output)
    {'columns': 'id,description', 'sort': 'urgency-'}
    """
    prefix = f"report.{report_name}."
    settings: Dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or not line.startswith(prefix):
            continue
        parts = line.split(None, 1)
        key = parts[0]
        value = parts[1].strip() if len(parts) > 1 else ""
        suffix = key[len(prefix):]
        settings[suffix] = value
    return settings


def get_report_settings(
    report_name: str,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> Dict[str, str]:
    """
    Read report settings from Taskwarrior.

    Parameters
    ----------
    report_name : str
        Report name to read.

    Returns
    -------
    Dict[str, str]
        Mapping of setting suffix to value.
    """
    if runner is None:
        runner = run_task_command
    result = runner(
        ["show", f"report.{report_name}"],
        capture_output=True,
    )
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return {}
    return parse_report_settings(report_name, result.stdout or "")


def configure_report(
    report_name: str,
    settings: Dict[str, str],
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> bool:
    """
    Persist report settings to Taskwarrior.

    Parameters
    ----------
    report_name : str
        Report name to configure.
    settings : Dict[str, str]
        Mapping of setting suffix to value.

    Returns
    -------
    bool
        True when configuration succeeds.
    """
    if runner is None:
        runner = run_task_command
    for key, value in settings.items():
        if not value:
            continue
        result = runner(
            ["config", f"report.{report_name}.{key}", value],
            capture_output=True,
        )
        if result.returncode != 0:
            if result.stderr:
                for line in filter_modified_zero_lines(result.stderr):
                    print(line, file=sys.stderr)
            return False
    return True


def ensure_simple_report(base_report: str) -> bool:
    """
    Ensure the ``simple`` report exists with annotation counts.

    Parameters
    ----------
    base_report : str
        Report name to copy from.

    Returns
    -------
    bool
        True when the report exists or is created.
    """
    existing_columns = get_taskwarrior_setting_simple("report.simple.columns")
    if existing_columns:
        if should_disable_simple_pager():
            existing_filter = get_taskwarrior_setting_simple("report.simple.filter")
            if existing_filter and "limit:page" in existing_filter:
                updated_filter = strip_limit_page(existing_filter)
                if updated_filter != existing_filter:
                    return configure_report(
                        "simple",
                        {"filter": updated_filter},
                        runner=run_simple_task_command,
                    )
        return True

    settings = get_report_settings(
        base_report,
        runner=run_simple_task_command,
    )
    if not settings:
        print("twh: unable to read base report settings.", file=sys.stderr)
        return False

    columns = settings.get("columns")
    if not columns:
        columns = get_taskwarrior_setting_simple(f"report.{base_report}.columns")

    if not columns:
        print("twh: base report has no columns defined.", file=sys.stderr)
        return False

    settings["columns"] = replace_description_column(columns)
    if should_disable_simple_pager() and settings.get("filter"):
        if "limit:page" in settings["filter"]:
            settings["filter"] = strip_limit_page(settings["filter"])
    return configure_report(
        "simple",
        settings,
        runner=run_simple_task_command,
    )


def run_simple_report(args: List[str]) -> int:
    """
    Run the Taskwarrior ``simple`` report with filters.

    The report is executed directly for speed; if it is missing, it is created
    from the default report and retried.

    Parameters
    ----------
    args : List[str]
        Filters and modifiers to pass through.

    Returns
    -------
    int
        Exit code from Taskwarrior.
    """
    if should_disable_simple_pager():
        args = strip_limit_page_tokens(args)
        existing_filter = get_taskwarrior_setting_simple("report.simple.filter")
        if existing_filter and "limit:page" in existing_filter:
            updated_filter = strip_limit_page(existing_filter)
            if updated_filter != existing_filter:
                configure_report(
                    "simple",
                    {"filter": updated_filter},
                    runner=run_simple_task_command,
                )
    task_args = [*args, "simple"]

    result = run_simple_task_command(task_args)
    if result.returncode == 0:
        return 0

    if get_taskwarrior_setting_simple("report.simple.columns"):
        return result.returncode

    base_report = get_default_report_name()
    if not ensure_simple_report(base_report):
        return 1

    retry = run_simple_task_command(task_args)
    return retry.returncode


def extract_blocks_tokens(args: List[str]) -> Tuple[List[str], List[str]]:
    """
    Extract blocks targets from task arguments.

    Parameters
    ----------
    args : List[str]
        Raw task arguments.

    Returns
    -------
    Tuple[List[str], List[str]]
        Arguments with blocks tokens removed and the block targets.

    Examples
    --------
    >>> extract_blocks_tokens(["blocks:32"])
    ([], ['32'])
    >>> extract_blocks_tokens(["blocks", "32", "project:work"])
    (['project:work'], ['32'])
    >>> extract_blocks_tokens(["blocks:32", "--", "blocks:99"])
    (['--', 'blocks:99'], ['32'])
    """
    cleaned: List[str] = []
    blocks: List[str] = []
    index = 0

    while index < len(args):
        arg = args[index]
        if arg == "--":
            cleaned.extend(args[index:])
            break
        if arg.startswith("blocks:"):
            value = arg[len("blocks:"):]
            if value:
                blocks.extend(split_csv(value))
            else:
                cleaned.append(arg)
            index += 1
            continue
        if arg == "blocks":
            if index + 1 < len(args) and args[index + 1] != "--":
                value = args[index + 1]
                if value:
                    blocks.extend(split_csv(value))
                    index += 2
                    continue
            cleaned.append(arg)
            index += 1
            continue
        cleaned.append(arg)
        index += 1

    return cleaned, blocks


def parse_created_task_id(output: str) -> Optional[str]:
    """
    Extract the created task ID from task add output.

    Parameters
    ----------
    output : str
        Standard output from ``task add``.

    Returns
    -------
    Optional[str]
        The created task ID if it can be determined.

    Examples
    --------
    >>> parse_created_task_id("Created task 45.")
    '45'
    >>> parse_created_task_id("No task created") is None
    True
    """
    if not output:
        return None
    match = re.search(r"Created task (\d+)", output)
    if match:
        return match.group(1)
    matches = re.findall(r"\b(\d+)\b", output)
    if matches:
        return matches[-1]
    return None


def run_task_command(
    args: List[str],
    capture_output: bool = False,
    stdin: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """
    Execute a taskwarrior command.

    Parameters
    ----------
    args : List[str]
        Taskwarrior arguments excluding the ``task`` executable.
    capture_output : bool, optional
        Whether to capture stdout/stderr (default: False).

    Returns
    -------
    subprocess.CompletedProcess
        Completed process result from ``subprocess.run``.
    """
    kwargs = {"check": False}
    if capture_output:
        kwargs.update({"capture_output": True, "text": True})
    if stdin is not None:
        kwargs["stdin"] = stdin
    task_args = apply_case_insensitive_overrides(args)
    task_args = apply_taskrc_overrides(task_args)
    return subprocess.run(["task", *task_args], **kwargs)


def exec_task_command(args: List[str]) -> int:
    """
    Replace the current process with a taskwarrior invocation.

    Parameters
    ----------
    args : List[str]
        Taskwarrior arguments excluding the executable.

    Returns
    -------
    int
        Unreachable return code placeholder.
    """
    task_args = apply_case_insensitive_overrides(args)
    task_args = apply_taskrc_overrides(task_args)
    os.execvp("task", ["task", *task_args])
    return 1


def get_taskwarrior_setting_simple(key: str) -> Optional[str]:
    """
    Read a Taskwarrior setting for ``twh simple``.

    Parameters
    ----------
    key : str
        Setting key to read.

    Returns
    -------
    Optional[str]
        Setting value or None if unavailable.
    """
    result = run_simple_task_command(["_get", key], capture_output=True)
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value if value else None


def merge_dependencies(existing: Sequence[str], additions: Iterable[str]) -> List[str]:
    """
    Merge dependency identifiers, preserving order and removing duplicates.

    Parameters
    ----------
    existing : Sequence[str]
        Existing dependency identifiers.
    additions : Iterable[str]
        Dependency identifiers to add.

    Returns
    -------
    List[str]
        Combined dependency identifiers.

    Examples
    --------
    >>> merge_dependencies(["a"], ["b", "a"])
    ['a', 'b']
    >>> merge_dependencies([], ["x", "y"])
    ['x', 'y']
    >>> merge_dependencies(["+a"], ["-b"])
    ['a', 'b']
    """
    merged: List[str] = []
    seen = set()
    for value in list(existing) + list(additions):
        dep = normalize_dependency_value(value)
        if not dep or dep in seen:
            continue
        merged.append(dep)
        seen.add(dep)
    return merged


def export_tasks(
    filter_args: Sequence[str],
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> Optional[List[Dict]]:
    """
    Export Taskwarrior moves for a filter.

    Parameters
    ----------
    filter_args : Sequence[str]
        Taskwarrior filter arguments.
    runner : Callable[..., subprocess.CompletedProcess], optional
        Runner for task commands (default: run_task_command).

    Returns
    -------
    Optional[List[Dict]]
        Parsed move payloads, or None on failure.
    """
    if runner is None:
        runner = run_task_command
    result = runner([*filter_args, "export"], capture_output=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return None
    return read_tasks_from_json(result.stdout or "")


def resolve_task_uuid(
    task_id: str,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> Optional[str]:
    """
    Resolve a move ID to a UUID using Taskwarrior export.

    Parameters
    ----------
    task_id : str
        Move ID to resolve.
    runner : Callable[..., subprocess.CompletedProcess], optional
        Runner for task commands (default: run_task_command).

    Returns
    -------
    Optional[str]
        Resolved UUID, or None if unavailable.
    """
    tasks = export_tasks([task_id], runner=runner)
    if tasks is None:
        return None
    for task in tasks:
        if isinstance(task, dict) and task.get("uuid"):
            return str(task["uuid"])
    return None


def apply_dependency_updates(
    targets: Iterable[str],
    additions: Iterable[str],
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    quiet: bool = False,
) -> int:
    """
    Apply dependency updates to target moves.

    Parameters
    ----------
    targets : Iterable[str]
        Target move identifiers or filters.
    additions : Iterable[str]
        Dependency identifiers to add.
    runner : Callable[..., subprocess.CompletedProcess], optional
        Runner for task commands (default: run_task_command).
    quiet : bool, optional
        Suppress Taskwarrior stdout when True (default: False).

    Returns
    -------
    int
        Exit code from Taskwarrior updates.
    """
    if runner is None:
        runner = run_task_command

    additions_list = merge_dependencies([], additions)
    if not additions_list:
        return 0

    exit_code = 0
    for target in targets:
        tasks = export_tasks([target], runner=runner)
        if tasks is None:
            exit_code = 1
            continue
        if not tasks:
            print(
                f"twh: no moves found for blocks target {target}.",
                file=sys.stderr,
            )
            exit_code = 1
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            uuid = task.get("uuid")
            if not uuid:
                print("twh: unable to resolve move uuid for blocks.", file=sys.stderr)
                exit_code = 1
                continue
            existing = parse_dependencies(task.get("depends"))
            merged = merge_dependencies(existing, additions_list)
            if merged == existing:
                continue
            depends_value = ",".join(merged)
            result = runner(
                [str(uuid), "modify", f"depends:{depends_value}"],
                capture_output=True,
            )
            if not quiet:
                for line in filter_modified_zero_lines(result.stdout):
                    print(line)
            if result.stderr:
                for line in filter_modified_zero_lines(result.stderr):
                    print(line, file=sys.stderr)
            if result.returncode != 0:
                exit_code = result.returncode
    return exit_code


def apply_blocks_relationship(
    argv: List[str],
    exec_task: Optional[Callable[[List[str]], int]] = None,
) -> int:
    """
    Translate blocks arguments into depends relationships.

    Parameters
    ----------
    argv : List[str]
        Taskwarrior arguments excluding the program name.
    exec_task : Callable[[List[str]], int] | None
        Optional exec-style runner to use when no blocks are present.

    Returns
    -------
    int
        Exit code from taskwarrior execution.
    """
    cleaned_args, blocks = extract_blocks_tokens(argv)
    if not blocks:
        if exec_task is not None:
            return exec_task(cleaned_args)
        result = run_task_command(cleaned_args)
        return result.returncode

    blocks = list(dict.fromkeys(blocks))

    if cleaned_args and cleaned_args[0] == "add":
        add_result = run_task_command(cleaned_args, capture_output=True)
        # Preserve Taskwarrior output when capture is required for parsing.
        if add_result.stdout:
            for line in filter_modified_zero_lines(add_result.stdout):
                print(line)
        if add_result.stderr:
            for line in filter_modified_zero_lines(add_result.stderr):
                print(line, file=sys.stderr)
        if add_result.returncode != 0:
            return add_result.returncode

        created_id = parse_created_task_id(add_result.stdout or "")
        if not created_id:
            print("twh: unable to parse created move id for blocks.", file=sys.stderr)
            return 1
        created_uuid = resolve_task_uuid(created_id)
        if not created_uuid:
            print("twh: unable to resolve created move uuid for blocks.", file=sys.stderr)
            return 1

        return apply_dependency_updates(blocks, [created_uuid])

    if "modify" not in cleaned_args:
        print("twh: blocks is only supported with add or modify.", file=sys.stderr)
        return 1

    modify_index = cleaned_args.index("modify")
    filter_args = cleaned_args[:modify_index]
    change_args = cleaned_args[modify_index + 1:]

    tasks = export_tasks(filter_args)
    if tasks is None:
        return 1
    blocking_uuids = [
        str(task["uuid"])
        for task in tasks
        if isinstance(task, dict) and task.get("uuid")
    ]
    blocking_uuids = merge_dependencies([], blocking_uuids)
    if not blocking_uuids:
        print("twh: no moves found to apply blocks.", file=sys.stderr)
        return 1

    if change_args:
        modify_result = run_task_command(cleaned_args)
        if modify_result.returncode != 0:
            return modify_result.returncode

    return apply_dependency_updates(blocks, blocking_uuids)


def get_graph_output_dir() -> Path:
    """
    Return the directory for default graph outputs, preferring /tmp.
    """
    tmp_dir = Path("/tmp")
    if tmp_dir.is_dir():
        return tmp_dir
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir
    except OSError:
        return Path(tempfile.gettempdir())


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_task_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    try:
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return datetime.strptime(value, "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def format_task_timestamp(value: Optional[str]) -> str:
    dt = parse_task_timestamp(value)
    if not dt:
        return ""
    return dt.strftime("%Y-%m-%d")


def format_age(value: Optional[str]) -> str:
    dt = parse_task_timestamp(value)
    if not dt:
        return ""
    now = datetime.now(tz=timezone.utc)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        seconds = 0
    units = [
        ("y", 365 * 86400),
        ("mo", 30 * 86400),
        ("wk", 7 * 86400),
        ("d", 86400),
        ("h", 3600),
        ("min", 60),
        ("s", 1),
    ]
    for suffix, unit in units:
        if seconds >= unit:
            return f"{seconds // unit}{suffix}"
    return "0s"


def label_for_column(column: str, label_map: Dict[str, str]) -> str:
    if column in label_map:
        return label_map[column]
    defaults = {
        "id": "ID",
        "uuid": "UUID",
        "description": "Description",
        "age": "Age",
        "project": "Project",
        "priority": "Pri",
        "due": "Due",
        "urgency": "Urg",
        "rank": "Rank",
        "score": "Score",
        "status": "Status",
        "tags": "Tags",
        "depends": "Deps",
    }
    return defaults.get(column, column.replace("_", " ").title())


def get_taskwarrior_columns_and_labels() -> Tuple[List[str], List[str]]:
    report = "list"
    default_cmd = get_taskwarrior_setting("default.command")
    if default_cmd:
        report = default_cmd.split()[0]

    columns_raw = split_csv(get_taskwarrior_setting(f"report.{report}.columns"))
    labels_raw = split_csv(get_taskwarrior_setting(f"report.{report}.labels"))

    if not columns_raw:
        columns_raw = ["description", "id", "age", "project", "priority", "due", "urgency"]
        labels_raw = ["Description", "ID", "Age", "Project", "Pri", "Due", "Urg"]

    columns = [col.lower() for col in columns_raw]
    label_map: Dict[str, str] = {}
    if labels_raw and len(labels_raw) == len(columns_raw):
        label_map = {col.lower(): label for col, label in zip(columns_raw, labels_raw)}

    ordered: List[str] = ["description"]
    if "id" in columns:
        ordered.append("id")
    ordered.extend([col for col in columns if col not in {"description", "id"}])
    # Drop duplicates while preserving order.
    seen = set()
    ordered = [col for col in ordered if not (col in seen or seen.add(col))]

    labels = [label_for_column(col, label_map) for col in ordered]
    return ordered, labels


def format_depends(depends_value: Optional[str], uuid_to_id: Dict[str, Optional[int]]) -> str:
    deps = parse_dependencies(depends_value)
    if not deps:
        return ""
    rendered = []
    for dep_uuid in deps:
        dep_id = uuid_to_id.get(dep_uuid)
        rendered.append(str(dep_id) if dep_id is not None else "?")
    return ",".join(rendered)


def format_task_field(task: Dict, column: str, uuid_to_id: Dict[str, Optional[int]]) -> str:
    col = column.lower()
    if col == "id":
        task_id = task.get("id")
        return str(task_id) if task_id is not None else ""
    if col == "description":
        return task.get("description", "")
    if col == "project":
        return task.get("project", "")
    if col == "priority":
        return task.get("priority", "")
    if col == "urgency":
        urgency = task.get("urgency")
        return f"{urgency:.2f}" if isinstance(urgency, (int, float)) else ""
    if col == "rank":
        rank_value = task.get("rank")
        if rank_value is None:
            return ""
        return str(rank_value)
    if col == "score":
        score_value = task.get("score")
        return f"{score_value:.2f}" if isinstance(score_value, (int, float)) else ""
    if col == "tags":
        tags = task.get("tags") or []
        return ",".join(tags) if isinstance(tags, list) else str(tags)
    if col == "depends":
        return format_depends(task.get("depends"), uuid_to_id)
    if col == "age":
        return format_age(task.get("entry"))
    if col in {"entry", "modified", "due", "wait", "scheduled", "until", "start", "end", "reviewed"}:
        return format_task_timestamp(task.get(col))
    if col == "status":
        return task.get("status", "")
    if col == "annotations":
        annotations = task.get("annotations") or []
        return str(len(annotations)) if isinstance(annotations, list) else ""
    if col == "uuid":
        return task.get("uuid", "")
    value = task.get(col)
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def apply_task_color(line: str, task: Dict) -> str:
    if task.get("start"):
        return f"\x1b[42m{line}\x1b[0m"
    priority = task.get("priority", "")
    if priority == "H":
        return f"\x1b[31m{line}\x1b[0m"
    if priority == "M":
        return f"\x1b[33m{line}\x1b[0m"
    if priority == "L":
        return f"\x1b[34m{line}\x1b[0m"
    return line


def get_tasks() -> List[Dict]:
    """
    Execute taskwarrior and return parsed task data using shared JSON parsing.

    Returns
    -------
    List[Dict]
        List of pending tasks as dictionaries.

    Raises
    ------
    SystemExit
        If taskwarrior command fails or output cannot be parsed.
    """
    try:
        return get_tasks_from_taskwarrior(status="pending")
    except subprocess.CalledProcessError:
        sys.exit(1)
    except json.JSONDecodeError:
        sys.exit(1)


def build_dependency_graph(tasks: List[Dict]) -> tuple[Dict[str, Dict], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Build dependency graph from tasks.

    Returns:
        - task_map: uuid -> task dict
        - depends_on: uuid -> set of uuids this task depends on
        - depended_by: uuid -> set of uuids that depend on this task
    """
    task_map = {t["uuid"]: t for t in tasks}
    depends_on = defaultdict(set)
    depended_by = defaultdict(set)

    for task in tasks:
        uuid = task["uuid"]
        for dep_uuid in parse_dependencies(task.get("depends")):
            depends_on[uuid].add(dep_uuid)
            depended_by[dep_uuid].add(uuid)

    return task_map, depends_on, depended_by


def format_task(task: Dict) -> str:
    """
    Format a task for simple tree output or tests.

    Parameters
    ----------
    task : Dict
        Task dictionary containing id, description, and urgency fields.

    Returns
    -------
    str
        Formatted task string in the format "[id] description (urgency: value)".
    """
    task_id = task.get("id", "?")
    description = task.get("description", "")
    urgency = task.get("urgency", 0)
    return f"[{task_id}] {description} (urgency: {urgency:.1f})"


def collect_tree_normal(task_map: Dict[str, Dict], depends_on: Dict[str, Set[str]],
                        depended_by: Dict[str, Set[str]]) -> List[Tuple[Dict, str]]:
    rows: List[Tuple[Dict, str]] = []
    visited: Set[str] = set()

    def add_task(uuid: str, ancestors_has_more: List[bool] = None,
                 has_more_siblings: bool = False):
        if ancestors_has_more is None:
            ancestors_has_more = []
        if uuid in visited:
            return
        if uuid not in task_map:
            return

        visited.add(uuid)
        task = task_map[uuid]
        prefix = build_tree_prefix(ancestors_has_more)
        rows.append((task, prefix))

        deps = depends_on.get(uuid, [])
        if deps:
            dep_list = sorted(
                deps,
                key=lambda u: task_map.get(u, {}).get("urgency", 0),
                reverse=True
            )
            for idx, dep_uuid in enumerate(dep_list):
                has_more = idx < len(dep_list) - 1
                add_task(dep_uuid, ancestors_has_more + [has_more_siblings], has_more)

    top_level = []
    for uuid in task_map:
        if not depended_by.get(uuid):
            top_level.append(uuid)
    top_level.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)

    for idx, uuid in enumerate(top_level):
        has_more = idx < len(top_level) - 1
        add_task(uuid, [], has_more)

    orphaned = [uuid for uuid in task_map if uuid not in visited]
    if orphaned:
        orphaned.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)
        for idx, uuid in enumerate(orphaned):
            has_more = idx < len(orphaned) - 1
            add_task(uuid, [], has_more)

    return rows


def collect_tree_reverse(task_map: Dict[str, Dict], depends_on: Dict[str, Set[str]],
                         depended_by: Dict[str, Set[str]]) -> List[Tuple[Dict, str]]:
    rows: List[Tuple[Dict, str]] = []

    def add_task(uuid: str, ancestors_has_more: List[bool] = None,
                 has_more_siblings: bool = False, ancestors: Set[str] = None):
        if ancestors_has_more is None:
            ancestors_has_more = []
        if ancestors is None:
            ancestors = set()
        if uuid in ancestors:
            return
        if uuid not in task_map:
            return

        task = task_map[uuid]
        prefix = build_tree_prefix(ancestors_has_more)
        rows.append((task, prefix))

        dependents = depended_by.get(uuid, [])
        if dependents:
            dep_list = sorted(
                dependents,
                key=lambda u: task_map.get(u, {}).get("urgency", 0),
                reverse=True
            )
            for idx, dep_uuid in enumerate(dep_list):
                has_more = idx < len(dep_list) - 1
                add_task(
                    dep_uuid,
                    ancestors_has_more + [has_more_siblings],
                    has_more,
                    ancestors | {uuid}
                )

    bottom_level = []
    for uuid in task_map:
        if not depends_on.get(uuid):
            bottom_level.append(uuid)
    bottom_level.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)

    for idx, uuid in enumerate(bottom_level):
        has_more = idx < len(bottom_level) - 1
        add_task(uuid, [], has_more)

    def has_bottom_level_dependency(uuid: str, visited: Set[str] = None) -> bool:
        if visited is None:
            visited = set()
        if uuid in visited:
            return False
        if uuid in bottom_level:
            return True
        visited.add(uuid)
        for dep_uuid in depends_on.get(uuid, []):
            if dep_uuid in task_map and has_bottom_level_dependency(dep_uuid, visited):
                return True
        return False

    orphaned = [uuid for uuid in task_map
                if uuid not in bottom_level and not has_bottom_level_dependency(uuid)]
    if orphaned:
        orphaned.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)
        for idx, uuid in enumerate(orphaned):
            has_more = idx < len(orphaned) - 1
            add_task(uuid, [], has_more)

    return rows


def render_task_table(rows: List[Tuple[Dict, str]],
                      columns: List[str],
                      labels: List[str],
                      uuid_to_id: Dict[str, Optional[int]]) -> List[str]:
    if not rows:
        return []

    normalized_columns = [col.lower() for col in columns]
    if len(labels) != len(normalized_columns):
        labels = [label_for_column(col, {}) for col in normalized_columns]
    widths = [len(label) for label in labels]
    rendered_rows: List[Tuple[Dict, List[str]]] = []

    for task, prefix in rows:
        values: List[str] = []
        for col in normalized_columns:
            if col == "description":
                value = f"{prefix}{format_task_field(task, col, uuid_to_id)}"
            else:
                value = format_task_field(task, col, uuid_to_id)
            values.append(value)
        for idx, value in enumerate(values):
            widths[idx] = max(widths[idx], len(value))
        rendered_rows.append((task, values))

    header = "  ".join(label.ljust(widths[idx]) for idx, label in enumerate(labels))
    separator = "  ".join("-" * width for width in widths)

    lines = [header, separator]
    for task, values in rendered_rows:
        line = "  ".join(values[idx].ljust(widths[idx]) for idx in range(len(values)))
        lines.append(apply_task_color(line, task))

    return lines


def build_task_table_lines(tasks: List[Dict], reverse: bool = False,
                           columns: Optional[List[str]] = None,
                           labels: Optional[List[str]] = None) -> List[str]:
    task_map, depends_on, depended_by = build_dependency_graph(tasks)
    rows = collect_tree_reverse(task_map, depends_on, depended_by) if reverse else \
        collect_tree_normal(task_map, depends_on, depended_by)

    if not columns or not labels:
        columns, labels = get_taskwarrior_columns_and_labels()

    uuid_to_id = {t.get("uuid"): t.get("id") for t in tasks if t.get("uuid")}
    return render_task_table(rows, columns, labels, uuid_to_id)


def print_tree_normal(task_map: Dict[str, Dict], depends_on: Dict[str, Set[str]],
                     depended_by: Dict[str, Set[str]], indent: str = "  ",
                     blank_line_between_top_level: bool = True):
    """
    Print dependency list with unblocked tasks at top level.

    Moves that are not blocked by other moves are shown at the top level,
    with their dependencies (blocking moves) indented below them.

    Parameters
    ----------
    task_map : Dict[str, Dict]
        Mapping from UUID to task dictionary.
    depends_on : Dict[str, Set[str]]
        Mapping from UUID to set of UUIDs this task depends on.
    depended_by : Dict[str, Set[str]]
        Mapping from UUID to set of UUIDs that depend on this task.
    indent : str, optional
        Indentation string for each level (default: "  ").
    blank_line_between_top_level : bool
        Whether to print a blank line between top-level tasks.
    """
    visited = set()

    def print_task_and_deps(uuid: str, ancestors_has_more: List[bool] = None,
                            has_more_siblings: bool = False):
        """Recursively print a task and its dependencies."""
        if ancestors_has_more is None:
            ancestors_has_more = []
        if uuid in visited:
            return

        if uuid not in task_map:
            # Dependency task might be completed/deleted - skip it
            return

        visited.add(uuid)
        task = task_map[uuid]
        prefix = build_tree_prefix(ancestors_has_more)
        print(f"{prefix}{format_task(task)}")

        # Print dependencies (tasks this one depends on)
        deps = depends_on.get(uuid, [])
        if deps:
            dep_list = sorted(
                deps,
                key=lambda u: task_map.get(u, {}).get("urgency", 0),
                reverse=True
            )
            for idx, dep_uuid in enumerate(dep_list):
                has_more = idx < len(dep_list) - 1
                print_task_and_deps(dep_uuid, ancestors_has_more + [has_more_siblings], has_more)

    # Find top-level tasks (tasks that no other pending task depends on)
    top_level = []
    for uuid in task_map:
        # A task is top-level if no other pending task depends on it
        if not depended_by.get(uuid):
            top_level.append(uuid)

    # Sort by urgency (descending)
    top_level.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)

    for idx, uuid in enumerate(top_level):
        has_more = idx < len(top_level) - 1
        print_task_and_deps(uuid, [], has_more)
        if blank_line_between_top_level:
            print()  # Blank line between top-level tasks

    # Handle orphaned tasks (e.g., circular dependencies)
    # These are tasks that weren't visited because they're in dependency cycles
    orphaned = [uuid for uuid in task_map if uuid not in visited]
    if orphaned:
        orphaned.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)
        for idx, uuid in enumerate(orphaned):
            has_more = idx < len(orphaned) - 1
            print_task_and_deps(uuid, [], has_more)
            if blank_line_between_top_level:
                print()


def print_tree_reverse(task_map: Dict[str, Dict], depends_on: Dict[str, Set[str]],
                      depended_by: Dict[str, Set[str]], indent: str = "  ",
                      blank_line_between_top_level: bool = True):
    """
    Print dependency list with blocking tasks at top level.

    Moves that have no dependencies (blocking moves) are shown at the top level,
    with moves that depend on them indented below.

    Parameters
    ----------
    task_map : Dict[str, Dict]
        Mapping from UUID to task dictionary.
    depends_on : Dict[str, Set[str]]
        Mapping from UUID to set of UUIDs this task depends on.
    depended_by : Dict[str, Set[str]]
        Mapping from UUID to set of UUIDs that depend on this task.
    indent : str, optional
        Indentation string for each level (default: "  ").
    blank_line_between_top_level : bool
        Whether to print a blank line between top-level tasks.
    """

    def print_task_and_dependents(uuid: str, ancestors_has_more: List[bool] = None,
                                  has_more_siblings: bool = False,
                                  ancestors: Set[str] = None):
        """Recursively print a task and tasks that depend on it."""
        if ancestors_has_more is None:
            ancestors_has_more = []
        if ancestors is None:
            ancestors = set()

        # Prevent infinite recursion by checking if we're already in the current path
        if uuid in ancestors:
            return

        if uuid not in task_map:
            # Task might be completed/deleted - skip it
            return

        task = task_map[uuid]
        prefix = build_tree_prefix(ancestors_has_more)
        print(f"{prefix}{format_task(task)}")

        # Print dependents (tasks that depend on this one)
        dependents = depended_by.get(uuid, [])
        if dependents:
            dep_list = sorted(
                dependents,
                key=lambda u: task_map.get(u, {}).get("urgency", 0),
                reverse=True
            )
            for idx, dep_uuid in enumerate(dep_list):
                has_more = idx < len(dep_list) - 1
                # Add current uuid to ancestors before recursing
                print_task_and_dependents(
                    dep_uuid,
                    ancestors_has_more + [has_more_siblings],
                    has_more,
                    ancestors | {uuid}
                )

    # Find bottom-level tasks (tasks that don't depend on any other pending tasks)
    bottom_level = []
    for uuid in task_map:
        deps = depends_on.get(uuid, set())
        # Include if no dependencies, or all dependencies are on completed/deleted tasks
        pending_deps = [d for d in deps if d in task_map]
        if not pending_deps:
            bottom_level.append(uuid)

    # Sort by urgency (descending)
    bottom_level.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)

    for idx, uuid in enumerate(bottom_level):
        has_more = idx < len(bottom_level) - 1
        print_task_and_dependents(uuid, [], has_more)
        if blank_line_between_top_level:
            print()  # Blank line between top-level tasks

    # Handle orphaned tasks (e.g., circular dependencies)
    # These are tasks that have dependencies but those dependencies form cycles,
    # so they never got printed as descendants of bottom-level tasks
    # To detect these, check if any of the task's dependencies are in bottom_level
    def has_bottom_level_dependency(uuid: str, visited: Set[str] = None) -> bool:
        """Check if a task's dependency chain eventually reaches a bottom-level task."""
        if visited is None:
            visited = set()
        if uuid in visited:
            return False  # Circular dependency
        if uuid in bottom_level:
            return True
        visited.add(uuid)
        # Check if any of this task's dependencies reach bottom level
        for dep_uuid in depends_on.get(uuid, []):
            if dep_uuid in task_map and has_bottom_level_dependency(dep_uuid, visited):
                return True
        return False

    orphaned = [uuid for uuid in task_map
                if uuid not in bottom_level and not has_bottom_level_dependency(uuid)]
    if orphaned:
        orphaned.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)
        for idx, uuid in enumerate(orphaned):
            has_more = idx < len(orphaned) - 1
            print_task_and_dependents(uuid, [], has_more)
            if blank_line_between_top_level:
                print()


def list_tasks(
    mode: Optional[str] = None,
    reverse: bool = False,
) -> None:
    """
    Display Taskwarrior moves in hierarchical dependency list view.

    Uses Taskwarrior JSON export data with a custom formatter that preserves
    hierarchy, description-first layout, and Taskwarrior-like colors.
    """
    tasks = get_tasks()

    if not tasks:
        print("No pending tasks found.")
        return

    if mode:
        if mode != "reverse":
            raise ValueError("Only 'reverse' is supported as a mode.")
        reverse = True

    lines = build_task_table_lines(tasks, reverse=reverse)
    for line in lines:
        print(line)


def list_reverse():
    """
    Alias for `twh list reverse`.
    """
    list_tasks(mode="reverse")


def tree_alias(reverse: bool = False) -> None:
    """
    Alias for `twh list`.
    """
    list_tasks(reverse=reverse)


def simple(args: Optional[List[str]] = None) -> int:
    """
    Display the Taskwarrior simple report with annotation counts.

    Parameters
    ----------
    args : Optional[List[str]]
        Filters and modifiers to pass through.

    Returns
    -------
    int
        Exit code from Taskwarrior.
    """
    if args is None:
        args = []
    return run_simple_report(args)


def graph(
    mode: Optional[str] = None,
    reverse: bool = False,
    png: Optional[str] = None,
    svg: Optional[str] = None,
    ascii_only: bool = False,
    edges: bool = False,
    rankdir: str = "LR",
) -> None:
    """
    Generate a Graphviz dependency graph with ASCII fallback.
    """
    from .graph import (
        ascii_forest,
        build_dependency_edges,
        format_edge_list,
        generate_dot,
        render_graphviz,
    )
    from .renderer import open_file, open_in_browser

    if mode:
        if mode != "reverse":
            raise ValueError("Only 'reverse' is supported as a mode.")
        reverse = True

    tasks = get_tasks_from_taskwarrior()
    if not tasks:
        print("No pending tasks found.")
        return

    rankdir = rankdir.strip().upper()
    if rankdir not in {"LR", "TB", "BT", "RL"}:
        raise ValueError("rankdir must be one of LR, TB, BT, or RL.")

    output_dir = get_graph_output_dir()
    png_path = Path(png) if png else None
    svg_path = Path(svg) if svg else None
    if not ascii_only and not png_path and not svg_path:
        svg_path = output_dir / "tasks-graph.svg"

    edges_list, by_uuid = build_dependency_edges(tasks, reverse=reverse)

    if edges:
        for line in format_edge_list(edges_list):
            print(line)

    wants_render = bool(png_path or svg_path)
    rendered = False
    render_error = None

    if wants_render and not ascii_only:
        dot_source = generate_dot(by_uuid, edges_list, rankdir=rankdir)
        rendered, render_error = render_graphviz(dot_source, png_path, svg_path)
        if rendered:
            if png_path:
                print(f"Generated Graphviz PNG: {png_path}")
                open_file(png_path)
            if svg_path:
                print(f"Generated Graphviz SVG: {svg_path}")
                open_in_browser(svg_path)
        elif render_error:
            print(f"twh: {render_error}", file=sys.stderr)

    if ascii_only or not rendered:
        for line in ascii_forest(edges_list, by_uuid):
            print(line)


def build_app():
    """
    Build the Typer app lazily to keep fast-path imports light.

    Returns
    -------
    typer.Typer
        Configured Typer application for the twh CLI.
    """
    import typer

    app = typer.Typer(help="Hierarchical views of Taskwarrior moves")

    @app.command("help")
    def help_cmd():
        """
        Print a brief reminder of twh-specific commands.

        Returns
        -------
        None
            This command prints the twh help summary.
        """
        for line in get_twh_help_lines():
            print(line)

    @app.command("list")
    def list_cmd(
        mode: Optional[str] = typer.Argument(
            None,
            help="Use 'reverse' for blocker-first view.",
        ),
        reverse: bool = typer.Option(
            False,
            "--reverse",
            "-r",
            help="Show most-depended-upon moves at top level (reverse view)",
        ),
    ):
        try:
            list_tasks(mode=mode, reverse=reverse)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    @app.command("reverse")
    def reverse_cmd():
        list_reverse()

    @app.command("tree")
    def tree_cmd(
        reverse: bool = typer.Option(
            False,
            "--reverse",
            "-r",
            help="Show most-depended-upon moves at top level (reverse view)",
        ),
    ):
        tree_alias(reverse=reverse)

    @app.command(
        "start",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def start_cmd(ctx: typer.Context):
        from . import time_log as time_module

        try:
            exit_code = time_module.run_start(list(ctx.args))
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @app.command(
        "stop",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def stop_cmd(ctx: typer.Context):
        from . import time_log as time_module

        try:
            exit_code = time_module.run_stop(list(ctx.args))
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    time_app = typer.Typer(help="Report or edit time logs.")

    @time_app.callback(invoke_without_command=True)
    def time_callback(
        ctx: typer.Context,
        by: str = typer.Option(
            "project",
            "--by",
            help="Group by task/project/tag/mode/total.",
        ),
        period: str = typer.Option(
            "week",
            "--period",
            help="Period bucket: day/week/month/year/range.",
        ),
        start: Optional[str] = typer.Option(
            None,
            "--from",
            help="Filter start date/time (YYYY-MM-DD or ISO).",
        ),
        end: Optional[str] = typer.Option(
            None,
            "--to",
            help="Filter end date/time (YYYY-MM-DD or ISO).",
        ),
    ):
        if ctx.invoked_subcommand is not None:
            return
        from . import time_log as time_module

        try:
            exit_code = time_module.run_time_report(
                group_by=by,
                period=period,
                range_start=start,
                range_end=end,
            )
        except ValueError as exc:
            print(f"twh: time report failed: {exc}", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @time_app.command("report")
    def time_report_cmd(
        by: str = typer.Option(
            "project",
            "--by",
            help="Group by task/project/tag/mode/total.",
        ),
        period: str = typer.Option(
            "week",
            "--period",
            help="Period bucket: day/week/month/year/range.",
        ),
        start: Optional[str] = typer.Option(
            None,
            "--from",
            help="Filter start date/time (YYYY-MM-DD or ISO).",
        ),
        end: Optional[str] = typer.Option(
            None,
            "--to",
            help="Filter end date/time (YYYY-MM-DD or ISO).",
        ),
    ):
        from . import time_log as time_module

        try:
            exit_code = time_module.run_time_report(
                group_by=by,
                period=period,
                range_start=start,
                range_end=end,
            )
        except ValueError as exc:
            print(f"twh: time report failed: {exc}", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @time_app.command("entries")
    def time_entries_cmd(
        start: Optional[str] = typer.Option(
            None,
            "--from",
            help="Filter start date/time (YYYY-MM-DD or ISO).",
        ),
        end: Optional[str] = typer.Option(
            None,
            "--to",
            help="Filter end date/time (YYYY-MM-DD or ISO).",
        ),
        limit: int = typer.Option(
            50,
            "--limit",
            help="Maximum entries to list.",
        ),
    ):
        from . import time_log as time_module

        try:
            exit_code = time_module.run_time_entries(
                range_start=start,
                range_end=end,
                limit=limit,
            )
        except ValueError as exc:
            print(f"twh: time entries failed: {exc}", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @time_app.command("edit")
    def time_edit_cmd(
        entry_id: int = typer.Argument(..., help="Entry id to edit."),
        start: Optional[str] = typer.Option(
            None,
            "--start",
            help="New start date/time.",
        ),
        end: Optional[str] = typer.Option(
            None,
            "--end",
            help="New end date/time.",
        ),
        duration: Optional[str] = typer.Option(
            None,
            "--duration",
            help="New duration (e.g., 1.5h, 90m).",
        ),
    ):
        from . import time_log as time_module

        try:
            exit_code = time_module.run_time_edit(
                entry_id=entry_id,
                start=start,
                end=end,
                duration=duration,
            )
        except ValueError as exc:
            print(f"twh: time edit failed: {exc}", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    app.add_typer(time_app, name="time")

    @app.command(
        "simple",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def simple_cmd(ctx: typer.Context):
        try:
            exit_code = run_simple_report(ctx.args)
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @app.command("graph")
    def graph_cmd(
        mode: Optional[str] = typer.Argument(
            None,
            help="Use 'reverse' for blocker-first view.",
        ),
        reverse: bool = typer.Option(
            False,
            "--reverse",
            "-r",
            help="Show blockers first by reversing edge direction",
        ),
        png: Optional[str] = typer.Option(
            None,
            "--png",
            help="Write PNG graph to this path",
        ),
        svg: Optional[str] = typer.Option(
            None,
            "--svg",
            help="Write SVG graph to this path",
        ),
        ascii_only: bool = typer.Option(
            False,
            "--ascii",
            help="Force ASCII output even when Graphviz is available",
        ),
        edges: bool = typer.Option(
            False,
            "--edges",
            help="Print the edge list (uuid -> uuid)",
        ),
        rankdir: str = typer.Option(
            "LR",
            "--rankdir",
            help="Graph layout direction (LR, TB, BT, RL)",
        ),
    ):
        try:
            graph(
                mode=mode,
                reverse=reverse,
                png=png,
                svg=svg,
                ascii_only=ascii_only,
                edges=edges,
                rankdir=rankdir,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    @app.command(
        "dominance",
        help="Collect dominance ordering for moves.",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def dominance_cmd(ctx: typer.Context):
        from . import dominance as dominance_module

        try:
            exit_code = dominance_module.run_dominance(filters=list(ctx.args))
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @app.command(
        "criticality",
        help="Collect time-criticality ordering for moves.",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def criticality_cmd(ctx: typer.Context):
        from . import criticality as criticality_module

        try:
            exit_code = criticality_module.run_criticality(filters=list(ctx.args))
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @app.command(
        "ondeck",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def ondeck_cmd(
        ctx: typer.Context,
        mode: Optional[str] = typer.Option(
            None,
            "--mode",
            help="Current mode context (analysis/editorial/etc).",
        ),
        limit: int = typer.Option(
            20,
            "--limit",
            help="Max moves to list when showing missing metadata.",
        ),
        top: int = typer.Option(
            25,
            "--top",
            help="Show top N candidate moves.",
        ),
        strict_mode: bool = typer.Option(
            False,
            "--strict-mode",
            help="Only keep moves with matching modes.",
        ),
        include_dominated: bool = typer.Option(
            True,
            "--include-dominated",
            help="Include moves dominated by other moves.",
        ),
    ):
        from . import review as review_module

        try:
            exit_code = review_module.run_ondeck(
                mode=mode,
                limit=limit,
                top=top,
                strict_mode=strict_mode,
                include_dominated=include_dominated,
                filters=list(ctx.args),
            )
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @app.command(
        "defer",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def defer_cmd(
        ctx: typer.Context,
        mode: Optional[str] = typer.Option(
            None,
            "--mode",
            help="Current mode context (analysis/editorial/etc).",
        ),
        strict_mode: bool = typer.Option(
            False,
            "--strict-mode",
            help="Only keep moves with matching modes.",
        ),
        include_dominated: bool = typer.Option(
            True,
            "--include-dominated",
            help="Include moves dominated by other moves.",
        ),
    ):
        from . import defer as defer_module

        try:
            exit_code = defer_module.run_defer(
                mode=mode,
                strict_mode=strict_mode,
                include_dominated=include_dominated,
                args=list(ctx.args),
            )
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @app.command(
        "resurface",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def resurface_cmd(
        ctx: typer.Context,
        mode: Optional[str] = typer.Option(
            None,
            "--mode",
            help="Current mode context (analysis/editorial/etc).",
        ),
        strict_mode: bool = typer.Option(
            False,
            "--strict-mode",
            help="Only keep moves with matching modes.",
        ),
        include_dominated: bool = typer.Option(
            True,
            "--include-dominated",
            help="Include moves dominated by other moves.",
        ),
    ):
        from . import defer as defer_module

        try:
            exit_code = defer_module.run_resurface(
                command_name="resurface",
                mode=mode,
                strict_mode=strict_mode,
                include_dominated=include_dominated,
                args=list(ctx.args),
            )
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @app.command("diagnose")
    def diagnose_cmd(
        selector: Optional[str] = typer.Argument(
            None,
            help="Move ID or UUID prefix to diagnose.",
        ),
        mode: Optional[str] = typer.Option(
            None,
            "--mode",
            help="Current mode context (analysis/editorial/etc).",
        ),
        strict_mode: bool = typer.Option(
            False,
            "--strict-mode",
            help="Only keep moves with matching modes.",
        ),
        include_dominated: bool = typer.Option(
            True,
            "--include-dominated",
            help="Include moves dominated by other moves.",
        ),
    ):
        from . import diagnose as diagnose_module

        try:
            exit_code = diagnose_module.run_diagnose(
                selector=selector,
                mode=mode,
                strict_mode=strict_mode,
                include_dominated=include_dominated,
            )
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @app.command(
        "option",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def option_cmd(
        ctx: typer.Context,
        apply: bool = typer.Option(
            False,
            "--apply",
            help="Write opt_auto values to moves in scope.",
        ),
        include_rated: bool = typer.Option(
            False,
            "--include-rated",
            help="Include moves with manual opt_human ratings in the output.",
        ),
        limit: int = typer.Option(
            25,
            "--limit",
            help="Maximum number of moves to display.",
        ),
        ridge: float = typer.Option(
            1.0,
            "--ridge",
            help="Ridge regularization strength for calibration.",
        ),
    ):
        from . import option_value as option_module

        try:
            exit_code = option_module.run_option_value(
                filters=list(ctx.args),
                apply=apply,
                include_rated=include_rated,
                limit=limit,
                ridge=ridge,
            )
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    @app.command(
        "calibrate",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def calibrate_cmd(
        ctx: typer.Context,
        pairs: int = typer.Option(
            10,
            "--pairs",
            help="Number of pairwise precedence comparisons to prompt.",
        ),
        alpha: float = typer.Option(
            0.15,
            "--alpha",
            help="Learning rate for precedence calibration.",
        ),
        ridge: float = typer.Option(
            1.0,
            "--ridge",
            help="Ridge regularization strength for option value calibration.",
        ),
        apply: bool = typer.Option(
            True,
            "--apply/--no-apply",
            help="Apply opt_auto values after calibration.",
        ),
    ):
        from . import calibrate as calibrate_module

        try:
            exit_code = calibrate_module.run_calibrate(
                pairs=pairs,
                alpha=alpha,
                ridge=ridge,
                apply=apply,
                filters=list(ctx.args),
            )
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            raise typer.Exit(code=1)
        raise typer.Exit(code=exit_code)

    return app


TWH_HELP_HEADER = "twh commands:"
TWH_HELP_FOOTER = "Use task help for Taskwarrior commands."
TWH_HELP_ENTRIES: Tuple[Tuple[str, str], ...] = (
    ("add", "Interactive add wizard for a move."),
    ("list", "Hierarchical list of moves."),
    ("reverse", "Blocker-first list."),
    ("tree", "Dependency tree of moves."),
    ("graph", "Graph view of move dependencies."),
    ("simple", "Taskwarrior report with annotation counts."),
    ("ondeck", "Rank moves and collect tie metadata."),
    ("defer", "Alias for resurface."),
    ("diagnose", "Stuck-move helper wizard."),
    ("dominance", "Pairwise dominance prompts for moves."),
    ("criticality", "Pairwise time-criticality prompts for moves."),
    ("resurface", "Resurface moves by delaying their start time."),
    ("option", "Estimate opt_auto values."),
    ("calibrate", "Calibrate precedence/option weights."),
    ("start", "Start time tracking for a move."),
    ("stop", "Stop time tracking for a move."),
    ("time", "Report or edit time logs."),
    ("help", "Show this help."),
)
TWH_COMMANDS: Set[str] = {command for command, _ in TWH_HELP_ENTRIES}
TWH_HELP_ARGS = {"-h", "--help", "--install-completion", "--show-completion"}


def get_twh_help_lines() -> List[str]:
    """
    Build the help text lines for twh-specific commands.

    Returns
    -------
    list[str]
        Lines to print for the `twh help` command.

    Examples
    --------
    >>> lines = get_twh_help_lines()
    >>> lines[0]
    'twh commands:'
    >>> any(line.strip().startswith("ondeck") for line in lines)
    True
    """
    max_width = max(len(command) for command, _ in TWH_HELP_ENTRIES)
    sorted_entries = sorted(TWH_HELP_ENTRIES, key=lambda entry: entry[0])
    lines = [TWH_HELP_HEADER]
    for command, description in sorted_entries:
        lines.append(f"  {command:<{max_width}}  {description}")
    lines.append(TWH_HELP_FOOTER)
    return lines


def should_delegate_to_task(argv: List[str]) -> bool:
    """
    Decide whether to forward a command to Taskwarrior.

    Parameters
    ----------
    argv : List[str]
        Command-line arguments excluding the program name.

    Returns
    -------
    bool
        True when the command should be executed by ``task``.

    Examples
    --------
    >>> should_delegate_to_task([])
    True
    >>> should_delegate_to_task(["project:work"])
    True
    >>> should_delegate_to_task(["add", "Next move"])
    False
    >>> should_delegate_to_task(["list"])
    False
    >>> should_delegate_to_task(["help"])
    False
    >>> should_delegate_to_task(["--help"])
    False
    """
    if not argv:
        return True
    first_arg = argv[0]
    if first_arg == "add":
        return False
    return first_arg not in TWH_COMMANDS and first_arg not in TWH_HELP_ARGS


def has_help_args(argv: List[str]) -> bool:
    """
    Return True when CLI arguments include a help flag.

    Parameters
    ----------
    argv : List[str]
        Command-line arguments excluding the program name.

    Returns
    -------
    bool
        True when a help flag is present.
    """
    return any(arg in TWH_HELP_ARGS for arg in argv)


def main():
    """
    Entry point for the twh command.

    If the command is not recognized, delegate to Taskwarrior.
    """
    enable_line_editing()
    argv = sys.argv[1:]
    if argv and argv[0] == "add":
        if has_help_args(argv):
            try:
                exit_code = apply_blocks_relationship(argv, exec_task=exec_task_command)
            except FileNotFoundError:
                print("Error: `task` command not found.", file=sys.stderr)
                sys.exit(1)
            sys.exit(exit_code)
        try:
            exit_code = run_interactive_add(argv[1:])
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            sys.exit(1)
        sys.exit(exit_code)
    if should_delegate_to_task(argv):
        argv, message = apply_context_to_add_args(argv)
        if message:
            print(message)
            sys.stdout.flush()
        try:
            exit_code = apply_blocks_relationship(argv, exec_task=exec_task_command)
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            sys.exit(1)
        sys.exit(exit_code)
    if argv and argv[0] == "simple" and not has_help_args(argv):
        try:
            exit_code = run_simple_report(argv[1:])
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            sys.exit(1)
        sys.exit(exit_code)

    app = build_app()
    app()


if __name__ == "__main__":
    main()

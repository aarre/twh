#!/usr/bin/env python3
"""
Hierarchical view of Taskwarrior tasks by dependency.
"""

import json
import os
import re
import shlex
import tempfile
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

import typer

from .taskwarrior import (
    get_tasks_from_taskwarrior,
    parse_dependencies,
    read_tasks_from_json,
)


def build_tree_prefix(ancestor_has_more: List[bool]) -> str:
    parts = []
    for has_more in ancestor_has_more:
        parts.append("|  " if has_more else "   ")
    parts.append("+- ")
    return "".join(parts)


def get_taskwarrior_setting(key: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["task", "_get", key],
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
    return parse_default_command(get_taskwarrior_setting("default.command"))


def get_default_command_tokens() -> Tuple[str, List[str]]:
    """
    Return the default report name and filters from Taskwarrior.

    Returns
    -------
    Tuple[str, List[str]]
        Report name and default filters.
    """
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


def get_report_settings(report_name: str) -> Dict[str, str]:
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
    result = run_task_command(
        ["show", f"report.{report_name}"],
        capture_output=True,
    )
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return {}
    return parse_report_settings(report_name, result.stdout or "")


def configure_report(report_name: str, settings: Dict[str, str]) -> bool:
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
    for key, value in settings.items():
        if not value:
            continue
        result = run_task_command(
            ["config", f"report.{report_name}.{key}", value],
            capture_output=True,
        )
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
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
    if get_taskwarrior_setting("report.simple.columns"):
        return True

    settings = get_report_settings(base_report)
    if not settings:
        print("twh: unable to read base report settings.", file=sys.stderr)
        return False

    columns = settings.get("columns")
    if not columns:
        columns = get_taskwarrior_setting(f"report.{base_report}.columns")

    if not columns:
        print("twh: base report has no columns defined.", file=sys.stderr)
        return False

    settings["columns"] = replace_description_column(columns)
    return configure_report("simple", settings)


def run_simple_report(args: List[str]) -> int:
    """
    Run the Taskwarrior ``simple`` report with filters.

    Parameters
    ----------
    args : List[str]
        Filters and modifiers to pass through.

    Returns
    -------
    int
        Exit code from Taskwarrior.
    """
    base_report, default_filters = get_default_command_tokens()
    if not ensure_simple_report(base_report):
        return 1
    task_args = [*default_filters, *args, "simple"]
    if should_disable_simple_pager():
        task_args = ["rc.pager=cat", *task_args]
    result = run_task_command(task_args)
    return result.returncode


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
    return subprocess.run(["task", *args], **kwargs)


def apply_blocks_relationship(argv: List[str]) -> int:
    """
    Translate blocks arguments into depends relationships.

    Parameters
    ----------
    argv : List[str]
        Taskwarrior arguments excluding the program name.

    Returns
    -------
    int
        Exit code from taskwarrior execution.
    """
    cleaned_args, blocks = extract_blocks_tokens(argv)
    if not blocks:
        result = run_task_command(argv)
        return result.returncode

    blocks = list(dict.fromkeys(blocks))

    if cleaned_args and cleaned_args[0] == "add":
        add_result = run_task_command(cleaned_args, capture_output=True)
        # Preserve Taskwarrior output when capture is required for parsing.
        if add_result.stdout:
            print(add_result.stdout, end="")
        if add_result.stderr:
            print(add_result.stderr, end="", file=sys.stderr)
        if add_result.returncode != 0:
            return add_result.returncode

        created_id = parse_created_task_id(add_result.stdout or "")
        if not created_id:
            print("twh: unable to parse created task id for blocks.", file=sys.stderr)
            return 1

        exit_code = add_result.returncode
        for target in blocks:
            result = run_task_command([target, "modify", f"depends:+{created_id}"])
            if result.returncode != 0:
                exit_code = result.returncode
        return exit_code

    if "modify" not in cleaned_args:
        print("twh: blocks is only supported with add or modify.", file=sys.stderr)
        return 1

    modify_index = cleaned_args.index("modify")
    filter_args = cleaned_args[:modify_index]
    change_args = cleaned_args[modify_index + 1:]

    export_result = run_task_command([*filter_args, "export"], capture_output=True)
    if export_result.returncode != 0:
        if export_result.stderr:
            print(export_result.stderr, end="", file=sys.stderr)
        return export_result.returncode

    tasks = read_tasks_from_json(export_result.stdout or "")
    blocking_uuids = [
        task["uuid"] for task in tasks
        if isinstance(task, dict) and task.get("uuid")
    ]
    blocking_uuids = list(dict.fromkeys(blocking_uuids))
    if not blocking_uuids:
        print("twh: no tasks found to apply blocks.", file=sys.stderr)
        return 1

    if change_args:
        modify_result = run_task_command(cleaned_args)
        if modify_result.returncode != 0:
            return modify_result.returncode

    exit_code = 0
    for target in blocks:
        for uuid in blocking_uuids:
            result = run_task_command([target, "modify", f"depends:+{uuid}"])
            if result.returncode != 0:
                exit_code = result.returncode
    return exit_code


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

    Tasks that are not blocked by other tasks are shown at the top level,
    with their dependencies (blocking tasks) indented below them.

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

    Tasks that have no dependencies (blocking tasks) are shown at the top level,
    with tasks that depend on them indented below.

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


app = typer.Typer(help="Hierarchical views of Taskwarrior tasks")


@app.command("list")
def list_tasks(
    mode: Optional[str] = typer.Argument(
        None,
        help="Use 'reverse' for blocker-first view."
    ),
    reverse: bool = typer.Option(
        False,
        "--reverse",
        "-r",
        help="Show most-depended-upon tasks at top level (reverse view)"
    )
):
    """
    Display Taskwarrior tasks in hierarchical dependency list view.

    Uses Taskwarrior JSON export data with a custom formatter that preserves
    hierarchy, description-first layout, and Taskwarrior-like colors.
    """
    tasks = get_tasks()

    if not tasks:
        print("No pending tasks found.")
        return

    if mode:
        if mode != "reverse":
            raise typer.BadParameter("Only 'reverse' is supported as a mode.")
        reverse = True

    lines = build_task_table_lines(tasks, reverse=reverse)
    for line in lines:
        print(line)


@app.command("reverse")
def list_reverse():
    """
    Alias for `twh list reverse`.
    """
    list_tasks(mode="reverse")


@app.command("tree")
def tree_alias(
    reverse: bool = typer.Option(
        False,
        "--reverse",
        "-r",
        help="Show most-depended-upon tasks at top level (reverse view)"
    )
):
    """
    Alias for `twh list`.
    """
    list_tasks(reverse=reverse)


@app.command(
    "simple",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def simple(ctx: typer.Context):
    """
    Display the Taskwarrior simple report with annotation counts.
    """
    try:
        exit_code = run_simple_report(ctx.args)
    except FileNotFoundError:
        print("Error: `task` command not found.", file=sys.stderr)
        raise typer.Exit(code=1)
    raise typer.Exit(code=exit_code)


@app.command()
def graph(
    mode: Optional[str] = typer.Argument(
        None,
        help="Use 'reverse' for blocker-first view."
    ),
    reverse: bool = typer.Option(
        False,
        "--reverse",
        "-r",
        help="Show blockers first by reversing edge direction"
    ),
    png: Optional[str] = typer.Option(
        None,
        "--png",
        help="Write PNG graph to this path"
    ),
    svg: Optional[str] = typer.Option(
        None,
        "--svg",
        help="Write SVG graph to this path"
    ),
    ascii_only: bool = typer.Option(
        False,
        "--ascii",
        help="Force ASCII output even when Graphviz is available"
    ),
    edges: bool = typer.Option(
        False,
        "--edges",
        help="Print the edge list (uuid -> uuid)"
    ),
    rankdir: str = typer.Option(
        "LR",
        "--rankdir",
        help="Graph layout direction (LR, TB, BT, RL)"
    ),
):
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
            raise typer.BadParameter("Only 'reverse' is supported as a mode.")
        reverse = True

    tasks = get_tasks_from_taskwarrior()
    if not tasks:
        print("No pending tasks found.")
        return

    rankdir = rankdir.strip().upper()
    if rankdir not in {"LR", "TB", "BT", "RL"}:
        raise typer.BadParameter("rankdir must be one of LR, TB, BT, or RL.")

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


TWH_COMMANDS = {"list", "reverse", "tree", "graph", "simple"}
TWH_HELP_ARGS = {"-h", "--help", "--install-completion", "--show-completion"}


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
    >>> should_delegate_to_task(["list"])
    False
    >>> should_delegate_to_task(["--help"])
    False
    """
    if not argv:
        return True
    first_arg = argv[0]
    return first_arg not in TWH_COMMANDS and first_arg not in TWH_HELP_ARGS


def main():
    """
    Entry point for the twh command.

    If the command is not recognized, delegate to Taskwarrior.
    """
    argv = sys.argv[1:]
    if should_delegate_to_task(argv):
        argv, message = apply_context_to_add_args(argv)
        if message:
            print(message)
        try:
            exit_code = apply_blocks_relationship(argv)
        except FileNotFoundError:
            print("Error: `task` command not found.", file=sys.stderr)
            sys.exit(1)
        sys.exit(exit_code)

    app()


if __name__ == "__main__":
    main()

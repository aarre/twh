#!/usr/bin/env python3
"""
Shared Taskwarrior export helpers.
"""

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


CASE_INSENSITIVE_OVERRIDE = "rc.search.case.sensitive=no"


def apply_case_insensitive_overrides(args: Sequence[str]) -> List[str]:
    """
    Prepend the case-insensitive search override to Taskwarrior args.

    Parameters
    ----------
    args : Sequence[str]
        Taskwarrior arguments excluding the executable.

    Returns
    -------
    List[str]
        Arguments with the case-insensitive override applied.

    Examples
    --------
    >>> apply_case_insensitive_overrides(["list"])
    ['rc.search.case.sensitive=no', 'list']
    >>> apply_case_insensitive_overrides(["rc.search.case.sensitive=yes", "list"])
    ['rc.search.case.sensitive=no', 'list']
    """
    cleaned = [
        arg
        for arg in args
        if not str(arg).startswith("rc.search.case.sensitive=")
    ]
    return [CASE_INSENSITIVE_OVERRIDE, *cleaned]


def apply_taskrc_overrides(args: Sequence[str]) -> List[str]:
    """
    Ensure Taskwarrior commands use the canonical taskrc path.

    Parameters
    ----------
    args : Sequence[str]
        Taskwarrior arguments excluding the executable.

    Returns
    -------
    List[str]
        Arguments with the canonical taskrc override applied.

    Examples
    --------
    >>> apply_taskrc_overrides(["rc:/other", "list"])[0].startswith("rc:")
    True
    """
    taskrc = get_taskrc_path()
    cleaned = [arg for arg in args if not str(arg).startswith("rc:")]
    if taskrc is None:
        return cleaned
    return [f"rc:{taskrc}", *cleaned]


def parse_taskwarrior_json(text: str) -> List[Dict]:
    """
    Parse Taskwarrior JSON output that may be an array or line-delimited JSON.
    """
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        tasks: List[Dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
        return tasks


def read_tasks_from_json(json_data: str) -> List[Dict]:
    """
    Parse tasks from a JSON string (array or line-delimited JSON).
    """
    return parse_taskwarrior_json(json_data)


def get_tasks_from_taskwarrior(status: Optional[str] = "pending") -> List[Dict]:
    """
    Execute Taskwarrior export and return parsed task data.
    """
    try:
        task_args = apply_case_insensitive_overrides(["export"])
        task_args = apply_taskrc_overrides(task_args)
        result = subprocess.run(
            ["task", *task_args],
            capture_output=True,
            text=True,
            check=True
        )
        tasks = parse_taskwarrior_json(result.stdout)
        if status:
            return [t for t in tasks if t.get("status") == status]
        return tasks
    except subprocess.CalledProcessError as e:
        print(f"Error executing taskwarrior: {e}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing taskwarrior output: {e}", file=sys.stderr)
        raise


def normalize_dependency_value(value: object) -> Optional[str]:
    """
    Normalize dependency identifiers from Taskwarrior payloads.

    Parameters
    ----------
    value : object
        Raw dependency value.

    Returns
    -------
    Optional[str]
        Cleaned dependency identifier, or None when empty.
    """
    if value is None:
        return None
    dep = str(value).strip()
    if not dep:
        return None
    dep = dep.lstrip("+-").strip()
    return dep or None


def parse_dependencies(dep_field: Optional[str]) -> List[str]:
    """
    Parse the dependencies field from a task.
    """
    if not dep_field:
        return []
    if isinstance(dep_field, list):
        values = dep_field
    else:
        values = str(dep_field).split(",")
    dependencies: List[str] = []
    for value in values:
        dep = normalize_dependency_value(value)
        if dep:
            dependencies.append(dep)
    return dependencies


def filter_modified_zero_lines(stdout: Optional[str]) -> List[str]:
    """
    Filter noisy Taskwarrior output lines.

    Parameters
    ----------
    stdout : Optional[str]
        Raw stdout from Taskwarrior.

    Returns
    -------
    List[str]
        Filtered stdout lines.

    Examples
    --------
    >>> filter_modified_zero_lines("Modified 0 tasks.\\n")
    []
    >>> filter_modified_zero_lines("Modified 1 task.\\n")
    ['Modified 1 task.']
    >>> filter_modified_zero_lines("Project 'work' is 0% complete (1 task remaining).\\n")
    []
    """
    if not stdout:
        return []
    lines = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == "Modified 0 tasks.":
            continue
        if stripped.startswith("Project '") and " complete (" in stripped:
            continue
        lines.append(line)
    return lines


def get_taskwarrior_setting(key: str) -> Optional[str]:
    """
    Fetch a Taskwarrior configuration value.
    """
    task_args = apply_taskrc_overrides(["_get", key])
    result = subprocess.run(
        ["task", *task_args],
        capture_output=True,
        text=True,
        check=False,
    )
    value = (result.stdout or "").strip()
    if value:
        return value
    if result.returncode != 0:
        return None
    return None


def _parse_taskrc_setting(path: Path, key: str, visited: Optional[set[Path]] = None) -> Optional[str]:
    if visited is None:
        visited = set()
    if path in visited or not path.exists():
        return None
    visited.add(path)
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("include "):
            include_path = line[len("include ") :].strip().strip('"')
            if not include_path:
                continue
            resolved = Path(os.path.expandvars(os.path.expanduser(include_path)))
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            value = _parse_taskrc_setting(resolved, key, visited)
            if value:
                return value
            continue
        if not line.startswith(f"{key}="):
            continue
        return line.split("=", 1)[1].strip() or None
    return None


def get_taskrc_path() -> Optional[Path]:
    """
    Return the canonical Taskwarrior config path.

    This ignores TASKRC environment overrides so twh always targets ~/.taskrc.
    """
    return Path.home() / ".taskrc"


def get_task_data_location(taskrc_path: Optional[Path] = None) -> Path:
    """
    Return the Taskwarrior data directory.

    Parameters
    ----------
    taskrc_path : Optional[Path], optional
        Path to the taskrc file (default: resolved taskrc).

    Returns
    -------
    Path
        Taskwarrior data directory.

    Examples
    --------
    >>> import tempfile
    >>> tmp = Path(tempfile.mkdtemp())
    >>> taskrc = tmp / ".taskrc"
    >>> _ = taskrc.write_text("data.location=task-data\\n", encoding="utf-8")
    >>> get_task_data_location(taskrc_path=taskrc) == (tmp / "task-data")
    True
    """
    taskrc = taskrc_path or get_taskrc_path()
    default = Path.home() / ".task"
    if not taskrc:
        return default
    raw_value = _parse_taskrc_setting(taskrc, "data.location")
    if not raw_value:
        raw_value = _parse_taskrc_setting(taskrc, "rc.data.location")
    if not raw_value:
        return default
    cleaned = raw_value.strip().strip('"').strip("'")
    if not cleaned:
        return default
    expanded = Path(os.path.expandvars(os.path.expanduser(cleaned)))
    if not expanded.is_absolute():
        expanded = (taskrc.parent / expanded).resolve()
    return expanded


def taskrc_udas_present(
    fields: Iterable[str],
    taskrc_path: Optional[Path] = None,
) -> List[str]:
    """
    Return UDA field names present in a taskrc file.

    Parameters
    ----------
    fields : Iterable[str]
        UDA field names to check.
    taskrc_path : Optional[Path], optional
        Path to the taskrc file (default: resolved taskrc).

    Returns
    -------
    List[str]
        Field names whose UDA type is defined in the taskrc.
    """
    taskrc = taskrc_path or get_taskrc_path()
    if not taskrc:
        return []
    present: List[str] = []
    for field in fields:
        setting_key = f"uda.{field}.type"
        if _parse_taskrc_setting(taskrc, setting_key):
            present.append(field)
    return present


def describe_missing_udas(
    missing: Sequence[str],
    taskrc_path: Optional[Path] = None,
) -> str:
    """
    Build a helpful error message for missing UDAs.

    Parameters
    ----------
    missing : Sequence[str]
        Missing UDA field names.
    taskrc_path : Optional[Path], optional
        Path to the taskrc file (default: resolved taskrc).

    Returns
    -------
    str
        Error message describing missing UDAs.
    """
    missing_list = ", ".join(missing)
    taskrc = taskrc_path or get_taskrc_path()
    present = taskrc_udas_present(missing, taskrc_path=taskrc)
    if present and taskrc:
        present_list = ", ".join(present)
        return (
            "Missing Taskwarrior UDA(s): "
            f"{missing_list}. Taskwarrior did not report "
            f"{present_list} even though it appears in {taskrc}. "
            "Ensure the active taskrc includes these settings or "
            "run `task config` to write them."
        )
    return (
        "Missing Taskwarrior UDA(s): "
        f"{missing_list}. Aborting to avoid modifying move descriptions."
    )


def missing_udas(
    fields: Iterable[str],
    get_setting: Optional[Callable[[str], Optional[str]]] = get_taskwarrior_setting,
    *,
    allow_taskrc_fallback: bool = True,
) -> List[str]:
    """
    Return missing UDA field names.

    Parameters
    ----------
    fields : Iterable[str]
        UDA field names to check.
    get_setting : Optional[Callable[[str], Optional[str]]]
        Getter for Taskwarrior settings (default: Taskwarrior config lookup).
    allow_taskrc_fallback : bool, optional
        When True, treat UDAs present in taskrc as available
        (default: True).
    """
    if get_setting is None:
        get_setting = get_taskwarrior_setting
    taskrc = get_taskrc_path() if allow_taskrc_fallback else None
    defined_udas: Optional[set[str]] = None
    missing: List[str] = []
    for field in fields:
        setting_key = f"uda.{field}.type"
        if get_setting(setting_key):
            continue
        if defined_udas is None:
            defined_udas = get_defined_udas()
        if field in (defined_udas or set()):
            continue
        if allow_taskrc_fallback and taskrc and _parse_taskrc_setting(taskrc, setting_key):
            continue
        missing.append(field)
    return missing


def _parse_defined_udas(output: str) -> set[str]:
    """
    Parse UDA names from `task udas` output.

    Parameters
    ----------
    output : str
        Raw output from `task udas`.

    Returns
    -------
    set[str]
        Parsed UDA names.

    Examples
    --------
    >>> sample = "Name Type\\nmode string\\nimp numeric\\n\\n20 UDAs defined"
    >>> sorted(_parse_defined_udas(sample))[:2]
    ['imp', 'mode']
    """
    udas: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Orphan UDA"):
            break
        if line.startswith("Name") and "Type" in line:
            continue
        if line.startswith("----"):
            continue
        if line.endswith("UDAs defined"):
            continue
        tokens = line.split()
        if not tokens:
            continue
        udas.add(tokens[0])
    return udas


def get_defined_udas(
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> set[str]:
    """
    Return UDAs reported by Taskwarrior.

    Parameters
    ----------
    runner : Optional[Callable[..., subprocess.CompletedProcess]]
        Runner to execute Taskwarrior commands (args exclude ``task``).

    Returns
    -------
    set[str]
        UDA names reported by Taskwarrior.
    """
    if runner is None:
        def task_runner(args, **kwargs):
            task_args = apply_taskrc_overrides(list(args))
            return subprocess.run(["task", *task_args], **kwargs)

        runner = task_runner
    result = runner(["udas"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return set()
    return _parse_defined_udas(result.stdout or "")


def _parse_columns_output(output: str) -> List[str]:
    """
    Parse Taskwarrior _columns output into column names.

    Parameters
    ----------
    output : str
        Raw _columns output.

    Returns
    -------
    List[str]
        Parsed column names.

    Examples
    --------
    >>> _parse_columns_output("id\\nproject\\n\\nstatus\\n")
    ['id', 'project', 'status']
    """
    return [token.strip() for token in output.split() if token.strip()]


def get_taskwarrior_columns(
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> List[str]:
    """
    Return Taskwarrior column names from ``task _columns``.

    Parameters
    ----------
    runner : Optional[Callable[..., subprocess.CompletedProcess]]
        Runner to execute Taskwarrior commands (args exclude ``task``).

    Returns
    -------
    List[str]
        Column names reported by Taskwarrior.
    """
    if runner is None:
        def task_runner(args, **kwargs):
            task_args = apply_taskrc_overrides(list(args))
            return subprocess.run(["task", *task_args], **kwargs)

        runner = task_runner
    result = runner(["_columns"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    return _parse_columns_output(result.stdout or "")


@lru_cache(maxsize=1)
def get_core_attribute_names(
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    udas_runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> set[str]:
    """
    Return Taskwarrior core attribute names (excluding UDAs).

    Parameters
    ----------
    runner : Optional[Callable[..., subprocess.CompletedProcess]]
        Runner for ``task _columns``.
    udas_runner : Optional[Callable[..., subprocess.CompletedProcess]]
        Runner for ``task udas``.

    Returns
    -------
    set[str]
        Core attribute names.
    """
    columns = set(get_taskwarrior_columns(runner=runner))
    if not columns:
        return set()
    udas = get_defined_udas(runner=udas_runner or runner)
    return {col for col in columns if col not in udas}

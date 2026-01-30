#!/usr/bin/env python3
"""
Shared Taskwarrior export helpers.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


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
        result = subprocess.run(
            ["task", "export"],
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


def parse_dependencies(dep_field: Optional[str]) -> List[str]:
    """
    Parse the dependencies field from a task.
    """
    if not dep_field:
        return []
    if isinstance(dep_field, list):
        return [str(value).strip() for value in dep_field if str(value).strip()]
    return [item.strip() for item in str(dep_field).split(",") if item.strip()]


def get_taskwarrior_setting(key: str) -> Optional[str]:
    """
    Fetch a Taskwarrior configuration value.
    """
    result = subprocess.run(
        ["task", "_get", key],
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
    taskrc = os.environ.get("TASKRC")
    if taskrc:
        return Path(os.path.expandvars(os.path.expanduser(taskrc)))
    default = Path.home() / ".taskrc"
    return default if default.exists() else None


def missing_udas(
    fields: Iterable[str],
    get_setting: Callable[[str], Optional[str]] = get_taskwarrior_setting,
) -> List[str]:
    """
    Return missing UDA field names.
    """
    taskrc = get_taskrc_path()
    missing: List[str] = []
    for field in fields:
        setting_key = f"uda.{field}.type"
        if get_setting(setting_key):
            continue
        if taskrc and _parse_taskrc_setting(taskrc, setting_key):
            continue
        missing.append(field)
    return missing

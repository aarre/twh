#!/usr/bin/env python3
"""
Shared Taskwarrior export helpers.
"""

import json
import subprocess
import sys
from typing import Dict, List, Optional


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

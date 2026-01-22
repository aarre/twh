#!/usr/bin/env python3
"""
Hierarchical view of Taskwarrior tasks by dependency.
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Set


def get_tasks() -> List[Dict]:
    """Execute taskwarrior and return parsed task data."""
    try:
        result = subprocess.run(
            ["task", "export"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse line by line as each line may be a separate JSON object
        tasks = []
        for line in result.stdout.strip().split('\n'):
            if line:
                tasks.append(json.loads(line))
        # Filter to pending tasks only
        return [t for t in tasks if t.get("status") == "pending"]
    except subprocess.CalledProcessError as e:
        print(f"Error executing taskwarrior: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing taskwarrior output: {e}", file=sys.stderr)
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
        if "depends" in task:
            # depends field is a comma-separated string of UUIDs
            deps = task["depends"].split(",") if task["depends"] else []
            for dep_uuid in deps:
                dep_uuid = dep_uuid.strip()
                if dep_uuid:
                    depends_on[uuid].add(dep_uuid)
                    depended_by[dep_uuid].add(uuid)

    return task_map, depends_on, depended_by


def format_task(task: Dict) -> str:
    """Format a task for display."""
    task_id = task.get("id", "?")
    description = task.get("description", "")
    urgency = task.get("urgency", 0)
    return f"[{task_id}] {description} (urgency: {urgency:.1f})"


def print_tree_normal(task_map: Dict[str, Dict], depends_on: Dict[str, Set[str]],
                     depended_by: Dict[str, Set[str]], indent: str = "  "):
    """
    Print dependency tree with unblocked tasks at top level.
    Tasks with no dependencies are shown first, their dependencies indented below.
    """
    visited = set()

    def print_task_and_deps(uuid: str, level: int = 0):
        """Recursively print a task and its dependencies."""
        if uuid in visited:
            return

        if uuid not in task_map:
            # Dependency task might be completed/deleted - skip it
            return

        visited.add(uuid)
        task = task_map[uuid]
        print(f"{indent * level}{format_task(task)}")

        # Print dependencies (tasks this one depends on)
        deps = depends_on.get(uuid, [])
        if deps:
            for dep_uuid in sorted(deps,
                                  key=lambda u: task_map.get(u, {}).get("urgency", 0),
                                  reverse=True):
                print_task_and_deps(dep_uuid, level + 1)

    # Find top-level tasks (tasks that no other pending task depends on)
    top_level = []
    for uuid in task_map:
        # A task is top-level if no other pending task depends on it
        if not depended_by.get(uuid):
            top_level.append(uuid)

    # Sort by urgency (descending)
    top_level.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)

    for uuid in top_level:
        print_task_and_deps(uuid)
        print()  # Blank line between top-level tasks

    # Handle orphaned tasks (e.g., circular dependencies)
    # These are tasks that weren't visited because they're in dependency cycles
    orphaned = [uuid for uuid in task_map if uuid not in visited]
    if orphaned:
        orphaned.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)
        for uuid in orphaned:
            print_task_and_deps(uuid)
            print()


def print_tree_reverse(task_map: Dict[str, Dict], depends_on: Dict[str, Set[str]],
                      depended_by: Dict[str, Set[str]], indent: str = "  "):
    """
    Print dependency tree with blocking tasks at top level.
    Tasks that have no dependencies are shown first, dependents below.
    """
    visited = set()

    def print_task_and_dependents(uuid: str, level: int = 0):
        """Recursively print a task and tasks that depend on it."""
        if uuid in visited:
            return

        if uuid not in task_map:
            # Task might be completed/deleted - skip it
            return

        visited.add(uuid)
        task = task_map[uuid]
        print(f"{indent * level}{format_task(task)}")

        # Print dependents (tasks that depend on this one)
        dependents = depended_by.get(uuid, [])
        if dependents:
            for dep_uuid in sorted(dependents,
                                  key=lambda u: task_map.get(u, {}).get("urgency", 0),
                                  reverse=True):
                print_task_and_dependents(dep_uuid, level + 1)

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

    for uuid in bottom_level:
        print_task_and_dependents(uuid)
        print()  # Blank line between top-level tasks

    # Handle orphaned tasks (e.g., circular dependencies)
    # These are tasks that weren't visited because they're in dependency cycles
    orphaned = [uuid for uuid in task_map if uuid not in visited]
    if orphaned:
        orphaned.sort(key=lambda u: task_map[u].get("urgency", 0), reverse=True)
        for uuid in orphaned:
            print_task_and_dependents(uuid)
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Display Taskwarrior tasks in hierarchical dependency view"
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Show most-depended-upon tasks at top level (reverse view)"
    )
    args = parser.parse_args()

    tasks = get_tasks()

    if not tasks:
        print("No pending tasks found.")
        return

    task_map, depends_on, depended_by = build_dependency_graph(tasks)

    if args.reverse:
        print_tree_reverse(task_map, depends_on, depended_by)
    else:
        print_tree_normal(task_map, depends_on, depended_by)


if __name__ == "__main__":
    main()

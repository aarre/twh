#!/usr/bin/env python3
"""
Hierarchical view of Taskwarrior tasks by dependency.
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Set, Optional

import typer


def get_tasks() -> List[Dict]:
    """
    Execute taskwarrior and return parsed task data.

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
    """
    Format a task for display in the list view.

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


def print_tree_normal(task_map: Dict[str, Dict], depends_on: Dict[str, Set[str]],
                     depended_by: Dict[str, Set[str]], indent: str = "  "):
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
    """

    def print_task_and_dependents(uuid: str, level: int = 0, ancestors: Set[str] = None):
        """Recursively print a task and tasks that depend on it."""
        if ancestors is None:
            ancestors = set()

        # Prevent infinite recursion by checking if we're already in the current path
        if uuid in ancestors:
            return

        if uuid not in task_map:
            # Task might be completed/deleted - skip it
            return

        task = task_map[uuid]
        print(f"{indent * level}{format_task(task)}")

        # Print dependents (tasks that depend on this one)
        dependents = depended_by.get(uuid, [])
        if dependents:
            for dep_uuid in sorted(dependents,
                                  key=lambda u: task_map.get(u, {}).get("urgency", 0),
                                  reverse=True):
                # Add current uuid to ancestors before recursing
                print_task_and_dependents(dep_uuid, level + 1, ancestors | {uuid})

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
        for uuid in orphaned:
            print_task_and_dependents(uuid)
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

    This is the default command that shows tasks organized by their
    dependencies.
    """
    tasks = get_tasks()

    if not tasks:
        print("No pending tasks found.")
        return

    task_map, depends_on, depended_by = build_dependency_graph(tasks)

    if mode:
        if mode != "reverse":
            raise typer.BadParameter("Only 'reverse' is supported as a mode.")
        reverse = True

    if reverse:
        print_tree_reverse(task_map, depends_on, depended_by)
    else:
        print_tree_normal(task_map, depends_on, depended_by)


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
        help="Show most-depended-upon tasks at top level (reverse view)"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for Mermaid file (default: tasks.mmd)"
    ),
    csv: Optional[str] = typer.Option(
        None,
        "--csv",
        "-c",
        help="Output path for CSV export (default: tasks.csv)"
    ),
    render: bool = typer.Option(
        True,
        "--render/--no-render",
        help="Render Mermaid and open it"
    ),
    png: bool = typer.Option(
        False,
        "--png",
        help="Render to PNG instead of SVG"
    ),
):
    """
    Generate Mermaid flowchart of task dependencies.

    Creates a Mermaid diagram showing task dependencies, optionally
    renders it to SVG (or PNG), and opens the result.
    """
    from pathlib import Path
    from .graph import get_tasks_from_taskwarrior, create_task_graph

    # Get tasks
    tasks = get_tasks_from_taskwarrior()

    if not tasks:
        print("No pending tasks found.")
        return

    if mode:
        if mode != "reverse":
            raise typer.BadParameter("Only 'reverse' is supported as a mode.")
        reverse = True

    # Set default output paths
    output_mmd = Path(output) if output else Path("tasks.mmd")
    output_csv = Path(csv) if csv else Path("tasks.csv")

    # Generate graph
    print(f"Generating Mermaid graph: {output_mmd}")
    mermaid_content = create_task_graph(
        tasks,
        output_mmd=output_mmd,
        output_csv=output_csv,
        reverse=reverse
    )

    print(f"Generated Mermaid file: {output_mmd}")
    print(f"Generated CSV file: {output_csv}")

    # Render output if requested
    if render:
        from .renderer import render_mermaid_to_png, render_mermaid_to_svg, open_file, open_in_browser

        if png:
            png_path = output_mmd.with_suffix('.png')
            print(f"Rendering to PNG: {png_path}")

            try:
                render_mermaid_to_png(output_mmd, png_path)
                print(f"Successfully rendered to: {png_path}")
                open_file(png_path)
            except Exception as e:
                print(f"Error rendering to PNG: {e}", file=sys.stderr)
                print(
                    "You can view the Mermaid file in a Mermaid-compatible editor "
                    "(e.g., VS Code with a Mermaid extension, or https://mermaid.live/)."
                )
        else:
            svg_path = output_mmd.with_suffix('.svg')
            print(f"Rendering to SVG: {svg_path}")
            try:
                render_mermaid_to_svg(output_mmd, svg_path)
                print(f"Successfully rendered to: {svg_path}")
                open_in_browser(svg_path)
            except Exception as e:
                print(f"Error rendering to SVG: {e}", file=sys.stderr)
                print(
                    "You can view the Mermaid file in a Mermaid-compatible editor "
                    "(e.g., VS Code with a Mermaid extension, or https://mermaid.live/)."
                )


def main():
    """
    Entry point for the twh command.

    If no command is provided, defaults to the 'list' command.
    """
    # Default to the list command when no explicit subcommand is provided.
    if len(sys.argv) == 1:
        sys.argv.append("list")
    elif sys.argv[1].startswith("-"):
        sys.argv.insert(1, "list")
    elif sys.argv[1] == "reverse":
        sys.argv[1:2] = ["list", "reverse"]
    app()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hierarchical view of Taskwarrior tasks by dependency.
"""

import json
import tempfile
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

import typer

from .taskwarrior import get_tasks_from_taskwarrior, parse_dependencies


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
    except subprocess.CalledProcessError:
        return None
    value = result.stdout.strip()
    return value if value else None


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
        help="Output path for Mermaid file (default: /tmp/tasks.mmd)"
    ),
    csv: Optional[str] = typer.Option(
        None,
        "--csv",
        "-c",
        help="Output path for CSV export (default: /tmp/tasks.csv)"
    ),
    render: bool = typer.Option(
        True,
        "--render/--no-render",
        help="Render Mermaid to SVG and open it in a browser"
    ),
    png: bool = typer.Option(
        False,
        "--png",
        help="Render to PNG instead of SVG"
    ),
):
    """
    Generate Mermaid flowchart of task dependencies.

    Creates a Mermaid diagram showing task dependencies, writes outputs to
    /tmp by default, and renders to SVG (open in browser) unless --png is set.
    """
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
    output_dir = get_graph_output_dir()
    output_mmd = Path(output) if output else output_dir / "tasks.mmd"
    output_csv = Path(csv) if csv else output_dir / "tasks.csv"

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

#!/usr/bin/env python3
"""
Graphviz-based dependency graph rendering with ASCII fallback.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .taskwarrior import parse_dependencies


def short_uuid(value: str, length: int = 8) -> str:
    """
    Return a short UUID prefix for display.

    Parameters
    ----------
    value : str
        Full UUID string.
    length : int, optional
        Prefix length (default: 8).

    Returns
    -------
    str
        Shortened UUID string.

    Examples
    --------
    >>> short_uuid("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    'aaaaaaaa'
    """
    return value[:length]


def format_task_label(task: Dict, uuid: str, max_length: int = 80) -> str:
    """
    Build a human-readable label for a task.

    Parameters
    ----------
    task : Dict
        Task payload.
    uuid : str
        Task UUID for fallback labeling.
    max_length : int, optional
        Maximum label length (default: 80).

    Returns
    -------
    str
        Label string for display.

    Examples
    --------
    >>> format_task_label({"id": 3, "description": "Task C"}, "uuid")
    '[3] Task C'
    """
    description = str(task.get("description", "")).strip()
    task_id = task.get("id")

    if description:
        if task_id is not None:
            label = f"[{task_id}] {description}"
        else:
            label = description
    else:
        label = f"[{task_id}]" if task_id is not None else uuid

    if len(label) > max_length:
        label = label[: max_length - 3] + "..."
    return label


def build_dependency_edges(
    tasks: List[Dict],
    reverse: bool = False,
) -> Tuple[List[Tuple[str, str]], Dict[str, Dict]]:
    """
    Build dependency edges for Graphviz rendering.

    Parameters
    ----------
    tasks : List[Dict]
        Taskwarrior task dictionaries.
    reverse : bool, optional
        When True, flip edge direction (default: False).

    Returns
    -------
    List[Tuple[str, str]]
        Edge list as (source_uuid, target_uuid).
    Dict[str, Dict]
        Mapping of UUID to task dictionary.

    Examples
    --------
    >>> tasks = [{"uuid": "a", "depends": "b"}, {"uuid": "b"}]
    >>> edges, _ = build_dependency_edges(tasks)
    >>> edges
    [('a', 'b')]
    """
    by_uuid = {t["uuid"]: t for t in tasks if t.get("uuid")}
    edges: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for task in tasks:
        uuid = task.get("uuid")
        if not uuid:
            continue
        for dep_uuid in parse_dependencies(task.get("depends")):
            if dep_uuid not in by_uuid:
                continue
            edge = (dep_uuid, uuid) if reverse else (uuid, dep_uuid)
            if edge in seen:
                continue
            edges.append(edge)
            seen.add(edge)

    return edges, by_uuid


def build_adjacency(
    edges: Iterable[Tuple[str, str]],
) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """
    Build adjacency mappings from edges.

    Parameters
    ----------
    edges : Iterable[Tuple[str, str]]
        Edge list as (source_uuid, target_uuid).

    Returns
    -------
    Dict[str, List[str]]
        Mapping from source UUID to sorted list of child UUIDs.
    Dict[str, Set[str]]
        Mapping from target UUID to parent UUIDs.
    """
    children: Dict[str, List[str]] = defaultdict(list)
    parents: Dict[str, Set[str]] = defaultdict(set)

    for source, target in edges:
        children[source].append(target)
        parents[target].add(source)

    for source in list(children.keys()):
        children[source] = sorted(set(children[source]))

    return children, parents


def ascii_forest(
    edges: List[Tuple[str, str]],
    by_uuid: Dict[str, Dict],
    max_label_length: int = 80,
) -> List[str]:
    """
    Render an ASCII forest for the dependency graph.

    Parameters
    ----------
    edges : List[Tuple[str, str]]
        Edge list as (source_uuid, target_uuid).
    by_uuid : Dict[str, Dict]
        Mapping from UUID to task dictionary.
    max_label_length : int, optional
        Maximum label length (default: 80).

    Returns
    -------
    List[str]
        Lines of ASCII output for printing.
    """
    children, parents = build_adjacency(edges)
    nodes = set(by_uuid.keys())
    roots = sorted(node for node in nodes if node not in parents)
    if not roots:
        roots = sorted(nodes)

    lines: List[str] = []

    def add_line(prefix: str, uuid: str) -> None:
        task = by_uuid.get(uuid, {"uuid": uuid})
        label = format_task_label(task, uuid, max_length=max_label_length)
        lines.append(f"{prefix}{label} [{short_uuid(uuid)}]")

    def walk(uuid: str, depth: int, path: Set[str]) -> None:
        prefix = "" if depth == 0 else "  " * depth + "- "
        add_line(prefix, uuid)
        for child in children.get(uuid, []):
            if child in path:
                cycle_prefix = "  " * (depth + 1) + "- "
                lines.append(f"{cycle_prefix}(cycle) [{short_uuid(child)}]")
                continue
            walk(child, depth + 1, path | {child})

    for root in roots:
        walk(root, 0, {root})

    return lines


def format_edge_list(edges: List[Tuple[str, str]]) -> List[str]:
    """
    Format edges as printable strings.

    Parameters
    ----------
    edges : List[Tuple[str, str]]
        Edge list as (source_uuid, target_uuid).

    Returns
    -------
    List[str]
        Lines in "source -> target" format.
    """
    return [f"{source} -> {target}" for source, target in edges]


def escape_dot_label(value: str) -> str:
    """
    Escape a string for DOT labels.

    Parameters
    ----------
    value : str
        Label text.

    Returns
    -------
    str
        Escaped label string.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def generate_dot(
    by_uuid: Dict[str, Dict],
    edges: List[Tuple[str, str]],
    rankdir: str = "LR",
    max_label_length: int = 80,
) -> str:
    """
    Generate DOT source for Graphviz.

    Parameters
    ----------
    by_uuid : Dict[str, Dict]
        Mapping from UUID to task dictionary.
    edges : List[Tuple[str, str]]
        Edge list as (source_uuid, target_uuid).
    rankdir : str, optional
        Graph layout direction (default: "LR").
    max_label_length : int, optional
        Maximum label length (default: 80).

    Returns
    -------
    str
        DOT source string.
    """
    lines = [
        "digraph twh {",
        f"  rankdir={rankdir};",
        "  node [shape=box, fontsize=10];",
    ]

    for uuid in sorted(by_uuid.keys()):
        task = by_uuid[uuid]
        label = format_task_label(task, uuid, max_length=max_label_length)
        label = escape_dot_label(label)
        lines.append(f'  "{uuid}" [label="{label}"];')

    for source, target in edges:
        lines.append(f'  "{source}" -> "{target}";')

    lines.append("}")
    return "\n".join(lines)


def render_graphviz(
    dot_source: str,
    png_path: Optional[Path],
    svg_path: Optional[Path],
) -> Tuple[bool, Optional[str]]:
    """
    Render DOT source to PNG/SVG using Graphviz.

    Parameters
    ----------
    dot_source : str
        DOT graph definition.
    png_path : Optional[Path]
        Output path for PNG rendering.
    svg_path : Optional[Path]
        Output path for SVG rendering.

    Returns
    -------
    Tuple[bool, Optional[str]]
        Render status and error message when rendering fails.
    """
    if not png_path and not svg_path:
        return False, None

    dot = shutil.which("dot")
    if not dot:
        return False, "Graphviz 'dot' not found on PATH."

    try:
        if png_path:
            png_path.parent.mkdir(parents=True, exist_ok=True)
            _run_dot(dot, dot_source, png_path, "png")
        if svg_path:
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            _run_dot(dot, dot_source, svg_path, "svg")
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, f"Graphviz render failed: {exc}"

    return True, None


def _run_dot(dot: str, dot_source: str, output_path: Path, fmt: str) -> None:
    """
    Invoke Graphviz to render DOT source.

    Parameters
    ----------
    dot : str
        Path to dot executable.
    dot_source : str
        DOT graph definition.
    output_path : Path
        Output file path.
    fmt : str
        Output format (png or svg).
    """
    subprocess.run(
        [dot, f"-T{fmt}", "-o", str(output_path)],
        input=dot_source,
        text=True,
        check=True,
    )

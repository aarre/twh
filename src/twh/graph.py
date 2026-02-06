#!/usr/bin/env python3
"""
Graphviz-based dependency graph rendering with ASCII fallback.
"""

from __future__ import annotations

import html
import shutil
import subprocess
import sys
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .taskwarrior import parse_dependencies

STATUS_COLORS = {
    "started": "#b7e1b2",
    "blocked": "#e0e0e0",
    "normal": "#ffffff",
}
URGENCY_PALETTE = [
    "#2b83ba",
    "#5aa6d1",
    "#8dd0e6",
    "#cfe9f2",
    "#fee8c8",
    "#fdbb84",
    "#f46d43",
    "#d73027",
    "#a50026",
]
DEFAULT_URGENCY_COLOR = "#cfe9f2"
URGENCY_OPACITY = 0.35
BOX_WIDTH_PX = 320
DESCRIPTION_WRAP_WIDTH = 42


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


def parse_task_timestamp(value: Optional[str]) -> Optional[datetime]:
    """
    Parse a Taskwarrior timestamp string into a datetime.

    Parameters
    ----------
    value : Optional[str]
        Taskwarrior timestamp value.

    Returns
    -------
    Optional[datetime]
        Parsed datetime, or None when parsing fails.
    """
    if not value:
        return None
    value = str(value).strip()
    try:
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return datetime.strptime(value, "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def parse_task_boolean(value: Optional[object]) -> bool:
    """
    Parse Taskwarrior boolean-ish values into a bool.

    Parameters
    ----------
    value : Optional[object]
        Raw Taskwarrior value.

    Returns
    -------
    bool
        Parsed boolean.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"", "0", "0.0", "false", "no", "off", "none", "null"}:
        return False
    return True


def format_task_date(value: Optional[str]) -> str:
    """
    Format a Taskwarrior timestamp as a date string.

    Parameters
    ----------
    value : Optional[str]
        Taskwarrior timestamp value.

    Returns
    -------
    str
        Date formatted as YYYY-MM-DD, or empty string when missing.
    """
    dt = parse_task_timestamp(value)
    if not dt:
        return ""
    return dt.strftime("%Y-%m-%d")


def format_urgency_text(task: Dict) -> str:
    """
    Format urgency text for display.

    Parameters
    ----------
    task : Dict
        Task payload.

    Returns
    -------
    str
        Urgency label text.
    """
    urgency = task.get("urgency")
    if urgency is None:
        value = "?"
    else:
        try:
            urgency_num = float(urgency)
            value = f"{urgency_num:.2f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            value = str(urgency)
    return f"Urg: {value}"


def task_state(task: Dict, by_uuid: Dict[str, Dict]) -> str:
    """
    Determine task status for rendering.

    Parameters
    ----------
    task : Dict
        Task payload.
    by_uuid : Dict[str, Dict]
        Mapping of UUID to task dictionary.

    Returns
    -------
    str
        One of "blocked", "started", or "normal".
    """
    if any(dep in by_uuid for dep in parse_dependencies(task.get("depends"))):
        return "blocked"
    if parse_task_boolean(task.get("wip")):
        return "started"
    return "normal"


def priority_score(task: Dict) -> float:
    """
    Compute a numeric score for urgency/priority ranking.

    Parameters
    ----------
    task : Dict
        Task payload.

    Returns
    -------
    float
        Priority score used for ranking.
    """
    urgency = task.get("urgency")
    if urgency is not None:
        try:
            return round(float(urgency), 2)
        except (TypeError, ValueError):
            pass
    priority = str(task.get("priority", "")).upper()
    mapping = {"H": 3.0, "M": 2.0, "L": 1.0}
    return mapping.get(priority, 0.0)


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    """
    Convert a hex color string to RGB.

    Parameters
    ----------
    value : str
        Hex color (e.g., "#ff0000").

    Returns
    -------
    Tuple[int, int, int]
        RGB tuple.
    """
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert an RGB tuple to a hex color string.

    Parameters
    ----------
    rgb : Tuple[int, int, int]
        RGB values.

    Returns
    -------
    str
        Hex color string.
    """
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def lerp(a: int, b: int, t: float) -> int:
    """
    Linearly interpolate between two integers.

    Parameters
    ----------
    a : int
        Start value.
    b : int
        End value.
    t : float
        Interpolation value between 0 and 1.

    Returns
    -------
    int
        Interpolated integer.
    """
    return int(round(a + (b - a) * t))


def interpolate_palette(palette: List[str], t: float) -> Tuple[int, int, int]:
    """
    Interpolate a color palette for a position.

    Parameters
    ----------
    palette : List[str]
        List of hex colors.
    t : float
        Position from 0 to 1.

    Returns
    -------
    Tuple[int, int, int]
        Interpolated RGB color.
    """
    if t <= 0:
        return hex_to_rgb(palette[0])
    if t >= 1:
        return hex_to_rgb(palette[-1])
    pos = t * (len(palette) - 1)
    idx = int(pos)
    frac = pos - idx
    c1 = hex_to_rgb(palette[idx])
    c2 = hex_to_rgb(palette[idx + 1])
    return (
        lerp(c1[0], c2[0], frac),
        lerp(c1[1], c2[1], frac),
        lerp(c1[2], c2[2], frac),
    )


def blend_with_white(rgb: Tuple[int, int, int], opacity: float) -> Tuple[int, int, int]:
    """
    Blend an RGB color with white for a given opacity.

    Parameters
    ----------
    rgb : Tuple[int, int, int]
        Base RGB color.
    opacity : float
        Opacity of the base color.

    Returns
    -------
    Tuple[int, int, int]
        Blended RGB color.
    """
    return tuple(
        int(round(255 * (1 - opacity) + channel * opacity))
        for channel in rgb
    )


def build_urgency_color_map(by_uuid: Dict[str, Dict]) -> Dict[str, str]:
    """
    Build urgency colors for each task based on relative priority.

    Parameters
    ----------
    by_uuid : Dict[str, Dict]
        Mapping of UUID to task dictionary.

    Returns
    -------
    Dict[str, str]
        Mapping of UUID to hex color.
    """
    if not by_uuid:
        return {}
    scores = [priority_score(task) for task in by_uuid.values()]
    unique_scores = sorted(set(scores))
    rank_map = {score: idx for idx, score in enumerate(unique_scores)}
    rank_denom = max(len(unique_scores) - 1, 1)
    colors: Dict[str, str] = {}

    for uuid, task in by_uuid.items():
        score = priority_score(task)
        rank = rank_map.get(score, 0)
        t = rank / rank_denom if rank_denom else 0.0
        base = interpolate_palette(URGENCY_PALETTE, t)
        blended = blend_with_white(base, URGENCY_OPACITY)
        colors[uuid] = rgb_to_hex(blended)

    return colors


def truncate_text(value: str, max_length: int) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Parameters
    ----------
    value : str
        Text to truncate.
    max_length : int
        Maximum length.

    Returns
    -------
    str
        Truncated text.
    """
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def wrap_text(value: str, width: int) -> List[str]:
    """
    Wrap text to a fixed character width.

    Parameters
    ----------
    value : str
        Text to wrap.
    width : int
        Maximum characters per line.

    Returns
    -------
    List[str]
        Wrapped lines.

    Examples
    --------
    >>> wrap_text("alpha beta gamma", 10)
    ['alpha beta', 'gamma']
    """
    if width <= 0:
        return [value] if value else []
    normalized = " ".join(value.split())
    if not normalized:
        return []
    return textwrap.wrap(
        normalized,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def sanitize_html(value: str) -> str:
    """
    Escape text for Graphviz HTML labels.

    Parameters
    ----------
    value : str
        Text to escape.

    Returns
    -------
    str
        Escaped HTML text.
    """
    return html.escape(value, quote=True)


def build_html_label(
    task: Dict,
    uuid: str,
    urgency_color: str,
    status_color: str,
    max_length: int,
) -> str:
    """
    Build a Graphviz HTML label matching the graph metadata layout.

    Parameters
    ----------
    task : Dict
        Task payload.
    uuid : str
        Task UUID.
    urgency_color : str
        Hex color for the urgency bar.
    status_color : str
        Hex color for the status panel.
    max_length : int
        Maximum description line length.

    Returns
    -------
    str
        HTML label string.
    """
    description = str(task.get("description", "")).strip() or uuid
    wrapped_lines = wrap_text(description, max_length)
    if not wrapped_lines:
        wrapped_lines = [uuid]
    description = "<BR/>".join(sanitize_html(line) for line in wrapped_lines)
    task_id = task.get("id")
    id_text = f"ID: {task_id}" if task_id is not None else "ID: ?"
    id_text = sanitize_html(id_text)
    due_value = format_task_date(task.get("due"))
    due_text = sanitize_html(f"Due: {due_value}") if due_value else ""
    urgency_text = sanitize_html(format_urgency_text(task))

    meta_lines = f'<FONT POINT-SIZE="9">{id_text}</FONT>'
    if due_text:
        meta_lines += f'<BR/><FONT POINT-SIZE="9">{due_text}</FONT>'

    table_width = f'{BOX_WIDTH_PX}'
    cell_width = f' WIDTH="{BOX_WIDTH_PX}"'

    return (
        f'<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" WIDTH="{table_width}">'
        f'<TR><TD BGCOLOR="{urgency_color}" ALIGN="LEFT"{cell_width}><B>{urgency_text}</B></TD></TR>'
        f'<TR><TD BGCOLOR="{status_color}" ALIGN="LEFT"{cell_width}>{description}<BR/>{meta_lines}</TD></TR>'
        "</TABLE>"
    )


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
    max_label_length: int = DESCRIPTION_WRAP_WIDTH,
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
        '  node [shape=plain, fontsize=10, fontname="Verdana"];',
    ]

    urgency_colors = build_urgency_color_map(by_uuid)

    for uuid in sorted(by_uuid.keys()):
        task = by_uuid[uuid]
        urgency_color = urgency_colors.get(uuid, DEFAULT_URGENCY_COLOR)
        status = task_state(task, by_uuid)
        status_color = STATUS_COLORS.get(status, STATUS_COLORS["normal"])
        label = build_html_label(
            task,
            uuid,
            urgency_color,
            status_color,
            max_label_length,
        )
        lines.append(f'  "{uuid}" [label=<{label}>];')

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
    output_path_str = _dot_output_path(output_path, dot)
    subprocess.run(
        [dot, f"-T{fmt}", "-o", output_path_str],
        input=dot_source,
        text=True,
        check=True,
    )


def _dot_output_path(output_path: Path, dot_path: str) -> str:
    """
    Convert output paths when invoking Windows Graphviz from Cygwin.

    Parameters
    ----------
    output_path : Path
        Output path for graph rendering.
    dot_path : str
        Dot executable path.

    Returns
    -------
    str
        Path string to pass to Graphviz.
    """
    if _is_windows_exe_on_cygwin(dot_path):
        return _cygwin_to_windows_path(output_path)
    return str(output_path)


def _is_windows_exe_on_cygwin(dot_path: str) -> bool:
    """
    Check if the Graphviz binary is a Windows executable on Cygwin.

    Parameters
    ----------
    dot_path : str
        Dot executable path.

    Returns
    -------
    bool
        True when running Cygwin with a Windows Graphviz binary.
    """
    if not sys.platform.startswith("cygwin"):
        return False
    lowered = dot_path.lower()
    if lowered.startswith("/cygdrive/"):
        return True
    return lowered.endswith((".exe", ".cmd", ".bat"))


def _cygwin_to_windows_path(path: Path) -> str:
    """
    Convert a Cygwin path to a Windows path using cygpath.

    Parameters
    ----------
    path : Path
        Path to convert.

    Returns
    -------
    str
        Windows-style path string.
    """
    try:
        result = subprocess.run(
            ["cygpath", "-w", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return str(path)
    converted = result.stdout.strip()
    return converted or str(path)

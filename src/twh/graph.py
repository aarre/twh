#!/usr/bin/env python3
"""
Taskwarrior dependency graph visualization using Mermaid.

This module provides functionality to convert Taskwarrior task exports into
Mermaid flowcharts and CSV files for import into other systems (e.g., Tana).
"""

import csv
import html
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from .taskwarrior import get_tasks_from_taskwarrior, read_tasks_from_json, parse_dependencies




def build_dependency_graph(
    tasks: List[Dict],
    reverse: bool = False
) -> Tuple[Dict[str, Dict], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Build dependency graph structures from task list.

    Parameters
    ----------
    tasks : List[Dict]
        List of task dictionaries.
    reverse : bool
        When True, build edges from dependency to dependent.

    Returns
    -------
    uid_map : Dict[str, Dict]
        Mapping from UUID to task dictionary.
    succ : Dict[str, Set[str]]
        Successors: mapping from UUID to set of UUIDs that depend on it.
    pred : Dict[str, Set[str]]
        Predecessors: mapping from UUID to set of UUIDs it depends on.
    """
    uid_map = {t['uuid']: t for t in tasks if 'uuid' in t}
    succ = defaultdict(set)
    pred = defaultdict(set)

    for t in tasks:
        u = t.get('uuid')
        if not u:
            continue
        for d in parse_dependencies(t.get('depends')):
            if d in uid_map:
                if reverse:
                    succ[d].add(u)
                    pred[u].add(d)
                else:
                    succ[u].add(d)
                    pred[d].add(u)

    return uid_map, succ, pred


def collapse_chains(uid_map: Dict[str, Dict], succ: Dict[str, Set[str]],
                    pred: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Collapse simple dependency chains into single edges for cleaner visualization.

    A chain is a sequence of tasks where each task (except the first and last)
    has exactly one predecessor and one successor.
    Branch points always emit edges so dependencies are never dropped.

    Parameters
    ----------
    uid_map : Dict[str, Dict]
        Mapping from UUID to task dictionary.
    succ : Dict[str, Set[str]]
        Successors mapping.
    pred : Dict[str, Set[str]]
        Predecessors mapping.

    Returns
    -------
    List[List[str]]
        List of chains, where each chain is a list of UUIDs.
    """
    visited = set()
    edges = []

    for u in uid_map:
        if not succ.get(u):
            continue
        is_middle = len(pred[u]) == 1 and len(succ[u]) == 1
        if not is_middle:
            for s in succ.get(u, []):
                if (u, s) in visited:
                    continue
                chain = [u]
                cur = s
                while True:
                    chain.append(cur)
                    visited.add((chain[-2], cur))
                    if len(pred[cur]) == 1 and len(succ[cur]) == 1:
                        nxt = next(iter(succ[cur]))
                        if (chain[-1], nxt) in visited:
                            break
                        cur = nxt
                        continue
                    break
                edges.append(chain)

    return edges


def generate_mermaid(uid_map: Dict[str, Dict], chains: List[List[str]]) -> str:
    """
    Generate Mermaid flowchart syntax from dependency chains with ID labels
    and status-based styling.

    Parameters
    ----------
    uid_map : Dict[str, Dict]
        Mapping from UUID to task dictionary.
    chains : List[List[str]]
        List of dependency chains.

    Returns
    -------
    str
        Mermaid flowchart definition.
    """
    def sanitize_label(text: str) -> str:
        text = text.replace('\r', ' ').replace('\n', ' ')
        text = text.replace('[', '(').replace(']', ')')
        return html.escape(text, quote=True)

    def parse_task_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        value = str(value).strip()
        try:
            if value.endswith("Z"):
                return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            return datetime.strptime(value, "%Y%m%dT%H%M%S")
        except ValueError:
            return None

    def format_task_date(value: Optional[str]) -> str:
        dt = parse_task_timestamp(value)
        if not dt:
            return ""
        return dt.strftime("%Y-%m-%d")

    def node_id(uuid: str) -> str:
        return f"t_{uuid.replace('-', '')}"

    def node_label(task: Dict) -> str:
        desc = sanitize_label(task.get('description', ''))
        task_id = task.get('id')
        id_text = f"ID: {task_id}" if task_id is not None else "ID: ?"
        id_text = sanitize_label(id_text)
        due_value = format_task_date(task.get("due"))
        due_html = f'<div class="twh-due">Due: {sanitize_label(due_value)}</div>' if due_value else ""
        return (
            '<div class="twh-node">'
            '<div class="twh-body">'
            f'<div class="twh-desc">{desc}</div>'
            f'{due_html}'
            '</div>'
            f'<div class="twh-id">{id_text}</div>'
            '</div>'
        )

    def node_state(task: Dict) -> Optional[str]:
        if any(dep in uid_map for dep in parse_dependencies(task.get('depends'))):
            return "blocked"
        if task.get("start"):
            return "started"
        return None

    def priority_score(task: Dict) -> float:
        urgency = task.get("urgency")
        if isinstance(urgency, (int, float)):
            return float(urgency)
        priority = str(task.get("priority", "")).upper()
        mapping = {"H": 3.0, "M": 2.0, "L": 1.0}
        return mapping.get(priority, 0.0)

    def hex_to_rgb(value: str) -> Tuple[int, int, int]:
        value = value.lstrip("#")
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def lerp(a: int, b: int, t: float) -> int:
        return int(round(a + (b - a) * t))

    def interpolate_palette(palette: List[str], t: float) -> Tuple[int, int, int]:
        if t <= 0:
            return hex_to_rgb(palette[0])
        if t >= 1:
            return hex_to_rgb(palette[-1])
        pos = t * (len(palette) - 1)
        idx = int(pos)
        frac = pos - idx
        c1 = hex_to_rgb(palette[idx])
        c2 = hex_to_rgb(palette[idx + 1])
        return (lerp(c1[0], c2[0], frac), lerp(c1[1], c2[1], frac), lerp(c1[2], c2[2], frac))

    def blend_with_white(rgb: Tuple[int, int, int], white_weight: float) -> Tuple[int, int, int]:
        white_weight = max(0.0, min(1.0, white_weight))
        return (
            lerp(rgb[0], 255, white_weight),
            lerp(rgb[1], 255, white_weight),
            lerp(rgb[2], 255, white_weight),
        )

    scores = [priority_score(task) for task in uid_map.values()]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    # A light segment of the magma palette for urgency gradients.
    palette = ["#4f0c6b", "#7e1d6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06"]

    def color_for_task(task: Dict) -> Optional[str]:
        if node_state(task) is not None:
            return None
        score = priority_score(task)
        if max_score <= min_score:
            return "#ffffff"
        if score <= min_score:
            return "#ffffff"
        t = (score - min_score) / (max_score - min_score)
        base = interpolate_palette(palette, t)
        white_weight = 0.85 - (0.35 * t)
        return rgb_to_hex(blend_with_white(base, white_weight))

    lines = [
        '%%{init: {"flowchart": {"htmlLabels": true}, "themeCSS": '
        '".twh-node{position:relative;font-family:Verdana,Arial,sans-serif;'
        'line-height:1.2;padding:0;text-align:left;word-break:break-word;}'
        '.twh-body{padding:6px 6px 24px 6px;}'
        '.twh-desc{display:block;margin-bottom:4px;}'
        '.twh-due{display:block;font-size:11px;color:#333;}'
        '.twh-id{position:absolute;left:0;bottom:0;font-size:12px;'
        'font-weight:600;border:1px solid #666;border-radius:3px;'
        'padding:2px 6px;background:rgba(255,255,255,0.75);}"} }%%',
        'flowchart LR',
        '  classDef default fill:#ffffff,stroke:#7a7a7a,color:#1f1f1f;',
        '  classDef started fill:#b7e1b2,stroke:#2e7d32,color:#1b5e20;',
        '  classDef blocked fill:#e0e0e0,stroke:#9e9e9e,color:#424242;'
    ]
    used = set()

    for chain in chains:
        labels = []
        for uuid in chain:
            labels.append(node_label(uid_map[uuid]))
            used.add(uuid)

        # Create arrow chain with stable node IDs and readable labels.
        mer = ' --> '.join([
            f'{node_id(chain[i])}["{labels[i]}"]'
            for i in range(len(chain))
        ])
        lines.append('  ' + mer)

    # Add standalone nodes for tasks without edges.
    for uuid in uid_map:
        if uuid in used:
            continue
        label = node_label(uid_map[uuid])
        lines.append(f'  {node_id(uuid)}["{label}"]')

    for uuid, task in uid_map.items():
        state = node_state(task)
        if state:
            lines.append(f'  class {node_id(uuid)} {state}')
            continue
        fill = color_for_task(task)
        if fill:
            lines.append(f'  style {node_id(uuid)} fill:{fill}')

    return '\n'.join(lines)


def write_csv_export(uid_map: Dict[str, Dict], pred: Dict[str, Set[str]],
                     output_path: Path) -> None:
    """
    Write tasks to CSV file for import into other systems (e.g., Tana).

    Parameters
    ----------
    uid_map : Dict[str, Dict]
        Mapping from UUID to task dictionary.
    pred : Dict[str, Set[str]]
        Predecessors mapping.
    output_path : Path
        Path to output CSV file.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Title', 'Body', 'Tags', 'UUID', 'DependsOn'])
        for u, t in uid_map.items():
            title = t.get('description', '')
            body = f"task:{t.get('id', '')} entry:{t.get('entry', '') or ''}"
            tags = ';'.join(t.get('tags', []))
            depends = ','.join(sorted(pred.get(u, [])))
            w.writerow([title, body, tags, u, depends])


def create_task_graph(
    tasks: List[Dict],
    output_mmd: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    reverse: bool = False
) -> str:
    """
    Create Mermaid graph and optional CSV export from task list.

    Parameters
    ----------
    tasks : List[Dict]
        List of task dictionaries.
    output_mmd : Optional[Path]
        Path to write Mermaid file. If None, no file is written.
    output_csv : Optional[Path]
        Path to write CSV file. If None, no file is written.
    reverse : bool
        When True, reverse edge direction (dependency to dependent).

    Returns
    -------
    str
        Mermaid flowchart definition.
    """
    uid_map, succ, pred = build_dependency_graph(tasks, reverse=reverse)
    chains = collapse_chains(uid_map, succ, pred)
    mermaid_content = generate_mermaid(uid_map, chains)

    if output_mmd:
        output_mmd.write_text(mermaid_content, encoding='utf-8')

    if output_csv:
        write_csv_export(uid_map, pred, output_csv)

    return mermaid_content

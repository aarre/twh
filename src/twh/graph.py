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
    Generate Mermaid flowchart syntax from dependency chains with urgency bars
    and status-based panels.

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

    def node_label(task: Dict, urgency_color: str) -> str:
        desc = sanitize_label(task.get('description', ''))
        task_id = task.get('id')
        id_text = f"ID: {task_id}" if task_id is not None else "ID: ?"
        id_text = sanitize_label(id_text)
        urg_value = task.get("urgency")
        urg_text = "?"
        if urg_value is not None:
            try:
                urg_num = float(urg_value)
                urg_text = f"{urg_num:.2f}".rstrip("0").rstrip(".")
            except (TypeError, ValueError):
                urg_text = str(urg_value)
        urg_text = sanitize_label(f"Urg: {urg_text}")
        due_value = format_task_date(task.get("due"))
        due_html = f'<span class="twh-due">Due: {sanitize_label(due_value)}</span>' if due_value else ""
        status = node_state(task) or "normal"
        rgb = hex_to_rgb(urgency_color)
        urgency_style = f"background-color: rgba({rgb[0]},{rgb[1]},{rgb[2]},{fill_opacity});"
        return (
            '<div class="twh-node">'
            f'<div class="twh-urgency" style="{urgency_style}">'
            f'{urg_text}'
            '</div>'
            f'<div class="twh-status twh-status-{status}">'
            f'<div class="twh-title">{desc}</div>'
            '<div class="twh-meta">'
            f'<span class="twh-id twh-badge">{id_text}</span>'
            f'{due_html}'
            '</div>'
            '</div>'
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
        if urgency is not None:
            try:
                return round(float(urgency), 2)
            except (TypeError, ValueError):
                pass
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

    scores = [priority_score(task) for task in uid_map.values()]
    unique_scores = sorted(set(scores))
    rank_map = {score: idx for idx, score in enumerate(unique_scores)}
    rank_denom = max(len(unique_scores) - 1, 1)

    # Cool-to-hot urgency palette (chill -> on fire).
    palette = ["#2b83ba", "#5aa6d1", "#8dd0e6", "#cfe9f2", "#fee8c8",
               "#fdbb84", "#f46d43", "#d73027", "#a50026"]
    fill_opacity = 0.35

    def color_for_task(task: Dict) -> Optional[str]:
        score = priority_score(task)
        if not unique_scores:
            return None
        rank = rank_map.get(score, 0)
        t = rank / rank_denom if rank_denom else 0.0
        base = interpolate_palette(palette, t)
        return rgb_to_hex(base)

    lines = [
        '%%{init: {"flowchart": {"htmlLabels": true}, "themeCSS": '
        '".twh-node{border:1px solid #7a7a7a;border-radius:4px;overflow:hidden;'
        'font-family:Verdana,Arial,sans-serif;line-height:1.2;'
        'text-align:left;word-break:break-word;}'
        '.twh-urgency{font-size:11px;font-weight:700;padding:4px 6px;'
        'text-transform:none;color:#1f1f1f;}'
        '.twh-status{padding:6px;border-top:1px solid #7a7a7a;}'
        '.twh-status-started{background:#b7e1b2;color:#1b5e20;}'
        '.twh-status-blocked{background:#e0e0e0;color:#424242;}'
        '.twh-status-normal{background:#ffffff;color:#1f1f1f;}'
        '.twh-title{display:block;margin-bottom:4px;}'
        '.twh-meta{display:flex;gap:6px;align-items:center;font-size:11px;}'
        '.twh-badge{font-weight:600;border:1px solid #666;border-radius:3px;'
        'padding:2px 6px;background:rgba(255,255,255,0.75);}"} }%%',
        'flowchart LR',
        '  classDef default fill:transparent,stroke:transparent,color:#1f1f1f;'
    ]
    used = set()

    for chain in chains:
        labels = []
        for uuid in chain:
            urgency_color = color_for_task(uid_map[uuid]) or "#cfe9f2"
            labels.append(node_label(uid_map[uuid], urgency_color))
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
        urgency_color = color_for_task(uid_map[uuid]) or "#cfe9f2"
        label = node_label(uid_map[uuid], urgency_color)
        lines.append(f'  {node_id(uuid)}["{label}"]')

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

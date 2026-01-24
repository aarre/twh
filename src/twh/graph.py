#!/usr/bin/env python3
"""
Taskwarrior dependency graph visualization using Mermaid.

This module provides functionality to convert Taskwarrior task exports into
Mermaid flowcharts and CSV files for import into other systems (e.g., Tana).
"""

import csv
from collections import defaultdict
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
    Generate Mermaid flowchart syntax from dependency chains.

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
        text = text.replace('"', '')
        text = text.replace('\\', '\\\\')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace('[', '(').replace(']', ')')
        return text

    def node_id(uuid: str) -> str:
        return f"t_{uuid.replace('-', '')}"

    lines = ['flowchart LR']
    used = set()

    for chain in chains:
        labels = []
        for uuid in chain:
            desc = uid_map[uuid].get('description', '')
            # Clean description: remove newlines, escape quotes/special chars.
            labels.append(sanitize_label(desc))
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
        desc = uid_map[uuid].get('description', '')
        label = sanitize_label(desc)
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

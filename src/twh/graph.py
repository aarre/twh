#!/usr/bin/env python3
"""
Taskwarrior dependency graph visualization using Mermaid.

This module provides functionality to convert Taskwarrior task exports into
Mermaid flowcharts and CSV files for import into other systems (e.g., Tana).
"""

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


def _parse_taskwarrior_json(text: str) -> List[Dict]:
    """
    Parse taskwarrior JSON output that may be an array or line-delimited JSON.
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


def get_tasks_from_taskwarrior() -> List[Dict]:
    """
    Execute taskwarrior and return parsed task data.

    Returns
    -------
    List[Dict]
        List of pending tasks as dictionaries.

    Raises
    ------
    subprocess.CalledProcessError
        If taskwarrior command fails.
    json.JSONDecodeError
        If taskwarrior output cannot be parsed.
    """
    try:
        result = subprocess.run(
            ["task", "export"],
            capture_output=True,
            text=True,
            check=True
        )
        tasks = _parse_taskwarrior_json(result.stdout)
        # Filter to pending tasks only
        return [t for t in tasks if t.get("status") == "pending"]
    except subprocess.CalledProcessError as e:
        print(f"Error executing taskwarrior: {e}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing taskwarrior output: {e}", file=sys.stderr)
        raise


def read_tasks_from_json(json_data: str) -> List[Dict]:
    """
    Parse tasks from JSON string.

    Parameters
    ----------
    json_data : str
        JSON string containing task data.

    Returns
    -------
    List[Dict]
        List of tasks as dictionaries.
    """
    return _parse_taskwarrior_json(json_data)


def parse_dependencies(dep_field: Optional[str]) -> List[str]:
    """
    Parse the dependencies field from a task.

    Parameters
    ----------
    dep_field : Optional[str]
        The 'depends' field from a task, which can be a comma-separated
        string of UUIDs or None.

    Returns
    -------
    List[str]
        List of dependency UUIDs.
    """
    if not dep_field:
        return []
    if isinstance(dep_field, list):
        return dep_field
    return [x.strip() for x in str(dep_field).split(',') if x.strip()]


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
        if len(pred[u]) != 1:
            for s in succ.get(u, []):
                if (u, s) in visited:
                    continue
                chain = [u]
                cur = s
                while True:
                    chain.append(cur)
                    visited.add((chain[-2], cur))
                    if len(pred[cur]) == 1 and len(succ[cur]) == 1:
                        cur = next(iter(succ[cur]))
                        if (chain[-1], cur) in visited:
                            break
                    else:
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

    for chain in chains:
        labels = []
        for uuid in chain:
            desc = uid_map[uuid].get('description', '')
            # Clean description: remove newlines, escape quotes/special chars.
            labels.append(sanitize_label(desc))

        # Create arrow chain with stable node IDs and readable labels.
        mer = ' --> '.join([
            f'{node_id(chain[i])}["{labels[i]}"]'
            for i in range(len(chain))
        ])
        lines.append('  ' + mer)

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

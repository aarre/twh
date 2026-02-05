#!/usr/bin/env python3
"""
Collect time-criticality ordering for moves with minimal comparisons.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from .review import ReviewTask, load_pending_tasks
from .taskwarrior import (
    describe_missing_udas,
    filter_modified_zero_lines,
    missing_udas,
)


CRITICALITY_PATH_ENV = "TWH_CRITICALITY_PATH"
CRITICALITY_STORE_VERSION = 1


class CriticalityChoice(IntEnum):
    """
    User choice for criticality prompts.
    """

    LEFT = 1
    RIGHT = 2
    TIE = 3


@dataclass(frozen=True)
class CriticalityState:
    """
    Criticality graph state for moves in scope.

    Attributes
    ----------
    tasks : Dict[str, ReviewTask]
        Mapping from UUID to move.
    more_critical : Dict[str, Set[str]]
        Directed edges for "more critical than" relations.
    ties : Set[frozenset[str]]
        Paired ties (no ordering).
    """

    tasks: Dict[str, ReviewTask]
    more_critical: Dict[str, Set[str]]
    ties: Set[frozenset[str]]


def criticality_store_path() -> Path:
    """
    Resolve the on-disk store for criticality comparisons.

    Returns
    -------
    Path
        Path to the comparison cache file.
    """
    override = os.environ.get(CRITICALITY_PATH_ENV)
    if override:
        return Path(os.path.expandvars(os.path.expanduser(override)))
    return Path.home() / ".config" / "twh" / "criticality.json"


def load_criticality_store(path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Load criticality comparison records from disk.

    Parameters
    ----------
    path : Optional[Path], optional
        Override path to load (default: resolved store path).

    Returns
    -------
    Dict[str, Dict[str, str]]
        Mapping with ``pairs`` for stored comparisons.
    """
    store_path = path or criticality_store_path()
    if not store_path.exists():
        return {"pairs": {}}
    raw = store_path.read_text(encoding="utf-8")
    data = json.loads(raw) if raw.strip() else {}
    pairs = data.get("pairs")
    if not isinstance(pairs, dict):
        pairs = {}
    return {"pairs": {str(key): str(value) for key, value in pairs.items()}}


def save_criticality_store(
    store: Dict[str, Dict[str, str]],
    path: Optional[Path] = None,
) -> None:
    """
    Persist criticality comparison records to disk.

    Parameters
    ----------
    store : Dict[str, Dict[str, str]]
        Store contents containing ``pairs``.
    path : Optional[Path], optional
        Override path to save (default: resolved store path).
    """
    store_path = path or criticality_store_path()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CRITICALITY_STORE_VERSION,
        "pairs": store.get("pairs", {}),
    }
    store_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _pair_key(left: str, right: str) -> tuple[str, str, bool]:
    if left <= right:
        return left, right, True
    return right, left, False


def record_criticality_comparison(
    left_uuid: str,
    right_uuid: str,
    choice: "CriticalityChoice",
    path: Optional[Path] = None,
) -> None:
    """
    Record a criticality comparison decision for recovery.

    Parameters
    ----------
    left_uuid : str
        Left move UUID.
    right_uuid : str
        Right move UUID.
    choice : CriticalityChoice
        Comparison outcome.
    path : Optional[Path], optional
        Override path to save (default: resolved store path).
    """
    if left_uuid == right_uuid:
        return
    store = load_criticality_store(path)
    left, right, left_is_first = _pair_key(left_uuid, right_uuid)
    if choice == CriticalityChoice.TIE:
        relation = "tie"
    elif choice == CriticalityChoice.LEFT:
        relation = "first" if left_is_first else "second"
    else:
        relation = "second" if left_is_first else "first"
    store.setdefault("pairs", {})[f"{left}|{right}"] = relation
    save_criticality_store(store, path)


def apply_saved_comparisons(
    state: CriticalityState,
    path: Optional[Path] = None,
) -> None:
    """
    Apply stored criticality comparisons to the current state.

    Parameters
    ----------
    state : CriticalityState
        Criticality graph state to update.
    path : Optional[Path], optional
        Override path to load (default: resolved store path).
    """
    store = load_criticality_store(path)
    pairs = store.get("pairs", {})
    for key, relation in pairs.items():
        if "|" not in key:
            continue
        left, right = key.split("|", 1)
        if left not in state.tasks or right not in state.tasks:
            continue
        if relation == "tie":
            state.ties.add(frozenset({left, right}))
            continue
        if relation == "first":
            state.more_critical.setdefault(left, set()).add(right)
            continue
        if relation == "second":
            state.more_critical.setdefault(right, set()).add(left)


def build_criticality_state(tasks: Iterable[ReviewTask]) -> CriticalityState:
    """
    Build an empty criticality state from move metadata.

    Parameters
    ----------
    tasks : Iterable[ReviewTask]
        Moves to include.

    Returns
    -------
    CriticalityState
        Empty state ready for comparisons.
    """
    task_list = list(tasks)
    task_map = {task.uuid: task for task in task_list}
    more_critical: Dict[str, Set[str]] = {task.uuid: set() for task in task_list}
    state = CriticalityState(tasks=task_map, more_critical=more_critical, ties=set())
    apply_saved_comparisons(state)
    return state


def _criticality_path(state: CriticalityState, start: str, target: str) -> bool:
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in state.more_critical.get(node, set()):
            if neighbor == target:
                return True
            if neighbor not in visited:
                stack.append(neighbor)
    return False


def _compute_reachability(
    state: CriticalityState,
    uuids: Sequence[str],
) -> Dict[str, Set[str]]:
    """
    Compute criticality reachability for each move UUID.

    Parameters
    ----------
    state : CriticalityState
        Criticality graph state.
    uuids : Sequence[str]
        UUIDs to include in reachability calculations.

    Returns
    -------
    Dict[str, Set[str]]
        Mapping from UUID to the set of UUIDs it outranks.

    Examples
    --------
    >>> from twh.review import ReviewTask
    >>> move_a = ReviewTask("a", 1, "A", None, [], 1, 1, 1)
    >>> move_b = ReviewTask("b", 2, "B", None, [], 1, 1, 1)
    >>> state = CriticalityState(
    ...     tasks={"a": move_a, "b": move_b},
    ...     more_critical={"a": {"b"}, "b": set()},
    ...     ties=set(),
    ... )
    >>> _compute_reachability(state, ["a", "b"])["a"] == {"b"}
    True
    """
    reachability: Dict[str, Set[str]] = {}
    uuid_set = set(uuids)
    visiting: Set[str] = set()

    def dfs(node: str) -> Set[str]:
        if node in reachability:
            return reachability[node]
        if node in visiting:
            return set()
        visiting.add(node)
        reachable: Set[str] = set()
        for neighbor in state.more_critical.get(node, set()):
            if neighbor not in uuid_set:
                continue
            reachable.add(neighbor)
            reachable.update(dfs(neighbor))
        visiting.remove(node)
        reachability[node] = reachable
        return reachable

    for uuid in uuids:
        dfs(uuid)
    return reachability


def criticality_relation(
    state: CriticalityState,
    left: str,
    right: str,
) -> str:
    """
    Determine the criticality relation between two moves.

    Parameters
    ----------
    state : CriticalityState
        Criticality graph state.
    left : str
        Left move UUID.
    right : str
        Right move UUID.

    Returns
    -------
    str
        One of "more_critical", "less_critical", "tie", or "unknown".
    """
    if left == right:
        return "tie"
    left_task = state.tasks.get(left)
    right_task = state.tasks.get(right)
    if left_task is not None and right_task is not None:
        left_value = left_task.criticality
        right_value = right_task.criticality
        if left_value is not None and right_value is not None:
            if left_value == right_value:
                return "tie"
            if left_value > right_value:
                return "more_critical"
            return "less_critical"
    if frozenset({left, right}) in state.ties:
        return "tie"
    if _criticality_path(state, left, right):
        return "more_critical"
    if _criticality_path(state, right, left):
        return "less_critical"
    return "unknown"


def criticality_missing_uuids(tasks: Iterable[ReviewTask]) -> Set[str]:
    """
    Identify moves missing criticality ratings.

    Parameters
    ----------
    tasks : Iterable[ReviewTask]
        Moves to check.

    Returns
    -------
    Set[str]
        UUIDs missing criticality values.
    """
    return {task.uuid for task in tasks if task.criticality is None}


def ensure_criticality_uda(
    get_setting: Optional[Callable[[str], Optional[str]]] = None,
) -> None:
    """
    Ensure the criticality UDA is configured before writing.

    Parameters
    ----------
    get_setting : Optional[Callable[[str], Optional[str]]], optional
        Getter for Taskwarrior settings.

    Raises
    ------
    RuntimeError
        If the criticality UDA is missing.
    """
    missing = missing_udas(
        ["criticality"],
        get_setting=get_setting,
        allow_taskrc_fallback=False,
    )
    if missing:
        raise RuntimeError(describe_missing_udas(missing))


def count_unknown_pairs(tasks: Sequence[ReviewTask], state: CriticalityState) -> int:
    """
    Count criticality pairs that still require user input.

    Parameters
    ----------
    tasks : Sequence[ReviewTask]
        Moves to evaluate.
    state : CriticalityState
        Criticality graph state.

    Returns
    -------
    int
        Number of unresolved criticality pairs.
    """
    task_list = list(tasks)
    if len(task_list) < 2:
        return 0
    uuid_list = [task.uuid for task in task_list]
    uuid_set = set(uuid_list)
    ties = state.ties
    reachability = _compute_reachability(state, uuid_list)
    reverse: Dict[str, Set[str]] = {uuid: set() for uuid in uuid_list}
    for source, targets in reachability.items():
        for target in targets:
            if target in reverse:
                reverse[target].add(source)
    tie_map: Dict[str, Set[str]] = {uuid: set() for uuid in uuid_list}
    for pair in ties:
        left, right = tuple(pair)
        if left in uuid_set and right in uuid_set:
            tie_map[left].add(right)
            tie_map[right].add(left)
    valued = {
        task.uuid
        for task in task_list
        if task.criticality is not None
    }
    total_unknown = 0
    comparable_count = len(uuid_list) - 1
    for uuid in uuid_list:
        comparable: Set[str] = set()
        if uuid in valued:
            comparable.update(valued)
            comparable.discard(uuid)
        comparable.update(reachability.get(uuid, set()))
        comparable.update(reverse.get(uuid, set()))
        comparable.update(tie_map.get(uuid, set()))
        total_unknown += max(0, comparable_count - len(comparable))
    return total_unknown // 2


def make_progress_chooser(
    tasks: Sequence[ReviewTask],
    state: CriticalityState,
    input_func: Callable[[str], str] = input,
) -> Callable[[ReviewTask, ReviewTask], CriticalityChoice]:
    """
    Build a criticality chooser that reports progress.

    Parameters
    ----------
    tasks : Sequence[ReviewTask]
        Moves in scope.
    state : CriticalityState
        Criticality graph state.
    input_func : Callable[[str], str], optional
        Input function for prompts.

    Returns
    -------
    Callable[[ReviewTask, ReviewTask], CriticalityChoice]
        Chooser that prints progress updates before prompting.
    """
    total = count_unknown_pairs(tasks, state)

    def chooser(left: ReviewTask, right: ReviewTask) -> CriticalityChoice:
        remaining = count_unknown_pairs(tasks, state)
        if total > 0:
            completed = max(0, total - remaining)
            percent_remaining = int(round((remaining / total) * 100))
            print(
                "Criticality progress: approximately "
                f"{completed} of {total} comparisons complete "
                f"({percent_remaining}% remaining)."
            )
        return prompt_criticality_choice(left, right, input_func=input_func)

    return chooser


def format_move_label(task: ReviewTask, label: str) -> str:
    """
    Format a move label for criticality prompts.

    Parameters
    ----------
    task : ReviewTask
        Move to label.
    label : str
        Prompt label (A/B).

    Returns
    -------
    str
        Formatted label text.
    """
    move_id = str(task.id) if task.id is not None else task.uuid[:8]
    description = task.description.strip()
    if description:
        return f"[{label}] Move ID {move_id}: {description}"
    return f"[{label}] Move ID {move_id}"


def prompt_criticality_choice(
    left: ReviewTask,
    right: ReviewTask,
    input_func: Callable[[str], str] = input,
) -> CriticalityChoice:
    """
    Ask the user to decide criticality between two moves.
    """
    print(format_move_label(left, "A"))
    print(format_move_label(right, "B"))
    print("Time-criticality measures how fast the value of a move decays if you wait.")
    print("Which move becomes pointless first if ignored?")
    print("[A] Move A, [B] Move B, [C] Tie")
    while True:
        choice = input_func("Selection (A/B/C): ").strip().upper()
        if choice == "A":
            return CriticalityChoice.LEFT
        if choice == "B":
            return CriticalityChoice.RIGHT
        if choice == "C":
            return CriticalityChoice.TIE
        print("Please enter A, B, or C.")


def _record_tie(state: CriticalityState, left: str, right: str) -> None:
    state.ties.add(frozenset({left, right}))
    state.more_critical.get(left, set()).discard(right)
    state.more_critical.get(right, set()).discard(left)


def _record_order(state: CriticalityState, higher: str, lower: str) -> None:
    if higher == lower:
        return
    state.more_critical.setdefault(higher, set()).add(lower)
    state.ties.discard(frozenset({higher, lower}))


def compare_moves(
    state: CriticalityState,
    left: ReviewTask,
    right: ReviewTask,
    chooser: Callable[[ReviewTask, ReviewTask], CriticalityChoice],
) -> int:
    """
    Compare two moves and update criticality state when needed.

    Returns
    -------
    int
        -1 when left is more critical, 1 when right is more critical, 0 for tie.
    """
    relation = criticality_relation(state, left.uuid, right.uuid)
    if relation == "more_critical":
        return -1
    if relation == "less_critical":
        return 1
    if relation == "tie":
        return 0

    choice = chooser(left, right)
    if choice == CriticalityChoice.LEFT:
        _record_order(state, left.uuid, right.uuid)
        record_criticality_comparison(left.uuid, right.uuid, choice)
        return -1
    if choice == CriticalityChoice.RIGHT:
        _record_order(state, right.uuid, left.uuid)
        record_criticality_comparison(left.uuid, right.uuid, choice)
        return 1

    _record_tie(state, left.uuid, right.uuid)
    record_criticality_comparison(left.uuid, right.uuid, choice)
    return 0


def sort_into_tiers(
    tasks: Iterable[ReviewTask],
    state: CriticalityState,
    chooser: Callable[[ReviewTask, ReviewTask], CriticalityChoice],
) -> List[List[ReviewTask]]:
    """
    Sort moves into criticality tiers using binary insertion.

    Parameters
    ----------
    tasks : Iterable[ReviewTask]
        Moves to sort.
    state : CriticalityState
        Criticality graph state.
    chooser : Callable[[ReviewTask, ReviewTask], CriticalityChoice]
        Chooser for unresolved comparisons.

    Returns
    -------
    List[List[ReviewTask]]
        Ordered tiers (most critical first).
    """
    tiers: List[List[ReviewTask]] = []
    ordered = sorted(
        tasks,
        key=lambda task: (task.id if task.id is not None else 10**9, task.uuid),
    )
    for task in ordered:
        if not tiers:
            tiers.append([task])
            continue
        lo = 0
        hi = len(tiers) - 1
        placed = False
        while lo <= hi:
            mid = (lo + hi) // 2
            rep = tiers[mid][0]
            comparison = compare_moves(state, task, rep, chooser)
            if comparison == 0:
                tiers[mid].append(task)
                placed = True
                break
            if comparison < 0:
                hi = mid - 1
            else:
                lo = mid + 1
        if not placed:
            tiers.insert(lo, [task])
    return tiers


def build_criticality_updates(
    tiers: List[List[ReviewTask]],
) -> Dict[str, float]:
    """
    Build criticality updates from ordered tiers.

    Parameters
    ----------
    tiers : List[List[ReviewTask]]
        Ordered tiers from most to least critical.

    Returns
    -------
    Dict[str, float]
        Criticality values keyed by move UUID.

    Examples
    --------
    >>> from twh.review import ReviewTask
    >>> tiers = [
    ...     [ReviewTask("a", 1, "A", None, [], 1, 1, 1)],
    ...     [ReviewTask("b", 2, "B", None, [], 1, 1, 1)],
    ... ]
    >>> updates = build_criticality_updates(tiers)
    >>> updates["a"] > updates["b"]
    True
    """
    updates: Dict[str, float] = {}
    if not tiers:
        return updates
    if len(tiers) == 1:
        value = 10.0
        for task in tiers[0]:
            updates[task.uuid] = value
        return updates
    step = 10.0 / (len(tiers) - 1)
    for tier_idx, tier in enumerate(tiers):
        value = 10.0 - (step * tier_idx)
        value = max(0.0, min(10.0, value))
        for task in tier:
            updates[task.uuid] = value
    return updates


def apply_criticality_updates(
    updates: Dict[str, float],
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    get_setting: Optional[Callable[[str], Optional[str]]] = None,
    quiet: bool = False,
) -> None:
    """
    Apply criticality updates to Taskwarrior.

    Parameters
    ----------
    updates : Dict[str, float]
        Criticality values keyed by move UUID.
    runner : Optional[Callable[..., subprocess.CompletedProcess]]
        Runner to execute Taskwarrior commands (args exclude ``task``).
    get_setting : Optional[Callable[[str], Optional[str]]]
        Getter for Taskwarrior settings.
    quiet : bool, optional
        Suppress Taskwarrior stdout when True (default: False).
    """
    if not updates:
        return
    if get_setting is None:
        from .taskwarrior import get_taskwarrior_setting

        get_setting = get_taskwarrior_setting
    missing = missing_udas(
        ["criticality"],
        get_setting=get_setting,
        allow_taskrc_fallback=False,
    )
    if missing:
        raise RuntimeError(describe_missing_udas(missing))
    if runner is None:
        def task_runner(args, **kwargs):
            return subprocess.run(["task", *args], **kwargs)

        runner = task_runner
    for uuid, value in updates.items():
        value_text = f"{value:.2f}".rstrip("0").rstrip(".")
        result = runner(
            [uuid, "modify", f"criticality:{value_text}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if not quiet:
            for line in filter_modified_zero_lines(result.stdout):
                print(line)
        if result.stderr:
            for line in filter_modified_zero_lines(result.stderr):
                print(line, file=sys.stderr)


def run_criticality(
    filters: Optional[Sequence[str]] = None,
    input_func: Callable[[str], str] = input,
    quiet: bool = False,
) -> int:
    """
    Run the criticality collection workflow for pending moves.

    Parameters
    ----------
    filters : Optional[Sequence[str]]
        Taskwarrior filter tokens to scope the move set.
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).
    quiet : bool, optional
        Suppress informational output when True (default: False).

    Returns
    -------
    int
        Exit code.
    """
    pending = load_pending_tasks(filters=filters)
    if not pending:
        if not quiet:
            print("No pending moves found.")
        return 0

    try:
        ensure_criticality_uda()
    except RuntimeError as exc:
        print(f"twh: criticality failed: {exc}", file=sys.stderr)
        return 1

    missing = criticality_missing_uuids(pending)
    if not missing:
        if not quiet:
            print("Criticality is already complete for these moves.")
        return 0

    state = build_criticality_state(pending)
    tiers = sort_into_tiers(
        pending,
        state,
        chooser=make_progress_chooser(pending, state, input_func=input_func),
    )
    updates = build_criticality_updates(tiers)
    try:
        apply_criticality_updates(updates, quiet=quiet)
    except RuntimeError as exc:
        print(f"twh: criticality failed: {exc}", file=sys.stderr)
        return 1
    if not quiet:
        print(f"Criticality updated for {len(pending)} moves.")
    return 0

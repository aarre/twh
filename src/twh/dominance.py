#!/usr/bin/env python3
"""
Collect dominance ordering for moves with minimal comparisons.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .review import ReviewTask, load_pending_tasks
from .taskwarrior import (
    describe_missing_udas,
    filter_modified_zero_lines,
    missing_udas,
)


class DominanceChoice(IntEnum):
    """
    User choice for dominance prompts.
    """

    LEFT = 1
    RIGHT = 2
    TIE = 3


@dataclass(frozen=True)
class DominanceState:
    """
    Dominance graph state for moves in scope.

    Attributes
    ----------
    tasks : Dict[str, ReviewTask]
        Mapping from UUID to move.
    dominates : Dict[str, Set[str]]
        Directed dominance edges.
    ties : Set[frozenset[str]]
        Paired ties (no dominance).
    """

    tasks: Dict[str, ReviewTask]
    dominates: Dict[str, Set[str]]
    ties: Set[frozenset[str]]


@dataclass(frozen=True)
class DominanceUpdate:
    """
    Dominance updates for a move.

    Attributes
    ----------
    dominates : List[str]
        UUIDs of moves dominated by this move.
    dominated_by : List[str]
        UUIDs that dominate this move.
    """

    dominates: List[str]
    dominated_by: List[str]


def build_dominance_state(tasks: Iterable[ReviewTask]) -> DominanceState:
    """
    Build dominance state from move metadata and dependencies.

    Parameters
    ----------
    tasks : Iterable[ReviewTask]
        Moves to include.

    Returns
    -------
    DominanceState
        State containing dominance edges and ties.
    """
    task_list = list(tasks)
    task_map = {task.uuid: task for task in task_list}
    id_map = {
        str(task.id): task.uuid for task in task_list if task.id is not None
    }
    dominates: Dict[str, Set[str]] = {task.uuid: set() for task in task_list}

    for task in task_list:
        for dep in task.depends:
            resolved = _resolve_uuid(dep, task_map, id_map)
            if resolved:
                dominates[resolved].add(task.uuid)
        for dom in task.dominates:
            resolved = _resolve_uuid(dom, task_map, id_map)
            if resolved:
                dominates[task.uuid].add(resolved)
        for dom_by in task.dominated_by:
            resolved = _resolve_uuid(dom_by, task_map, id_map)
            if resolved:
                dominates[resolved].add(task.uuid)

    ties: Set[frozenset[str]] = set()
    for left, edges in dominates.items():
        for right in list(edges):
            if left == right:
                edges.discard(right)
                continue
            if left in dominates.get(right, set()):
                edges.discard(right)
                dominates[right].discard(left)
                ties.add(frozenset({left, right}))

    return DominanceState(tasks=task_map, dominates=dominates, ties=ties)


def _resolve_uuid(
    value: str,
    task_map: Dict[str, ReviewTask],
    id_map: Dict[str, str],
) -> Optional[str]:
    if value in task_map:
        return value
    return id_map.get(str(value))


def _dominates_path(state: DominanceState, start: str, target: str) -> bool:
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in state.dominates.get(node, set()):
            if neighbor == target:
                return True
            if neighbor not in visited:
                stack.append(neighbor)
    return False


def _compute_reachability(
    state: DominanceState,
    uuids: Sequence[str],
) -> Dict[str, Set[str]]:
    """
    Compute dominance reachability for each move UUID.

    Parameters
    ----------
    state : DominanceState
        Dominance graph state.
    uuids : Sequence[str]
        UUIDs to include in reachability calculations.

    Returns
    -------
    Dict[str, Set[str]]
        Mapping from UUID to the set of UUIDs it dominates via paths.

    Examples
    --------
    >>> from twh.review import ReviewTask
    >>> move_a = ReviewTask("a", 1, "A", None, [], 1, 1, 1)
    >>> move_b = ReviewTask("b", 2, "B", None, [], 1, 1, 1)
    >>> state = DominanceState(
    ...     tasks={"a": move_a, "b": move_b},
    ...     dominates={"a": {"b"}, "b": set()},
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
        for neighbor in state.dominates.get(node, set()):
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


def dominance_relation(state: DominanceState, left: str, right: str) -> str:
    """
    Determine the dominance relation between two moves.

    Parameters
    ----------
    state : DominanceState
        Dominance graph state.
    left : str
        Left move UUID.
    right : str
        Right move UUID.

    Returns
    -------
    str
        One of "dominates", "dominated_by", "tie", or "unknown".
    """
    if left == right:
        return "tie"
    if frozenset({left, right}) in state.ties:
        return "tie"
    if _dominates_path(state, left, right):
        return "dominates"
    if _dominates_path(state, right, left):
        return "dominated_by"
    return "unknown"


def dominance_missing_uuids(
    tasks: Iterable[ReviewTask],
    state: DominanceState,
) -> Set[str]:
    """
    Identify moves missing dominance ordering.

    Parameters
    ----------
    tasks : Iterable[ReviewTask]
        Moves to check.
    state : DominanceState
        Dominance graph state.

    Returns
    -------
    Set[str]
        UUIDs missing dominance information.
    """
    task_list = list(tasks)
    if len(task_list) < 2:
        return set()
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
    missing: Set[str] = set()
    comparable_count = len(uuid_list) - 1
    for uuid in uuid_list:
        comparable = (
            reachability.get(uuid, set())
            | reverse.get(uuid, set())
            | tie_map.get(uuid, set())
        )
        if len(comparable) < comparable_count:
            missing.add(uuid)
    return missing


def ensure_dominance_udas(
    get_setting: Optional[Callable[[str], Optional[str]]] = None,
) -> None:
    """
    Ensure dominance UDAs are configured before writing updates.

    Parameters
    ----------
    get_setting : Optional[Callable[[str], Optional[str]]], optional
        Getter for Taskwarrior settings.

    Raises
    ------
    RuntimeError
        If required dominance UDAs are missing.
    """
    if get_setting is None:
        from .taskwarrior import get_taskwarrior_setting

        get_setting = get_taskwarrior_setting
    missing = missing_udas(
        ["dominates", "dominated_by"],
        get_setting=get_setting,
        allow_taskrc_fallback=False,
    )
    if missing:
        raise RuntimeError(describe_missing_udas(missing))


def build_incremental_updates(
    state: DominanceState,
    uuids: Iterable[str],
) -> Dict[str, DominanceUpdate]:
    """
    Build dominance updates for a subset of moves.

    Parameters
    ----------
    state : DominanceState
        Dominance graph state.
    uuids : Iterable[str]
        UUIDs to update.

    Returns
    -------
    Dict[str, DominanceUpdate]
        Incremental updates keyed by UUID.
    """
    uuid_set = {uuid for uuid in uuids if uuid in state.tasks}
    if not uuid_set:
        return {}
    incoming: Dict[str, Set[str]] = {uuid: set() for uuid in state.tasks}
    for src, edges in state.dominates.items():
        for target in edges:
            if target in incoming:
                incoming[target].add(src)
    tie_map: Dict[str, Set[str]] = {uuid: set() for uuid in state.tasks}
    for pair in state.ties:
        left, right = tuple(pair)
        if left in tie_map and right in tie_map:
            tie_map[left].add(right)
            tie_map[right].add(left)
    updates: Dict[str, DominanceUpdate] = {}
    for uuid in uuid_set:
        dominates = sorted(state.dominates.get(uuid, set()))
        dominated_by = sorted(incoming.get(uuid, set()) | tie_map.get(uuid, set()))
        updates[uuid] = DominanceUpdate(
            dominates=dominates,
            dominated_by=dominated_by,
        )
    return updates


def count_unknown_pairs(tasks: Sequence[ReviewTask], state: DominanceState) -> int:
    """
    Count dominance pairs that still require user input.

    Parameters
    ----------
    tasks : Sequence[ReviewTask]
        Moves to evaluate.
    state : DominanceState
        Dominance graph state.

    Returns
    -------
    int
        Number of unresolved dominance pairs.
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
    total_unknown = 0
    comparable_count = len(uuid_list) - 1
    for uuid in uuid_list:
        comparable = (
            reachability.get(uuid, set())
            | reverse.get(uuid, set())
            | tie_map.get(uuid, set())
        )
        total_unknown += max(0, comparable_count - len(comparable))
    return total_unknown // 2


def make_progress_chooser(
    tasks: Sequence[ReviewTask],
    state: DominanceState,
    input_func: Callable[[str], str] = input,
) -> Callable[[ReviewTask, ReviewTask], DominanceChoice]:
    """
    Build a dominance chooser that reports progress.

    Parameters
    ----------
    tasks : Sequence[ReviewTask]
        Moves in scope.
    state : DominanceState
        Dominance graph state.
    input_func : Callable[[str], str], optional
        Input function for prompts.

    Returns
    -------
    Callable[[ReviewTask, ReviewTask], DominanceChoice]
        Chooser that prints progress updates before prompting.
    """
    total = count_unknown_pairs(tasks, state)

    def chooser(left: ReviewTask, right: ReviewTask) -> DominanceChoice:
        remaining = count_unknown_pairs(tasks, state)
        if total > 0:
            completed = max(0, total - remaining)
            percent_remaining = int(round((remaining / total) * 100))
            print(
                "Dominance progress: approximately "
                f"{completed} of {total} comparisons complete "
                f"({percent_remaining}% remaining)."
            )
        return prompt_dominance_choice(left, right, input_func=input_func)

    return chooser


def format_move_label(task: ReviewTask, label: str) -> str:
    """
    Format a move label for dominance prompts.

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


def prompt_dominance_choice(
    left: ReviewTask,
    right: ReviewTask,
    input_func: Callable[[str], str] = input,
) -> DominanceChoice:
    """
    Ask the user to decide dominance between two moves.
    """
    print(format_move_label(left, "A"))
    print(format_move_label(right, "B"))
    print(
        "Which of these moves dominates the other? (Which move clearly must be completed "
        "before the other, all things considered?) For example, Does one move make another "
        "easier, cheaper, or unnecessary? Does delaying a move foreclose something (missed "
        "feedback, lost window, stale data)?"
    )
    print("[A] Move A dominates, [B] Move B dominates, [C] Neither dominates (it is a tie)")
    while True:
        choice = input_func("Selection (A/B/C): ").strip().upper()
        if choice == "A":
            return DominanceChoice.LEFT
        if choice == "B":
            return DominanceChoice.RIGHT
        if choice == "C":
            return DominanceChoice.TIE
        print("Please enter A, B, or C.")


def _record_tie(state: DominanceState, left: str, right: str) -> None:
    state.ties.add(frozenset({left, right}))
    state.dominates.get(left, set()).discard(right)
    state.dominates.get(right, set()).discard(left)


def _record_dominance(state: DominanceState, dominant: str, dominated: str) -> None:
    if dominant == dominated:
        return
    state.dominates.setdefault(dominant, set()).add(dominated)
    state.ties.discard(frozenset({dominant, dominated}))


def compare_moves(
    state: DominanceState,
    left: ReviewTask,
    right: ReviewTask,
    chooser: Callable[[ReviewTask, ReviewTask], DominanceChoice],
    on_update: Optional[Callable[[DominanceState, Iterable[str]], None]] = None,
) -> int:
    """
    Compare two moves and update dominance state when needed.

    Returns
    -------
    int
        -1 when left dominates, 1 when right dominates, 0 for tie.
    """
    relation = dominance_relation(state, left.uuid, right.uuid)
    if relation == "dominates":
        return -1
    if relation == "dominated_by":
        return 1
    if relation == "tie":
        return 0

    choice = chooser(left, right)
    if choice == DominanceChoice.LEFT:
        _record_dominance(state, left.uuid, right.uuid)
        if on_update is not None:
            on_update(state, {left.uuid, right.uuid})
        return -1
    if choice == DominanceChoice.RIGHT:
        _record_dominance(state, right.uuid, left.uuid)
        if on_update is not None:
            on_update(state, {left.uuid, right.uuid})
        return 1

    _record_tie(state, left.uuid, right.uuid)
    if on_update is not None:
        on_update(state, {left.uuid, right.uuid})
    return 0


def sort_into_tiers(
    tasks: Iterable[ReviewTask],
    state: DominanceState,
    chooser: Callable[[ReviewTask, ReviewTask], DominanceChoice],
    on_update: Optional[Callable[[DominanceState, Iterable[str]], None]] = None,
) -> List[List[ReviewTask]]:
    """
    Sort moves into dominance tiers using binary insertion.

    Parameters
    ----------
    tasks : Iterable[ReviewTask]
        Moves to sort.
    state : DominanceState
        Dominance graph state.
    chooser : Callable[[ReviewTask, ReviewTask], DominanceChoice]
        Chooser for unresolved comparisons.

    Returns
    -------
    List[List[ReviewTask]]
        Ordered tiers (highest dominance first).
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
            comparison = compare_moves(
                state,
                task,
                rep,
                chooser,
                on_update=on_update,
            )
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


def build_dominance_updates(tiers: List[List[ReviewTask]]) -> Dict[str, DominanceUpdate]:
    """
    Build dominance updates from ordered tiers.

    Parameters
    ----------
    tiers : List[List[ReviewTask]]
        Ordered tiers from most to least dominant.

    Returns
    -------
    Dict[str, DominanceUpdate]
        Updates keyed by move UUID.
    """
    updates: Dict[str, DominanceUpdate] = {}
    for tier_idx, tier in enumerate(tiers):
        upper = [task.uuid for upper_tier in tiers[:tier_idx] for task in upper_tier]
        lower = [task.uuid for lower_tier in tiers[tier_idx + 1 :] for task in lower_tier]
        for task in tier:
            # Persist ties without marking moves as dominated.
            ties = [other.uuid for other in tier if other.uuid != task.uuid]
            updates[task.uuid] = DominanceUpdate(
                dominates=[uuid for uuid in lower if uuid != task.uuid],
                dominated_by=[uuid for uuid in upper if uuid != task.uuid] + ties,
            )
    return updates


def apply_dominance_updates(
    updates: Dict[str, DominanceUpdate],
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    get_setting: Optional[Callable[[str], Optional[str]]] = None,
    quiet: bool = False,
    validate_udas: bool = True,
) -> None:
    """
    Apply dominance updates to Taskwarrior.

    Parameters
    ----------
    updates : Dict[str, DominanceUpdate]
        Updates keyed by move UUID.
    runner : Optional[Callable[..., subprocess.CompletedProcess]]
        Runner to execute Taskwarrior commands (args exclude ``task``).
    get_setting : Optional[Callable[[str], Optional[str]]]
        Getter for Taskwarrior settings.
    quiet : bool, optional
        Suppress Taskwarrior stdout when True (default: False).
    """
    if get_setting is None:
        from .taskwarrior import get_taskwarrior_setting

        get_setting = get_taskwarrior_setting
    if validate_udas:
        missing = missing_udas(
            ["dominates", "dominated_by"],
            get_setting=get_setting,
            allow_taskrc_fallback=False,
        )
        if missing:
            raise RuntimeError(describe_missing_udas(missing))
    if runner is None:
        def task_runner(args, **kwargs):
            return subprocess.run(["task", *args], **kwargs)

        runner = task_runner
    for uuid, update in updates.items():
        dominates_value = ",".join(update.dominates)
        dominated_by_value = ",".join(update.dominated_by)
        result = runner(
            [
                uuid,
                "modify",
                f"dominates:{dominates_value}",
                f"dominated_by:{dominated_by_value}",
            ],
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


def build_tiers_from_state(
    tasks: Iterable[ReviewTask],
    state: DominanceState,
) -> List[List[ReviewTask]]:
    """
    Build dominance tiers from existing dominance edges.

    Parameters
    ----------
    tasks : Iterable[ReviewTask]
        Moves to include.
    state : DominanceState
        Dominance graph state.

    Returns
    -------
    List[List[ReviewTask]]
        Dominance tiers.
    """
    task_list = list(tasks)
    uuid_set = {task.uuid for task in task_list}
    incoming: Dict[str, Set[str]] = {task.uuid: set() for task in task_list}
    outgoing: Dict[str, Set[str]] = {task.uuid: set() for task in task_list}

    for left, edges in state.dominates.items():
        if left not in uuid_set:
            continue
        for right in edges:
            if right not in uuid_set:
                continue
            outgoing[left].add(right)
            incoming[right].add(left)

    remaining = set(uuid_set)
    tiers: List[List[ReviewTask]] = []
    while remaining:
        tier = sorted(
            [uuid for uuid in remaining if not incoming[uuid]],
            key=lambda u: (state.tasks[u].id if state.tasks[u].id is not None else 10**9, u),
        )
        if not tier:
            tier = sorted(
                list(remaining),
                key=lambda u: (state.tasks[u].id if state.tasks[u].id is not None else 10**9, u),
            )
        tiers.append([state.tasks[uuid] for uuid in tier])
        for uuid in tier:
            remaining.discard(uuid)
            for child in outgoing[uuid]:
                incoming[child].discard(uuid)
    return tiers


def run_dominance(
    filters: Optional[Sequence[str]] = None,
    input_func: Callable[[str], str] = input,
    quiet: bool = False,
) -> int:
    """
    Run the dominance collection workflow for pending moves.

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

    state = build_dominance_state(pending)
    missing = dominance_missing_uuids(pending, state)
    if not missing:
        if not quiet:
            print("Dominance is already complete for these moves.")
        return 0

    try:
        ensure_dominance_udas()
    except RuntimeError as exc:
        print(f"twh: dominance failed: {exc}", file=sys.stderr)
        return 1

    updated = False

    def on_update(current_state: DominanceState, uuids: Iterable[str]) -> None:
        nonlocal updated
        updates = build_incremental_updates(current_state, uuids)
        if not updates:
            return
        apply_dominance_updates(
            updates,
            quiet=True,
            validate_udas=False,
        )
        updated = True

    _ = sort_into_tiers(
        pending,
        state,
        chooser=make_progress_chooser(pending, state, input_func=input_func),
        on_update=on_update,
    )
    if not quiet:
        if updated:
            print(f"Dominance updated for {len(pending)} moves.")
        else:
            print("Dominance comparisons completed without changes.")
    return 0

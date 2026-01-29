#!/usr/bin/env python3
"""
Review Taskwarrior metadata and suggest next actions.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .taskwarrior import parse_dependencies, read_tasks_from_json


@dataclass(frozen=True)
class ReviewTask:
    """
    Normalized Taskwarrior task data for review workflows.

    Attributes
    ----------
    uuid : str
        Task UUID.
    id : Optional[int]
        Task ID, if available.
    description : str
        Task description.
    project : Optional[str]
        Project name.
    depends : List[str]
        UUIDs of tasks this task depends on.
    imp : Optional[int]
        Importance horizon in days.
    urg : Optional[int]
        Urgency horizon in days.
    opt : Optional[int]
        Option value (0-10).
    mode : Optional[str]
        Mode associated with the task.
    dominates : List[str]
        UUIDs explicitly dominated by this task.
    raw : Dict[str, Any]
        Raw Taskwarrior payload.
    """

    uuid: str
    id: Optional[int]
    description: str
    project: Optional[str]
    depends: List[str]
    imp: Optional[int]
    urg: Optional[int]
    opt: Optional[int]
    mode: Optional[str]
    dominates: List[str]
    raw: Dict[str, Any]

    @staticmethod
    def from_json(payload: Dict[str, Any]) -> "ReviewTask":
        """
        Build a ReviewTask from Taskwarrior export data.

        Parameters
        ----------
        payload : Dict[str, Any]
            Taskwarrior export dictionary.

        Returns
        -------
        ReviewTask
            Parsed task instance.

        Examples
        --------
        >>> task = ReviewTask.from_json({"uuid": "u1", "description": "Test", "imp": "3"})
        >>> task.imp
        3
        """

        def parse_int(key: str) -> Optional[int]:
            value = payload.get(key)
            if value is None or value == "":
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def normalize_mode(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            normalized = str(value).strip()
            return normalized if normalized else None

        return ReviewTask(
            uuid=str(payload["uuid"]),
            id=payload.get("id"),
            description=str(payload.get("description", "")).strip(),
            project=payload.get("project"),
            depends=parse_dependencies(payload.get("depends")),
            imp=parse_int("imp"),
            urg=parse_int("urg"),
            opt=parse_int("opt"),
            mode=normalize_mode(payload.get("mode")),
            dominates=parse_uuid_list(payload.get("dominates")),
            raw=payload,
        )


@dataclass(frozen=True)
class MissingMetadata:
    """
    Record missing metadata for a review task.

    Attributes
    ----------
    task : ReviewTask
        Task with missing metadata.
    missing : Tuple[str, ...]
        Missing metadata field names.
    is_ready : bool
        True when the task is ready to act on.
    """

    task: ReviewTask
    missing: Tuple[str, ...]
    is_ready: bool


@dataclass(frozen=True)
class ScoredTask:
    """
    Task paired with its review score.

    Attributes
    ----------
    task : ReviewTask
        Task being scored.
    score : float
        Composite score for the task.
    components : Dict[str, float]
        Score components for diagnostics.
    """

    task: ReviewTask
    score: float
    components: Dict[str, float]


@dataclass(frozen=True)
class ReviewReport:
    """
    Summary of review findings.

    Attributes
    ----------
    missing : List[MissingMetadata]
        Tasks missing metadata.
    candidates : List[ScoredTask]
        Scored ready tasks.
    """

    missing: List[MissingMetadata]
    candidates: List[ScoredTask]


def run_task_command(
    args: Sequence[str],
    capture_output: bool = False,
    stdin: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """
    Execute a Taskwarrior command.

    Parameters
    ----------
    args : Sequence[str]
        Taskwarrior arguments excluding the executable.
    capture_output : bool, optional
        Whether to capture stdout/stderr (default: False).
    stdin : Optional[int], optional
        Optional stdin to supply to subprocess (default: None).

    Returns
    -------
    subprocess.CompletedProcess
        Process completion data.
    """
    kwargs: Dict[str, Any] = {"check": False}
    if capture_output:
        kwargs.update({"capture_output": True, "text": True})
    if stdin is not None:
        kwargs["stdin"] = stdin
    return subprocess.run(["task", *args], **kwargs)


def parse_uuid_list(value: Any) -> List[str]:
    """
    Parse UUID lists stored as strings or lists.

    Parameters
    ----------
    value : Any
        Value from Taskwarrior export.

    Returns
    -------
    List[str]
        Parsed UUID list.

    Examples
    --------
    >>> parse_uuid_list("a,b")
    ['a', 'b']
    >>> parse_uuid_list(["c"])
    ['c']
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def ready_tasks(pending: Iterable[ReviewTask]) -> List[ReviewTask]:
    """
    Return tasks that are ready (no pending dependencies).

    Parameters
    ----------
    pending : Iterable[ReviewTask]
        Pending tasks to evaluate.

    Returns
    -------
    List[ReviewTask]
        Ready tasks.
    """
    pending_list = list(pending)
    pending_uuids = {task.uuid for task in pending_list}
    ready: List[ReviewTask] = []
    for task in pending_list:
        # Dependency is blocking only if it is still pending.
        if any(dep in pending_uuids for dep in task.depends):
            continue
        ready.append(task)
    return ready


def missing_fields(task: ReviewTask) -> Tuple[str, ...]:
    """
    Return the missing metadata fields for a task.

    Parameters
    ----------
    task : ReviewTask
        Task to inspect.

    Returns
    -------
    Tuple[str, ...]
        Missing fields in canonical order.

    Examples
    --------
    >>> task = ReviewTask("u1", 1, "x", None, [], None, 2, None, None, [], {})
    >>> missing_fields(task)
    ('imp', 'opt', 'mode')
    """
    missing: List[str] = []
    if task.imp is None:
        missing.append("imp")
    if task.urg is None:
        missing.append("urg")
    if task.opt is None:
        missing.append("opt")
    if not task.mode:
        missing.append("mode")
    return tuple(missing)


def collect_missing_metadata(
    pending: Iterable[ReviewTask],
    ready: Iterable[ReviewTask],
) -> List[MissingMetadata]:
    """
    Collect tasks missing metadata, ordering ready items first.

    Parameters
    ----------
    pending : Iterable[ReviewTask]
        Pending tasks.
    ready : Iterable[ReviewTask]
        Ready tasks.

    Returns
    -------
    List[MissingMetadata]
        Missing metadata records.
    """
    ready_uuids = {task.uuid for task in ready}
    items: List[MissingMetadata] = []
    for task in pending:
        missing = missing_fields(task)
        if not missing:
            continue
        items.append(
            MissingMetadata(
                task=task,
                missing=missing,
                is_ready=task.uuid in ready_uuids,
            )
        )
    items.sort(
        key=lambda item: (
            not item.is_ready,
            item.task.project or "",
            item.task.id if item.task.id is not None else 10**9,
        )
    )
    return items


def dominated_set(tasks: Iterable[ReviewTask]) -> set[str]:
    """
    Return UUIDs explicitly dominated by other tasks.

    Parameters
    ----------
    tasks : Iterable[ReviewTask]
        Tasks to scan for dominance.

    Returns
    -------
    set[str]
        UUIDs that are dominated by other tasks.
    """
    dominated: set[str] = set()
    for task in tasks:
        dominated.update(task.dominates)
    return dominated


def mode_multiplier(current: Optional[str], required: Optional[str]) -> float:
    """
    Compute the multiplier based on task mode alignment.

    Parameters
    ----------
    current : Optional[str]
        Current mode.
    required : Optional[str]
        Task-required mode.

    Returns
    -------
    float
        Mode multiplier.
    """
    if not required:
        return 1.0
    if not current:
        return 0.95
    if current.strip().lower() == required.strip().lower():
        return 1.15
    return 0.85


def score_task(
    task: ReviewTask,
    current_mode: Optional[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Score a task based on importance, urgency, option value, and mode.

    Parameters
    ----------
    task : ReviewTask
        Task to score.
    current_mode : Optional[str]
        Current mode for scoring.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        Total score and component breakdown.
    """
    missing_penalty = 1.0
    if task.imp is None or task.urg is None:
        missing_penalty *= 0.92
    if task.opt is None:
        missing_penalty *= 0.96

    imp_days = max(0, task.imp or 0)
    urg_days = max(0, task.urg or 0)
    opt = max(0, min(10, task.opt if task.opt is not None else 0))

    imp_score = math.log1p(imp_days)
    urg_score = 1.0 / (urg_days + 1.0)
    opt_score = opt / 10.0

    mode_mult = mode_multiplier(current_mode, task.mode)
    base = (1.4 * urg_score) + (1.0 * imp_score) + (0.9 * opt_score)
    total = base * mode_mult * missing_penalty

    components = {
        "imp_score": imp_score,
        "urg_score": urg_score,
        "opt_score": opt_score,
        "mode_mult": mode_mult,
        "missing_mult": missing_penalty,
        "total": total,
    }
    return total, components


def format_task_rationale(task: ReviewTask, components: Dict[str, float]) -> str:
    """
    Format a task rationale block for review output.

    Parameters
    ----------
    task : ReviewTask
        Task to describe.
    components : Dict[str, float]
        Score component data.

    Returns
    -------
    str
        Multi-line rationale string.
    """
    imp = task.imp if task.imp is not None else "?"
    urg = task.urg if task.urg is not None else "?"
    opt = task.opt if task.opt is not None else "?"
    mode = task.mode or "-"
    proj = task.project or "-"
    task_id = str(task.id) if task.id is not None else task.uuid[:8]
    return (
        f"[{task_id}] {task.description}\n"
        f"  project={proj} mode={mode} imp={imp}d urg={urg}d opt={opt}\n"
        "  "
        f"score={components['total']:.3f} "
        f"(urg={components['urg_score']:.3f}, "
        f"imp={components['imp_score']:.3f}, "
        f"opt={components['opt_score']:.2f}, "
        f"mode*{components['mode_mult']:.2f}, "
        f"spec*{components['missing_mult']:.2f})"
    )


def filter_candidates(
    ready: Iterable[ReviewTask],
    current_mode: Optional[str],
    strict_mode: bool,
    include_dominated: bool,
) -> List[ReviewTask]:
    """
    Filter ready tasks based on mode and dominance rules.

    Parameters
    ----------
    ready : Iterable[ReviewTask]
        Ready tasks.
    current_mode : Optional[str]
        Current mode context.
    strict_mode : bool
        Require mode match when True.
    include_dominated : bool
        Keep dominated tasks when True.

    Returns
    -------
    List[ReviewTask]
        Filtered candidate tasks.
    """
    ready_list = list(ready)
    dominated = set() if include_dominated else dominated_set(ready_list)

    filtered: List[ReviewTask] = []
    for task in ready_list:
        if task.uuid in dominated:
            continue
        if current_mode:
            task_mode = (task.mode or "").lower()
            desired = current_mode.lower()
            if strict_mode and task_mode != desired:
                continue
            if not strict_mode and task.mode is not None and task_mode != desired:
                continue
        filtered.append(task)
    return filtered


def rank_candidates(
    candidates: Iterable[ReviewTask],
    current_mode: Optional[str],
    top: int,
) -> List[ScoredTask]:
    """
    Rank candidate tasks by score.

    Parameters
    ----------
    candidates : Iterable[ReviewTask]
        Tasks to score.
    current_mode : Optional[str]
        Current mode for scoring.
    top : int
        Number of top candidates to return.

    Returns
    -------
    List[ScoredTask]
        Ranked candidates.
    """
    scored: List[ScoredTask] = []
    for task in candidates:
        score, components = score_task(task, current_mode)
        scored.append(ScoredTask(task=task, score=score, components=components))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:max(0, top)]


def build_review_report(
    pending: Iterable[ReviewTask],
    current_mode: Optional[str],
    strict_mode: bool,
    include_dominated: bool,
    top: int,
) -> ReviewReport:
    """
    Build a review report from pending tasks.

    Parameters
    ----------
    pending : Iterable[ReviewTask]
        Pending tasks.
    current_mode : Optional[str]
        Current mode for filtering/scoring.
    strict_mode : bool
        Require strict mode matching when True.
    include_dominated : bool
        Include dominated tasks when True.
    top : int
        Number of candidates to include.

    Returns
    -------
    ReviewReport
        Review report data.
    """
    pending_list = list(pending)
    ready = ready_tasks(pending_list)
    missing = collect_missing_metadata(pending_list, ready)
    candidates = filter_candidates(
        ready,
        current_mode=current_mode,
        strict_mode=strict_mode,
        include_dominated=include_dominated,
    )
    ranked = rank_candidates(candidates, current_mode=current_mode, top=top)
    return ReviewReport(missing=missing, candidates=ranked)


def interactive_fill_missing(
    task: ReviewTask,
    input_func: Callable[[str], str] = input,
) -> Dict[str, str]:
    """
    Interactively prompt for missing metadata fields.

    Parameters
    ----------
    task : ReviewTask
        Task to update.
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).

    Returns
    -------
    Dict[str, str]
        Fields to update.
    """
    updates: Dict[str, str] = {}
    print("\nFill missing fields (press Enter to skip)")
    if task.imp is None:
        value = input_func("  imp days (importance horizon - for how many days will you remember whether you did this?): ").strip()
        if value:
            updates["imp"] = value
    if task.urg is None:
        value = input_func("  urg days (urgency horizon - how many days until doing this loses all value?): ").strip()
        if value:
            updates["urg"] = value
    if task.opt is None:
        value = input_func("  opt 0-10 (option value - to what extent does this move, "
                           "if taken now, preserve or increase my future options? For example, does it unblock multiple downstream tasks; "
                           "clarify scope, structure, or uncertainty; generate artifacts others can react to; "
                           "or reduce coordination risk (waiting on others, data access, approvals)): ").strip()
        if value:
            updates["opt"] = value
    if not task.mode:
        value = input_func("  mode (e.g., analysis/structural/developmental/writing/illustration/copyediting/programming/mechanical/errand/chore): ").strip()
        if value:
            updates["mode"] = value
    return updates


def apply_updates(uuid: str, updates: Dict[str, str]) -> None:
    """
    Apply updates to a Taskwarrior task.

    Parameters
    ----------
    uuid : str
        Task UUID.
    updates : Dict[str, str]
        Field updates.

    Returns
    -------
    None
        Updates are applied to Taskwarrior.
    """
    if not updates:
        return
    parts = [f"{key}:{value}" for key, value in updates.items()]
    run_task_command([uuid, "modify", *parts])


def load_pending_tasks() -> List[ReviewTask]:
    """
    Load pending tasks from Taskwarrior export.

    Returns
    -------
    List[ReviewTask]
        Pending tasks.

    Raises
    ------
    RuntimeError
        If Taskwarrior export fails.
    """
    result = run_task_command(["status:pending", "export"], capture_output=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or "Taskwarrior export failed.")
    tasks = read_tasks_from_json(result.stdout or "")
    return [
        ReviewTask.from_json(task)
        for task in tasks
        if isinstance(task, dict) and task.get("uuid")
    ]


def run_review(
    *,
    mode: Optional[str],
    limit: int,
    top: int,
    strict_mode: bool,
    include_dominated: bool,
    wizard: bool,
    wizard_once: bool,
    input_func: Callable[[str], str] = input,
) -> int:
    """
    Execute the review flow: missing metadata + recommendations.

    Parameters
    ----------
    mode : Optional[str]
        Current mode context.
    limit : int
        Max missing tasks to list.
    top : int
        Number of top candidates to show.
    strict_mode : bool
        Require mode match when True.
    include_dominated : bool
        Include dominated tasks when True.
    wizard : bool
        Prompt for missing metadata when True.
    wizard_once : bool
        Only prompt for the first ready task when True.
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).

    Returns
    -------
    int
        Exit code.
    """
    try:
        pending = load_pending_tasks()
    except RuntimeError as exc:
        print(f"twh: review failed: {exc}")
        return 1

    if not pending:
        print("No pending tasks found.")
        return 0

    ready = ready_tasks(pending)
    missing = collect_missing_metadata(pending, ready)

    if not missing:
        print("All pending tasks have imp/urg/opt/mode set.")
    else:
        print("\nTasks missing metadata (ready tasks first):")
        shown = 0
        for item in missing:
            if shown >= limit:
                break
            task = item.task
            task_id = str(task.id) if task.id is not None else task.uuid[:8]
            missing_fields_str = ",".join(item.missing)
            project = task.project or "-"
            print(f"[{task_id}] {task.description}  project={project}  missing={missing_fields_str}")
            shown += 1

    updated = False
    if wizard and missing:
        for item in missing:
            if not item.is_ready:
                continue
            task = item.task
            task_id = str(task.id) if task.id is not None else task.uuid[:8]
            print("\n---")
            print(f"Task [{task_id}] {task.description}")
            updates = interactive_fill_missing(task, input_func=input_func)
            if updates:
                apply_updates(task.uuid, updates)
                updated = True
                print("Updated.")
            if wizard_once:
                break

    if updated:
        try:
            pending = load_pending_tasks()
        except RuntimeError as exc:
            print(f"twh: review failed: {exc}")
            return 1

    report = build_review_report(
        pending,
        current_mode=mode,
        strict_mode=strict_mode,
        include_dominated=include_dominated,
        top=top,
    )

    if not report.candidates:
        print("No ready tasks found (or all were filtered).")
        return 0

    print("\nTop candidates:")
    for candidate in report.candidates:
        print(format_task_rationale(candidate.task, candidate.components))

    best = report.candidates[0].task
    best_id = str(best.id) if best.id is not None else best.uuid[:8]
    print(f"\nNext move suggestion: {best_id} - {best.description}")
    return 0

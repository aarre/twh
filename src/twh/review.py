#!/usr/bin/env python3
"""
Assess Taskwarrior move metadata and suggest next actions.
"""

from __future__ import annotations

import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone, tzinfo
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from .taskwarrior import (
    apply_case_insensitive_overrides,
    describe_missing_udas,
    filter_modified_zero_lines,
    missing_udas,
    parse_dependencies,
    read_tasks_from_json,
)
from . import modes as modes_module
from . import calibration

if TYPE_CHECKING:
    from .dominance import DominanceState


IMMINENT_SCHEDULE_WINDOW = timedelta(hours=24)
PRECEDENCE_HOURS_CAP = 40.0
PRECEDENCE_PERCENTILE = 95.0
PRECEDENCE_WEIGHTS = {
    "enablement": 0.35,
    "blocker": 0.30,
    "difficulty": 0.20,
    "dependency": 0.15,
}
PRECEDENCE_MODE_MULTIPLIERS = {
    "strategic": 1.2,
    "operational": 1.0,
    "explore": 0.9,
}
IN_PROGRESS_LABEL = "[IN PROGRESS]"
IN_PROGRESS_COLOR = "\x1b[42m"
ANSI_RESET = "\x1b[0m"


def load_precedence_weights() -> Dict[str, float]:
    """
    Load precedence weights from calibration data when available.

    Returns
    -------
    Dict[str, float]
        Precedence weights.
    """
    return calibration.load_precedence_weights(PRECEDENCE_WEIGHTS)


@dataclass(frozen=True)
class ReviewTask:
    """
    Normalized Taskwarrior move data for review workflows.

    Attributes
    ----------
    uuid : str
        Move UUID.
    id : Optional[int]
        Move ID, if available.
    description : str
        Move description.
    project : Optional[str]
        Project name.
    depends : List[str]
        UUIDs of moves this move depends on.
    imp : Optional[int]
        Importance horizon in days.
    urg : Optional[int]
        Urgency horizon in days.
    opt : Optional[int]
        Option value (0-10).
    opt_human : Optional[float]
        Manual option value rating.
    opt_auto : Optional[float]
        Auto-calculated option value estimate.
    diff : Optional[float]
        Estimated difficulty in hours.
    criticality : Optional[float]
        Time-criticality rating (0-10) derived from ranking.
    enablement : Optional[float]
        Enablement rating (0-10) for unlocking downstream moves.
    blocker_relief : Optional[float]
        Blocker relief rating (0-10) for removing blockers.
    estimate_hours : Optional[float]
        Estimated effort in hours for precedence scoring.
    mode : Optional[str]
        Mode associated with the move.
    scheduled : Optional[datetime]
        Scheduled datetime for the move.
    wait : Optional[datetime]
        Wait-until datetime for the move.
    start : Optional[datetime]
        Start datetime for in-progress moves.
    dominates : List[str]
        UUIDs explicitly dominated by this move.
    dominated_by : List[str]
        UUIDs that dominate this move.
    annotations : List[str]
        Annotation descriptions attached to the move.
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
    opt_human: Optional[float] = None
    opt_auto: Optional[float] = None
    diff: Optional[float] = None
    enablement: Optional[float] = None
    blocker_relief: Optional[float] = None
    estimate_hours: Optional[float] = None
    mode: Optional[str] = None
    scheduled: Optional[datetime] = None
    wait: Optional[datetime] = None
    start: Optional[datetime] = None
    dominates: List[str] = field(default_factory=list)
    dominated_by: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    criticality: Optional[float] = None

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
            Parsed move instance.

        Examples
        --------
        >>> move = ReviewTask.from_json({"uuid": "u1", "description": "Test", "imp": "3"})
        >>> move.imp
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

        def parse_float(key: str) -> Optional[float]:
            value = payload.get(key)
            if value is None or value == "":
                return None
            try:
                return float(value)
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
            opt_human=parse_float("opt_human"),
            opt_auto=parse_float("opt_auto"),
            diff=parse_float("diff"),
            criticality=parse_float("criticality"),
            enablement=parse_float("enablement"),
            blocker_relief=parse_float("blocker_relief"),
            estimate_hours=parse_float("estimate_hours"),
            mode=normalize_mode(payload.get("mode")),
            scheduled=parse_task_timestamp(payload.get("scheduled")),
            wait=parse_task_timestamp(payload.get("wait")),
            start=parse_task_timestamp(payload.get("start")),
            dominates=parse_uuid_list(payload.get("dominates")),
            dominated_by=parse_uuid_list(payload.get("dominated_by")),
            annotations=parse_annotations(payload.get("annotations")),
            raw=payload,
        )


@dataclass(frozen=True)
class MissingMetadata:
    """
    Record missing metadata for a review move.

    Attributes
    ----------
    task : ReviewTask
        Move with missing metadata.
    missing : Tuple[str, ...]
        Missing metadata field names.
    is_ready : bool
        True when the move is ready to act on.
    """

    task: ReviewTask
    missing: Tuple[str, ...]
    is_ready: bool


@dataclass(frozen=True)
class ScoredTask:
    """
    Move paired with its review score.

    Attributes
    ----------
    task : ReviewTask
        Move being scored.
    score : float
        Composite score for the move.
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
        Moves missing metadata.
    candidates : List[ScoredTask]
        Scored ready moves.
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
    task_args = apply_case_insensitive_overrides(list(args))
    return subprocess.run(["task", *task_args], **kwargs)


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


def get_local_timezone() -> tzinfo:
    """
    Return the local timezone for formatting timestamps.

    Returns
    -------
    timezone
        Local timezone, defaulting to UTC if unavailable.
    """
    tzinfo = datetime.now().astimezone().tzinfo
    return tzinfo if tzinfo is not None else timezone.utc


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
    text = str(value).strip()
    if not text:
        return None
    formats = [
        ("%Y%m%dT%H%M%SZ", True),
        ("%Y%m%dT%H%M%S", False),
        ("%Y-%m-%dT%H:%M:%SZ", True),
        ("%Y-%m-%dT%H:%M:%S", False),
        ("%Y-%m-%d %H:%M:%S", False),
    ]
    for fmt, is_utc in formats:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if is_utc:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def format_local_datetime(dt: datetime) -> str:
    """
    Format a datetime in local-friendly 12-hour notation.

    Parameters
    ----------
    dt : datetime
        Datetime to format.

    Returns
    -------
    str
        Formatted timestamp string.
    """
    hour = dt.strftime("%I").lstrip("0") or "12"
    return f"{dt:%Y-%m-%d} {hour}:{dt:%M:%S} {dt:%p}"


def format_task_timestamp(value: Optional[str]) -> str:
    """
    Format a Taskwarrior timestamp for review output.

    Parameters
    ----------
    value : Optional[str]
        Taskwarrior timestamp value.

    Returns
    -------
    str
        Formatted timestamp string, or the original value when parsing fails.
    """
    dt = parse_task_timestamp(value)
    if not dt:
        return str(value).strip() if value is not None else ""
    if dt.tzinfo is not None:
        dt = dt.astimezone(get_local_timezone())
    return format_local_datetime(dt)


def normalize_review_datetime(value: datetime) -> datetime:
    """
    Normalize datetimes for schedule comparisons.

    Parameters
    ----------
    value : datetime
        Datetime to normalize.

    Returns
    -------
    datetime
        Datetime localized to the system timezone.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=get_local_timezone())
    return value.astimezone(get_local_timezone())


def filter_future_start_tasks(
    tasks: Iterable[ReviewTask],
    now: datetime,
) -> List[ReviewTask]:
    """
    Exclude moves whose start time is in the future.

    Parameters
    ----------
    tasks : Iterable[ReviewTask]
        Moves to filter.
    now : datetime
        Reference datetime for comparison.

    Returns
    -------
    List[ReviewTask]
        Moves whose start time has arrived.

    Examples
    --------
    >>> now = datetime(2026, 2, 2, 18, 45)
    >>> future = now + timedelta(hours=1)
    >>> tasks = [ReviewTask("u1", 1, "Move", None, [], 1, 1, 1, start=future)]
    >>> filter_future_start_tasks(tasks, now)
    []
    """
    now_value = normalize_review_datetime(now)
    ready: List[ReviewTask] = []
    for task in tasks:
        if task.start:
            start_value = normalize_review_datetime(task.start)
            if start_value > now_value:
                continue
        ready.append(task)
    return ready


@dataclass(frozen=True)
class PrecedenceGraphStats:
    """
    Dependency stats for precedence scoring.

    Attributes
    ----------
    out_degree : Dict[str, int]
        Count of dependent moves for each UUID.
    critical_path_len : Dict[str, int]
        Longest downstream path length for each UUID.
    max_out_degree : int
        Normalization cap for out-degree.
    max_critical_path_len : int
        Normalization cap for critical path length.
    """

    out_degree: Dict[str, int]
    critical_path_len: Dict[str, int]
    max_out_degree: int
    max_critical_path_len: int


def _resolve_dependency_uuid(
    value: str,
    uuid_set: set[str],
    id_map: Dict[str, str],
) -> Optional[str]:
    if value in uuid_set:
        return value
    return id_map.get(str(value))


def _percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = math.ceil((percentile / 100.0) * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


def build_precedence_graph_stats(
    tasks: Sequence[ReviewTask],
) -> PrecedenceGraphStats:
    """
    Build dependency stats for precedence scoring.

    Parameters
    ----------
    tasks : Sequence[ReviewTask]
        Moves in scope.

    Returns
    -------
    PrecedenceGraphStats
        Graph statistics for precedence calculations.
    """
    uuid_set = {task.uuid for task in tasks}
    id_map = {
        str(task.id): task.uuid for task in tasks if task.id is not None
    }
    depended_by: Dict[str, Set[str]] = {uuid: set() for uuid in uuid_set}

    for task in tasks:
        for dep in task.depends:
            resolved = _resolve_dependency_uuid(str(dep), uuid_set, id_map)
            if resolved:
                depended_by.setdefault(resolved, set()).add(task.uuid)

    out_degree = {uuid: len(depended_by.get(uuid, set())) for uuid in uuid_set}

    cache: Dict[str, int] = {}
    visiting: Set[str] = set()

    def critical_path(node: str) -> int:
        if node in cache:
            return cache[node]
        if node in visiting:
            return 0
        visiting.add(node)
        max_len = 0
        for child in depended_by.get(node, set()):
            max_len = max(max_len, 1 + critical_path(child))
        visiting.discard(node)
        cache[node] = max_len
        return max_len

    critical_path_len = {uuid: critical_path(uuid) for uuid in uuid_set}
    max_out_degree = _percentile(list(out_degree.values()), PRECEDENCE_PERCENTILE)
    max_critical_path_len = _percentile(
        list(critical_path_len.values()),
        PRECEDENCE_PERCENTILE,
    )
    return PrecedenceGraphStats(
        out_degree=out_degree,
        critical_path_len=critical_path_len,
        max_out_degree=max_out_degree,
        max_critical_path_len=max_critical_path_len,
    )


def effective_schedule_time(task: ReviewTask) -> Optional[datetime]:
    """
    Return the effective schedule time for a move.

    When both scheduled and wait-until values are set, the later timestamp is
    treated as the effective schedule time.

    Parameters
    ----------
    task : ReviewTask
        Move to inspect.

    Returns
    -------
    Optional[datetime]
        Effective schedule time, or None when unset.
    """
    times = []
    if task.scheduled:
        times.append(normalize_review_datetime(task.scheduled))
    if task.wait:
        times.append(normalize_review_datetime(task.wait))
    if not times:
        return None
    return max(times)


def schedule_sort_key(
    task: ReviewTask,
    now: datetime,
    imminent_window: timedelta = IMMINENT_SCHEDULE_WINDOW,
) -> Tuple[int, datetime]:
    """
    Build the schedule-aware sort key for a move.

    Parameters
    ----------
    task : ReviewTask
        Move to evaluate.
    now : datetime
        Reference time for comparing schedules.
    imminent_window : timedelta, optional
        Imminent window used to prioritize scheduled moves.

    Returns
    -------
    Tuple[int, datetime]
        Tuple of (group, schedule time) for sorting.

    Examples
    --------
    >>> task = ReviewTask("u1", 1, "Move", None, [], 1, 1, 1)
    >>> schedule_sort_key(task, datetime(2024, 1, 1, 9, 0, 0))[0]
    1
    >>> scheduled = ReviewTask(
    ...     "u2",
    ...     2,
    ...     "Scheduled",
    ...     None,
    ...     [],
    ...     1,
    ...     1,
    ...     1,
    ...     scheduled=datetime(2024, 1, 1, 10, 0, 0),
    ... )
    >>> schedule_sort_key(scheduled, datetime(2024, 1, 1, 9, 0, 0))[0]
    0
    """
    now_local = normalize_review_datetime(now)
    scheduled_time = effective_schedule_time(task)
    if scheduled_time is None:
        return 1, now_local
    cutoff = now_local + imminent_window
    if scheduled_time <= cutoff:
        return 0, scheduled_time
    return 2, scheduled_time


def parse_annotations(value: Any) -> List[str]:
    """
    Parse annotation payloads into human-readable strings.

    Parameters
    ----------
    value : Any
        Annotation payload from Taskwarrior export.

    Returns
    -------
    List[str]
        Annotation descriptions (optionally prefixed with entry timestamps).

    Examples
    --------
    >>> parse_annotations([{"description": "Note"}])
    ['Note']
    >>> parse_annotations([])
    []
    """
    if not value:
        return []
    annotations: List[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                description = str(item.get("description", "")).strip()
                entry = str(item.get("entry", "")).strip()
                if description and entry:
                    formatted_entry = format_task_timestamp(entry)
                    if formatted_entry:
                        annotations.append(f"{formatted_entry}: {description}")
                    else:
                        annotations.append(description)
                elif description:
                    annotations.append(description)
                else:
                    continue
            else:
                raw = str(item).strip()
                if raw:
                    annotations.append(raw)
        return annotations
    raw = str(value).strip()
    return [raw] if raw else []


def ready_tasks(pending: Iterable[ReviewTask]) -> List[ReviewTask]:
    """
    Return moves that are ready (no pending dependencies).

    Parameters
    ----------
    pending : Iterable[ReviewTask]
        Pending moves to evaluate.

    Returns
    -------
    List[ReviewTask]
        Ready moves.
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
    Return the missing metadata fields for a move.

    Parameters
    ----------
    task : ReviewTask
        Move to inspect.

    Returns
    -------
    Tuple[str, ...]
        Missing fields in canonical order.

    Examples
    --------
    >>> move = ReviewTask("u1", 1, "x", None, [], None, 2, None)
    >>> missing_fields(move)
    ('imp', 'opt_human', 'diff', 'mode', 'criticality')
    """
    missing: List[str] = []
    if task.imp is None:
        missing.append("imp")
    if task.urg is None:
        missing.append("urg")
    if manual_option_value(task) is None:
        missing.append("opt_human")
    if task.diff is None:
        missing.append("diff")
    if not task.mode:
        missing.append("mode")
    if task.criticality is None:
        missing.append("criticality")
    return tuple(missing)


def collect_missing_metadata(
    pending: Iterable[ReviewTask],
    ready: Iterable[ReviewTask],
    dominance_missing: Optional[set[str]] = None,
) -> List[MissingMetadata]:
    """
    Collect moves missing metadata, ordering ready items first.

    Parameters
    ----------
    pending : Iterable[ReviewTask]
        Pending moves.
    ready : Iterable[ReviewTask]
        Ready moves.
    dominance_missing : Optional[set[str]]
        UUIDs missing dominance ordering.

    Returns
    -------
    List[MissingMetadata]
        Missing metadata records.
    """
    ready_uuids = {task.uuid for task in ready}
    dominance_missing = dominance_missing or set()
    items: List[MissingMetadata] = []
    for task in pending:
        missing = list(missing_fields(task))
        if task.uuid in dominance_missing:
            missing.append("dominance")
        missing = tuple(missing)
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


def colorize_in_progress(label: str) -> str:
    """
    Colorize the in-progress marker for terminal output.

    Parameters
    ----------
    label : str
        Marker label to color.

    Returns
    -------
    str
        Colored marker string.

    Examples
    --------
    >>> colorize_in_progress(IN_PROGRESS_LABEL)
    '\\x1b[42m[IN PROGRESS]\\x1b[0m'
    """
    return f"{IN_PROGRESS_COLOR}{label}{ANSI_RESET}"


def started_marker(task: ReviewTask) -> str:
    """
    Return an in-progress marker for started moves.

    Parameters
    ----------
    task : ReviewTask
        Move to inspect.

    Returns
    -------
    str
        Marker string when the move is in progress.

    Examples
    --------
    >>> task = ReviewTask(
    ...     "u1",
    ...     1,
    ...     "Move",
    ...     None,
    ...     [],
    ...     1,
    ...     1,
    ...     1,
    ...     start=datetime(2024, 1, 1, 9, 0, 0),
    ... )
    >>> started_marker(task)
    ' \\x1b[42m[IN PROGRESS]\\x1b[0m'
    >>> started_marker(ReviewTask("u2", 2, "Move", None, [], 1, 1, 1))
    ''
    """
    if not task.start:
        return ""
    return f" {colorize_in_progress(IN_PROGRESS_LABEL)}"


def format_missing_metadata_line(item: MissingMetadata) -> str:
    """
    Format a missing metadata list line for ondeck.

    Parameters
    ----------
    item : MissingMetadata
        Missing metadata entry.

    Returns
    -------
    str
        Formatted line for display.

    Examples
    --------
    >>> task = ReviewTask(
    ...     "u1",
    ...     1,
    ...     "Move",
    ...     "work",
    ...     [],
    ...     1,
    ...     1,
    ...     1,
    ...     start=datetime(2024, 1, 1, 9, 0, 0),
    ... )
    >>> item = MissingMetadata(task=task, missing=("imp", "mode"), is_ready=True)
    >>> format_missing_metadata_line(item)
    '[1] \\x1b[42m[IN PROGRESS]\\x1b[0m Move  project=work  missing=imp,mode'
    """
    move = item.task
    move_id = str(move.id) if move.id is not None else move.uuid[:8]
    missing_fields_str = ",".join(item.missing)
    project = move.project or "-"
    marker = started_marker(move)
    return (
        f"[{move_id}]{marker} {move.description}  "
        f"project={project}  missing={missing_fields_str}"
    )


def _build_dominance_context(
    pending: Sequence[ReviewTask],
) -> Tuple["DominanceState", List[List[ReviewTask]], set[str]]:
    """
    Build dominance state, tiers, and missing UUIDs for the scope.

    Parameters
    ----------
    pending : Sequence[ReviewTask]
        Pending moves in scope.

    Returns
    -------
    Tuple[DominanceState, List[List[ReviewTask]], set[str]]
        Dominance state, ordered tiers, and missing UUIDs.
    """
    from . import dominance as dominance_module

    state = dominance_module.build_dominance_state(pending)
    missing = dominance_module.dominance_missing_uuids(pending, state)
    tiers = dominance_module.build_tiers_from_state(pending, state)
    return state, tiers, missing


def _build_dominance_state_and_missing(
    pending: Sequence[ReviewTask],
) -> Tuple["DominanceState", set[str]]:
    """
    Build dominance state and missing UUIDs without tiers.

    Parameters
    ----------
    pending : Sequence[ReviewTask]
        Pending moves in scope.

    Returns
    -------
    Tuple[DominanceState, set[str]]
        Dominance state and missing UUIDs.
    """
    from . import dominance as dominance_module

    state = dominance_module.build_dominance_state(pending)
    missing = dominance_module.dominance_missing_uuids(pending, state)
    return state, missing


def dominated_set(tasks: Iterable[ReviewTask]) -> set[str]:
    """
    Return UUIDs explicitly dominated by other moves.

    Parameters
    ----------
    tasks : Iterable[ReviewTask]
        Moves to scan for dominance.

    Returns
    -------
    set[str]
        UUIDs that are dominated by other moves.
    """
    dominated: set[str] = set()
    for task in tasks:
        dominated.update(task.dominates)
    return dominated


def mode_multiplier(current: Optional[str], required: Optional[str]) -> float:
    """
    Compute the multiplier based on move mode alignment.

    Parameters
    ----------
    current : Optional[str]
        Current mode.
    required : Optional[str]
        Move-required mode.

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


def normalize_score(value: float, low: float, high: float) -> float:
    """
    Normalize a value to the 0-1 range.

    Parameters
    ----------
    value : float
        Value to normalize.
    low : float
        Minimum expected value.
    high : float
        Maximum expected value.

    Returns
    -------
    float
        Normalized score between 0 and 1.
    """
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def hours_score(hours: Optional[float], cap: float = PRECEDENCE_HOURS_CAP) -> float:
    """
    Score estimated hours, favoring shorter durations.

    Parameters
    ----------
    hours : Optional[float]
        Estimated hours.
    cap : float, optional
        Upper bound for normalization (default: PRECEDENCE_HOURS_CAP).

    Returns
    -------
    float
        Score between 0 and 1, where lower hours yield higher scores.
    """
    return 1.0 - normalize_score(float(hours or 0.0), 0.0, cap)


def precedence_mode_multiplier(mode: Optional[str]) -> float:
    """
    Return the precedence mode multiplier for a move.

    Parameters
    ----------
    mode : Optional[str]
        Move mode.

    Returns
    -------
    float
        Mode multiplier for precedence scoring.
    """
    key = (mode or "").strip().lower()
    return PRECEDENCE_MODE_MULTIPLIERS.get(key, 1.0)


def precedence_score(
    task: ReviewTask,
    stats: PrecedenceGraphStats,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute precedence score based on enablement, blockers, and graph impact.

    Parameters
    ----------
    task : ReviewTask
        Move to score.
    stats : PrecedenceGraphStats
        Dependency graph statistics.

    Returns
    -------
    float
        Precedence score for the move.
    """
    enable = normalize_score(task.enablement or 0.0, 0.0, 10.0)
    blocker = normalize_score(task.blocker_relief or 0.0, 0.0, 10.0)
    estimate = task.estimate_hours
    if estimate is None:
        estimate = task.diff
    difficulty = hours_score(estimate)
    out_degree = normalize_score(
        float(stats.out_degree.get(task.uuid, 0)),
        0.0,
        float(stats.max_out_degree),
    )
    critical_len = normalize_score(
        float(stats.critical_path_len.get(task.uuid, 0)),
        0.0,
        float(stats.max_critical_path_len),
    )
    dep = out_degree * (1.0 + critical_len)
    if weights is None:
        weights = load_precedence_weights()
    base = (
        weights["enablement"] * enable
        + weights["blocker"] * blocker
        + weights["difficulty"] * difficulty
    )
    return base * (1.0 + weights["dependency"] * dep) * precedence_mode_multiplier(
        task.mode
    )


def manual_option_value(task: ReviewTask) -> Optional[float]:
    """
    Return the manual option value when provided.

    Parameters
    ----------
    task : ReviewTask
        Move to inspect.

    Returns
    -------
    Optional[float]
        Manual option value from opt_human or legacy opt.
    """
    if task.opt_human is not None:
        return float(task.opt_human)
    if task.opt is not None:
        return float(task.opt)
    return None


def effective_option_value(task: ReviewTask) -> float:
    """
    Return the option value to use for scoring.

    Parameters
    ----------
    task : ReviewTask
        Move to score.

    Returns
    -------
    float
        Manual option value when present, otherwise opt_auto, else 0.0.

    Examples
    --------
    >>> task = ReviewTask("u1", 1, "Move", None, [], 1, 1, None, opt_auto=7.5)
    >>> effective_option_value(task)
    7.5
    """
    manual = manual_option_value(task)
    if manual is not None:
        return manual
    if task.opt_auto is not None:
        return float(task.opt_auto)
    return 0.0


def score_task(
    task: ReviewTask,
    current_mode: Optional[str],
    precedence: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Score a move based on importance, urgency, option value, difficulty, criticality,
    and precedence.

    Parameters
    ----------
    task : ReviewTask
        Move to score.
    current_mode : Optional[str]
        Current mode for scoring.
    precedence : float, optional
        Precedence score multiplier component (default: 0.0).

    Returns
    -------
    Tuple[float, Dict[str, float]]
        Total score and component breakdown.
    """
    missing_penalty = 1.0
    if task.imp is None or task.urg is None:
        missing_penalty *= 0.92
    if manual_option_value(task) is None:
        missing_penalty *= 0.96
    if task.diff is None:
        missing_penalty *= 0.95
    if task.criticality is None:
        missing_penalty *= 0.95

    imp_days = max(0, task.imp or 0)
    urg_days = max(0, task.urg or 0)
    opt_value = max(0.0, min(10.0, effective_option_value(task)))
    diff_hours = max(0.0, float(task.diff) if task.diff is not None else 0.0)
    crit_value = max(
        0.0,
        min(10.0, float(task.criticality) if task.criticality is not None else 0.0),
    )

    imp_score = math.log1p(imp_days)
    urg_hours = max(urg_days * 24.0, 1.0)
    time_pressure = diff_hours / urg_hours
    urg_score = (1.0 / (urg_days + 1.0)) * (1.0 + time_pressure)
    opt_score = opt_value / 10.0
    diff_ease = 1.0 / (diff_hours + 1.0)
    crit_score = crit_value / 10.0

    mode_mult = mode_multiplier(current_mode, task.mode)
    base = (
        (1.4 * urg_score)
        + (1.0 * imp_score)
        + (0.9 * opt_score)
        + (0.25 * diff_ease)
        + (0.3 * crit_score)
    )
    o_mult = 1.0 + max(0.0, precedence)
    total = base * mode_mult * missing_penalty * o_mult

    components = {
        "imp_score": imp_score,
        "urg_score": urg_score,
        "opt_score": opt_score,
        "diff_score": diff_ease,
        "crit_score": crit_score,
        "o_score": precedence,
        "o_mult": o_mult,
        "time_pressure": time_pressure,
        "mode_mult": mode_mult,
        "missing_mult": missing_penalty,
        "total": total,
    }
    return total, components


def format_task_rationale(task: ReviewTask, components: Dict[str, float]) -> str:
    """
    Format a move rationale block for review output.

    Parameters
    ----------
    task : ReviewTask
        Move to describe.
    components : Dict[str, float]
        Score component data.

    Returns
    -------
    str
        Multi-line rationale string.
    """
    imp = task.imp if task.imp is not None else "?"
    urg = task.urg if task.urg is not None else "?"
    manual = manual_option_value(task)
    opt = "?" if manual is None else f"{manual:g}"
    diff = task.diff if task.diff is not None else "?"
    crit = task.criticality if task.criticality is not None else "?"
    mode = task.mode or "-"
    proj = task.project or "-"
    move_id = str(task.id) if task.id is not None else task.uuid[:8]
    marker = started_marker(task)
    return (
        f"[{move_id}]{marker} {task.description}\n"
        f"  project={proj} mode={mode} imp={imp}d urg={urg}d opt={opt} "
        f"diff={diff}h crit={crit}\n"
        "  "
        f"score={components['total']:.3f} "
        f"(urg={components['urg_score']:.3f}, "
        f"imp={components['imp_score']:.3f}, "
        f"opt={components['opt_score']:.2f}, "
        f"diff={components['diff_score']:.2f}, "
        f"crit={components['crit_score']:.2f}, "
        f"o={components['o_score']:.2f}, "
        f"mode*{components['mode_mult']:.2f}, "
        f"spec*{components['missing_mult']:.2f})"
    )


def format_annotation_lines(task: ReviewTask) -> List[str]:
    """
    Format annotation lines for a move.

    Parameters
    ----------
    task : ReviewTask
        Move to format.

    Returns
    -------
    List[str]
        Annotation lines prefixed for display.
    """
    return [f"  Annotation: {note}" for note in task.annotations]


def format_dominance_lines(
    task: ReviewTask,
    state: "DominanceState",
    limit: int = 3,
) -> List[str]:
    """
    Format first-order dominance lines for a move.

    Parameters
    ----------
    task : ReviewTask
        Move to format.
    state : DominanceState
        Dominance graph state.
    limit : int, optional
        Maximum dominance edges to show.

    Returns
    -------
    List[str]
        Dominance summary lines.
    """
    dominated = sorted(
        state.dominates.get(task.uuid, set()),
        key=lambda uuid: (
            state.tasks[uuid].id if state.tasks[uuid].id is not None else 10**9,
            uuid,
        ),
    )
    lines: List[str] = []
    for uuid in dominated[:limit]:
        target = state.tasks.get(uuid)
        if not target:
            continue
        target_id = str(target.id) if target.id is not None else target.uuid[:8]
        description = target.description.strip()
        if description:
            lines.append(f"  Dominates move ID {target_id}: {description}")
        else:
            lines.append(f"  Dominates move ID {target_id}")
    if len(dominated) > limit:
        remaining = len(dominated) - limit
        lines.append(f"  Dominates {remaining} more move(s)")
    return lines


def _build_ondeck_display_payload(
    candidate: ScoredTask,
    rank: int,
) -> Dict[str, Any]:
    """
    Build a Taskwarrior-style payload for ondeck display rows.

    Parameters
    ----------
    candidate : ScoredTask
        Scored move candidate.

    Returns
    -------
    Dict[str, Any]
        Payload with the ondeck score stored under ``urgency`` and
        description updated to include an in-progress marker when started.
    """
    task = candidate.task
    payload = dict(task.raw)
    payload["uuid"] = task.uuid
    payload["id"] = task.id
    description = task.description.strip()
    marker = IN_PROGRESS_LABEL if task.start else ""
    if marker:
        description = f"{marker} {description}" if description else marker
    payload["description"] = description
    payload["rank"] = rank
    payload["score"] = candidate.score
    if "project" not in payload and task.project is not None:
        payload["project"] = task.project
    if "depends" not in payload and task.depends:
        payload["depends"] = task.depends
    return payload


def _prepare_ondeck_columns(
    columns: List[str],
    labels: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    """
    Normalize ondeck columns so ID is first and urgency is labeled as score.

    Parameters
    ----------
    columns : List[str]
        Column names to normalize.
    labels : Optional[List[str]]
        Column labels to normalize.

    Returns
    -------
    Tuple[List[str], List[str]]
        Normalized columns and labels.
    """
    normalized = [col.lower() for col in columns]
    if not labels or len(labels) != len(normalized):
        from . import label_for_column

        labels = [label_for_column(col, {}) for col in normalized]
    label_map = {col: label for col, label in zip(normalized, labels)}
    normalized = ["rank" if col == "urgency" else col for col in normalized]
    if "id" not in normalized:
        normalized.insert(0, "id")
    if "rank" not in normalized:
        normalized.append("rank")
    if "score" not in normalized:
        normalized.append("score")
    ordered = ["id"] + [col for col in normalized if col != "id"]
    ordered_labels: List[str] = []
    for col in ordered:
        label = label_map.get(col)
        if not label:
            from . import label_for_column

            label = label_for_column(col, {})
        if col == "rank":
            label = "Rank"
        if col == "score":
            label = "Score"
        ordered_labels.append(label)
    return ordered, ordered_labels


def format_ondeck_candidates(
    candidates: Sequence[ScoredTask],
    columns: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
) -> List[str]:
    """
    Format ondeck candidates using Taskwarrior-style columns and colors.

    The ondeck score is shown in the urgency column.

    Parameters
    ----------
    candidates : Sequence[ScoredTask]
        Scored move candidates to display.
    columns : Optional[List[str]], optional
        Column names to render (default: Taskwarrior report columns).
    labels : Optional[List[str]], optional
        Column labels to render (default: Taskwarrior report labels).

    Returns
    -------
    List[str]
        Rendered report lines, including header and separator.

    Examples
    --------
    >>> task = ReviewTask(
    ...     uuid="u1",
    ...     id=1,
    ...     description="Move A",
    ...     project=None,
    ...     depends=[],
    ...     imp=1,
    ...     urg=1,
    ...     opt=1,
    ...     raw={"uuid": "u1", "id": 1, "description": "Move A"},
    ... )
    >>> candidate = ScoredTask(task=task, score=12.345, components={})
    >>> lines = format_ondeck_candidates(
    ...     [candidate],
    ...     columns=["description", "id", "urgency"],
    ...     labels=["Description", "ID", "Urg"],
    ... )
    >>> lines[0].lstrip().startswith("ID")
    True
    >>> "Rank" in lines[0]
    True
    >>> "Score" in lines[0]
    True
    >>> lines[2].strip().startswith("1")
    True
    """
    if not candidates:
        return []
    from . import get_taskwarrior_columns_and_labels, render_task_table

    if not columns or not labels:
        columns, labels = get_taskwarrior_columns_and_labels()
    columns, labels = _prepare_ondeck_columns(columns, labels)
    uuid_to_id = {item.task.uuid: item.task.id for item in candidates}
    rows = [
        (_build_ondeck_display_payload(item, rank=index + 1), "")
        for index, item in enumerate(candidates)
    ]
    return render_task_table(rows, columns, labels, uuid_to_id)


def format_candidate_output(
    candidate: ScoredTask,
    dominance_state: "DominanceState",
    dominance_limit: int = 3,
) -> List[str]:
    """
    Format the output line for a scored move.

    Parameters
    ----------
    candidate : ScoredTask
        Scored move candidate.
    dominance_state : DominanceState
        Dominance graph state (unused for the summary output).
    dominance_limit : int, optional
        Ignored; retained for signature compatibility.

    Returns
    -------
    List[str]
        Single-line summary ready to print.
    """
    _ = dominance_state, dominance_limit
    task = candidate.task
    move_id = str(task.id) if task.id is not None else task.uuid[:8]
    marker = started_marker(task)
    description = task.description.strip()
    if description:
        return [f"[{move_id}]{marker} {description}"]
    return [f"[{move_id}]{marker}"]


def filter_candidates(
    ready: Iterable[ReviewTask],
    current_mode: Optional[str],
    strict_mode: bool,
    include_dominated: bool,
) -> List[ReviewTask]:
    """
    Filter ready moves based on mode and dominance rules.

    Parameters
    ----------
    ready : Iterable[ReviewTask]
        Ready moves.
    current_mode : Optional[str]
        Current mode context.
    strict_mode : bool
        Require mode match when True.
    include_dominated : bool
        Keep dominated moves when True.

    Returns
    -------
    List[ReviewTask]
        Filtered candidate moves.
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
    dominance_tiers: Optional[List[List[ReviewTask]]] = None,
    now: Optional[datetime] = None,
    graph_tasks: Optional[Iterable[ReviewTask]] = None,
) -> List[ScoredTask]:
    """
    Rank candidate moves by dominance tier and score.

    Parameters
    ----------
    candidates : Iterable[ReviewTask]
        Moves to score.
    current_mode : Optional[str]
        Current mode for scoring.
    top : int
        Number of top candidates to return.
    dominance_tiers : Optional[List[List[ReviewTask]]]
        Dominance tiers to apply (highest first). When omitted, tiers are built
        from the candidate set.
    now : Optional[datetime]
        Reference time for schedule-aware ordering (default: current time).
    graph_tasks : Optional[Iterable[ReviewTask]]
        Tasks to use for precedence graph stats (defaults to candidates).

    Returns
    -------
    List[ScoredTask]
        Ranked candidates.
    """
    candidate_list = list(candidates)
    graph_source = list(graph_tasks) if graph_tasks is not None else candidate_list
    stats = build_precedence_graph_stats(graph_source)
    precedence_weights = load_precedence_weights()
    scored: List[ScoredTask] = []
    for task in candidate_list:
        o_score = precedence_score(task, stats, weights=precedence_weights)
        score, components = score_task(task, current_mode, precedence=o_score)
        scored.append(ScoredTask(task=task, score=score, components=components))
    if dominance_tiers is None:
        from . import dominance as dominance_module

        state = dominance_module.build_dominance_state(candidate_list)
        dominance_tiers = dominance_module.build_tiers_from_state(candidate_list, state)
    tier_index = {
        task.uuid: idx
        for idx, tier in enumerate(dominance_tiers or [])
        for task in tier
    }
    default_tier = len(dominance_tiers or [])
    now_value = normalize_review_datetime(now or datetime.now().astimezone())
    scored.sort(
        key=lambda item: (
            tier_index.get(item.task.uuid, default_tier),
            *schedule_sort_key(item.task, now_value),
            -item.score,
            item.task.id if item.task.id is not None else 10**9,
            item.task.uuid,
        )
    )
    return scored[:max(0, top)]


def build_review_report(
    pending: Iterable[ReviewTask],
    current_mode: Optional[str],
    strict_mode: bool,
    include_dominated: bool,
    top: int,
    now: Optional[datetime] = None,
    *,
    ready: Optional[Iterable[ReviewTask]] = None,
    dominance_tiers: Optional[List[List[ReviewTask]]] = None,
    dominance_missing: Optional[set[str]] = None,
) -> ReviewReport:
    """
    Build a review report from pending moves.

    Parameters
    ----------
    pending : Iterable[ReviewTask]
        Pending moves.
    current_mode : Optional[str]
        Current mode for filtering/scoring.
    strict_mode : bool
        Require strict mode matching when True.
    include_dominated : bool
        Include dominated moves when True.
    top : int
        Number of candidates to include.
    now : Optional[datetime]
        Reference time for filtering and scheduling (default: now).
    ready : Optional[Iterable[ReviewTask]]
        Precomputed ready moves (default: compute from pending).
    dominance_tiers : Optional[List[List[ReviewTask]]]
        Precomputed dominance tiers (default: compute from pending).
    dominance_missing : Optional[set[str]]
        Precomputed missing dominance UUIDs (default: compute from pending).

    Returns
    -------
    ReviewReport
        Review report data.
    """
    now_value = normalize_review_datetime(now or datetime.now().astimezone())
    pending_list = list(pending)
    ready_list = list(ready) if ready is not None else ready_tasks(pending_list)
    if dominance_tiers is None or dominance_missing is None:
        _dominance_state, dominance_tiers, dominance_missing = _build_dominance_context(
            pending_list
        )
    missing = collect_missing_metadata(
        pending_list,
        ready_list,
        dominance_missing=dominance_missing,
    )
    candidates = filter_candidates(
        ready_list,
        current_mode=current_mode,
        strict_mode=strict_mode,
        include_dominated=include_dominated,
    )
    candidates = filter_future_start_tasks(candidates, now_value)
    ranked = rank_candidates(
        candidates,
        current_mode=current_mode,
        top=top,
        dominance_tiers=dominance_tiers,
        now=now_value,
        graph_tasks=pending_list,
    )
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
        Move to update.
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).

    Returns
    -------
    Dict[str, str]
        Fields to update.
    """
    updates: Dict[str, str] = {}
    known_modes = modes_module.load_known_modes()
    print("\nFill missing fields for this move (press Enter to skip)")
    if task.imp is None:
        value = input_func(
            "  Importance horizon - how long will you remember whether this move was done? (days): "
        ).strip()
        if value:
            updates["imp"] = value
    if task.urg is None:
        value = input_func(
            "  Urgency horizon - how long before acting loses value? (days): "
        ).strip()
        if value:
            updates["urg"] = value
    if manual_option_value(task) is None:
        value = input_func(
            "  Option value - to what extent does doing this move preserve, unlock, or multiply future moves? (0-10): "
        ).strip()
        if value:
            updates["opt_human"] = value
    if task.diff is None:
        value = input_func("  Difficulty, i.e., estimated effort (hours): ").strip()
        if value:
            updates["diff"] = value
    if not task.mode:
        prompt = modes_module.format_mode_prompt(known_modes)
        while True:
            value = modes_module.prompt_mode_value(
                prompt,
                known_modes,
                input_func=input_func,
            ).strip()
            if not value:
                break
            normalized = modes_module.normalize_mode_value(value)
            if modes_module.is_reserved_mode_value(normalized):
                print(modes_module.format_reserved_mode_error(normalized))
                continue
            updates["mode"] = normalized
            known_modes = modes_module.register_mode(normalized, modes=known_modes)
            modes_module.ensure_taskwarrior_mode_value(normalized)
            break
    return updates


def interactive_fill_missing_incremental(
    task: ReviewTask,
    input_func: Callable[[str], str] = input,
    apply_func: Callable[[str, Dict[str, str]], None] = None,
) -> bool:
    """
    Prompt for missing metadata and apply updates after each entry.

    Parameters
    ----------
    task : ReviewTask
        Move to update.
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).
    apply_func : Callable[[str, Dict[str, str]], None], optional
        Function to apply updates immediately (default: apply_updates).

    Returns
    -------
    bool
        True when any updates were applied.
    """
    if apply_func is None:
        apply_func = apply_updates
    updated = False
    known_modes = modes_module.load_known_modes()
    print("\nFill missing fields for this move (press Enter to skip)")
    if task.imp is None:
        value = input_func(
            "  Importance horizon - how long will you remember whether this move was done? (days): "
        ).strip()
        if value:
            apply_func(task.uuid, {"imp": value})
            updated = True
    if task.urg is None:
        value = input_func(
            "  Urgency horizon - how long before acting loses value? (days): "
        ).strip()
        if value:
            apply_func(task.uuid, {"urg": value})
            updated = True
    if manual_option_value(task) is None:
        value = input_func(
            "  Option value - to what extent does doing this move preserve, unlock, or multiply future moves? (0-10): "
        ).strip()
        if value:
            apply_func(task.uuid, {"opt_human": value})
            updated = True
    if task.diff is None:
        value = input_func("  Difficulty, i.e., estimated effort (hours): ").strip()
        if value:
            apply_func(task.uuid, {"diff": value})
            updated = True
    if not task.mode:
        prompt = modes_module.format_mode_prompt(known_modes)
        while True:
            value = modes_module.prompt_mode_value(
                prompt,
                known_modes,
                input_func=input_func,
            ).strip()
            if not value:
                break
            normalized = modes_module.normalize_mode_value(value)
            if modes_module.is_reserved_mode_value(normalized):
                print(modes_module.format_reserved_mode_error(normalized))
                continue
            known_modes = modes_module.register_mode(normalized, modes=known_modes)
            modes_module.ensure_taskwarrior_mode_value(normalized)
            apply_func(task.uuid, {"mode": normalized})
            updated = True
            break
    return updated


def apply_updates(
    uuid: str,
    updates: Dict[str, str],
    get_setting: Callable[[str], Optional[str]] = None,
) -> None:
    """
    Apply updates to a Taskwarrior move.

    Parameters
    ----------
    uuid : str
        Move UUID.
    updates : Dict[str, str]
        Field updates.
    get_setting : Callable[[str], Optional[str]], optional
        Getter for Taskwarrior settings (default: taskwarrior helper).

    Returns
    -------
    None
        Updates are applied to Taskwarrior.

    Raises
    ------
    RuntimeError
        If required UDAs are missing or Taskwarrior rejects updates.
    """
    if not updates:
        return
    if get_setting is None:
        from .taskwarrior import get_taskwarrior_setting

        get_setting = get_taskwarrior_setting
    missing = missing_udas(
        updates.keys(),
        get_setting=get_setting,
        allow_taskrc_fallback=False,
    )
    if missing:
        raise RuntimeError(describe_missing_udas(missing))
    if "mode" in updates:
        modes_module.validate_taskwarrior_mode_config(
            updates.get("mode"),
            get_setting=get_setting,
        )
    parts = [f"{key}:{value}" for key, value in updates.items()]
    result = run_task_command([uuid, "modify", *parts], capture_output=True)
    stdout_lines = filter_modified_zero_lines(result.stdout)
    stderr_text = (result.stderr or "").strip()
    if result.returncode != 0:
        details = stderr_text or "\n".join(stdout_lines).strip()
        raise RuntimeError(details or "Taskwarrior update failed.")
    for line in stdout_lines:
        print(line)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if "mode" in updates:
        expected = modes_module.normalize_mode_value(updates.get("mode"))
        if expected:
            verify_result = run_task_command([uuid, "export"], capture_output=True)
            if verify_result.returncode != 0:
                stderr = (verify_result.stderr or "").strip()
                raise RuntimeError(stderr or "Taskwarrior export failed after update.")
            payloads = read_tasks_from_json(verify_result.stdout or "")
            stored_value: Optional[str] = None
            for payload in payloads:
                if isinstance(payload, dict) and payload.get("uuid") == uuid:
                    stored_value = modes_module.normalize_mode_value(
                        payload.get("mode")
                    )
                    break
            if stored_value != expected:
                raise RuntimeError(
                    "Mode update did not persist. Expected "
                    f"'{expected}', got '{stored_value or ''}'. "
                    "Check uda.mode.type/values in your active Taskwarrior config."
                )


def load_pending_tasks(filters: Optional[Sequence[str]] = None) -> List[ReviewTask]:
    """
    Load pending moves from Taskwarrior export.

    Parameters
    ----------
    filters : Optional[Sequence[str]]
        Additional Taskwarrior filter tokens.

    Returns
    -------
    List[ReviewTask]
        Pending moves.

    Raises
    ------
    RuntimeError
        If Taskwarrior export fails.
    """
    filter_tokens = list(filters) if filters else []
    result = run_task_command(
        [*filter_tokens, "status:pending", "export"],
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or "Taskwarrior export failed.")
    tasks = read_tasks_from_json(result.stdout or "")
    return [
        ReviewTask.from_json(task)
        for task in tasks
        if isinstance(task, dict) and task.get("uuid")
    ]


def run_ondeck(
    *,
    mode: Optional[str],
    limit: int,
    top: int,
    strict_mode: bool,
    include_dominated: bool,
    filters: Optional[Sequence[str]] = None,
    input_func: Callable[[str], str] = input,
) -> int:
    """
    Execute the ondeck flow: fill missing metadata + recommend next moves.

    Parameters
    ----------
    mode : Optional[str]
        Current mode context.
    limit : int
        Max missing moves to list.
    top : int
        Number of top candidates to show.
    strict_mode : bool
        Require mode match when True.
    include_dominated : bool
        Include dominated moves when True.
    filters : Optional[Sequence[str]]
        Additional Taskwarrior filter tokens.
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).

    Returns
    -------
    int
        Exit code.
    """
    try:
        pending = load_pending_tasks(filters=filters)
    except RuntimeError as exc:
        print(f"twh: ondeck failed: {exc}")
        return 1

    if not pending:
        print("No pending moves found.")
        return 0

    ready = ready_tasks(pending)
    dominance_state: Optional[DominanceState] = None
    dominance_tiers: Optional[List[List[ReviewTask]]] = None
    dominance_missing: set[str] = set()
    criticality_missing: set[str] = {
        task.uuid for task in pending if task.criticality is None
    }
    missing = collect_missing_metadata(
        pending,
        ready,
        dominance_missing=set(),
    )
    needs_wizard = bool(missing)

    if not needs_wizard:
        dominance_state, dominance_missing = _build_dominance_state_and_missing(
            pending
        )
        if dominance_missing:
            missing = collect_missing_metadata(
                pending,
                ready,
                dominance_missing=dominance_missing,
            )
            needs_wizard = True

    if needs_wizard:
        print("\nMoves missing metadata or dominance ordering (ready moves first):")
        shown = 0
        for item in missing:
            if shown >= limit:
                break
            print(format_missing_metadata_line(item))
            shown += 1

    updated = False
    option_applied = False
    if needs_wizard:
        for item in missing:
            if all(
                field in {"dominance", "criticality"}
                for field in item.missing
            ):
                continue
            move = item.task
            move_id = str(move.id) if move.id is not None else move.uuid[:8]
            print("\n---")
            print(f"Move [{move_id}] {move.description}")
            try:
                was_updated = interactive_fill_missing_incremental(
                    move,
                    input_func=input_func,
                )
            except RuntimeError as exc:
                print(f"twh: ondeck failed: {exc}")
                return 1
            if was_updated:
                updated = True
                print("Updated.")

    if needs_wizard and dominance_state is None:
        dominance_state, dominance_missing = _build_dominance_state_and_missing(
            pending
        )

    if criticality_missing:
        from . import criticality as criticality_module

        try:
            criticality_module.ensure_criticality_uda()
        except RuntimeError as exc:
            print(f"twh: ondeck failed: {exc}")
            return 1

        state = criticality_module.build_criticality_state(pending)
        tiers = criticality_module.sort_into_tiers(
            pending,
            state,
            chooser=criticality_module.make_progress_chooser(
                pending,
                state,
                input_func=input_func,
            ),
        )
        updates = criticality_module.build_criticality_updates(tiers)
        try:
            criticality_module.apply_criticality_updates(updates)
        except RuntimeError as exc:
            print(f"twh: ondeck failed: {exc}")
            return 1
        updated = True
        print(f"Criticality updated for {len(pending)} moves.")

    if needs_wizard and dominance_missing:
        from . import dominance as dominance_module

        try:
            dominance_module.ensure_dominance_udas()
        except RuntimeError as exc:
            print(f"twh: ondeck failed: {exc}")
            return 1

        dominance_updated = False

        def on_update(
            current_state: "DominanceState",
            uuids: Iterable[str],
        ) -> None:
            nonlocal dominance_updated
            updates = dominance_module.build_incremental_updates(current_state, uuids)
            if not updates:
                return
            dominance_module.apply_dominance_updates(
                updates,
                quiet=True,
                validate_udas=False,
            )
            dominance_updated = True

        _ = dominance_module.sort_into_tiers(
            pending,
            dominance_state,
            chooser=dominance_module.make_progress_chooser(
                pending,
                dominance_state,
                input_func=input_func,
            ),
            on_update=on_update,
        )
        if dominance_updated:
            updated = True
            print(f"Dominance updated for {len(pending)} moves.")

    if updated:
        from . import option_value as option_module

        option_exit = option_module.run_option_value(
            filters=filters,
            apply=True,
            include_rated=True,
            limit=0,
        )
        if option_exit != 0:
            return option_exit
        option_applied = True

    if updated or option_applied:
        try:
            pending = load_pending_tasks(filters=filters)
        except RuntimeError as exc:
            print(f"twh: ondeck failed: {exc}")
            return 1
        ready = ready_tasks(pending)
        dominance_state = None
        dominance_tiers = None
        dominance_missing = set()

    if dominance_tiers is None:
        from . import dominance as dominance_module

        if dominance_state is None:
            dominance_state, dominance_tiers, dominance_missing = _build_dominance_context(
                pending
            )
        else:
            dominance_tiers = dominance_module.build_tiers_from_state(
                pending,
                dominance_state,
            )

    now_value = normalize_review_datetime(datetime.now().astimezone())
    report = build_review_report(
        pending,
        current_mode=mode,
        strict_mode=strict_mode,
        include_dominated=include_dominated,
        top=top,
        now=now_value,
        ready=ready,
        dominance_tiers=dominance_tiers,
        dominance_missing=dominance_missing,
    )

    if not report.candidates:
        print("No ready moves found (or all were filtered).")
        return 0

    print("\nTop move candidates:")
    for line in format_ondeck_candidates(report.candidates):
        print(line)

    best = report.candidates[0].task
    best_id = str(best.id) if best.id is not None else best.uuid[:8]
    print(
        f"\nNext move suggestion: {best_id} - {best.description}{started_marker(best)}"
    )
    return 0

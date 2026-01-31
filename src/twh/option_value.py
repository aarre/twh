#!/usr/bin/env python3
"""
Compute and calibrate option value estimates for Taskwarrior moves.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .taskwarrior import (
    filter_modified_zero_lines,
    missing_udas,
    parse_dependencies,
    read_tasks_from_json,
)

INFO_TAGS = {"probe", "explore", "call", "prototype", "test"}
ONEWAY_TAGS = {"oneway"}


def get_local_timezone() -> tzinfo:
    """
    Return the local timezone for timestamp comparisons.

    Returns
    -------
    tzinfo
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


def normalize_option_datetime(value: datetime) -> datetime:
    """
    Normalize datetimes for option value comparisons.

    Parameters
    ----------
    value : datetime
        Datetime to normalize.

    Returns
    -------
    datetime
        Normalized datetime in the local timezone.
    """
    local = get_local_timezone()
    if value.tzinfo is None:
        return value.replace(tzinfo=local)
    return value.astimezone(local)


@dataclass(frozen=True)
class OptionTask:
    """
    Taskwarrior move data for option value scoring.

    Attributes
    ----------
    uuid : str
        Move UUID.
    id : Optional[int]
        Move ID, if available.
    description : str
        Move description.
    status : str
        Taskwarrior status.
    project : Optional[str]
        Project name.
    tags : Tuple[str, ...]
        Tags associated with the move.
    due : Optional[datetime]
        Due datetime.
    priority : Optional[str]
        Priority string.
    estimate_minutes : Optional[int]
        Estimated effort in minutes, when provided.
    door : Optional[str]
        Door classification (e.g., "oneway", "twoway").
    kind : Optional[str]
        Kind classification (e.g., "probe").
    depends : List[str]
        Dependency identifiers.
    opt_human : Optional[float]
        Manual option value rating.
    opt : Optional[float]
        Legacy manual option value (deprecated).
    opt_auto : Optional[float]
        Auto-calculated option value.
    """

    uuid: str
    id: Optional[int]
    description: str
    status: str
    project: Optional[str]
    tags: Tuple[str, ...]
    due: Optional[datetime]
    priority: Optional[str]
    estimate_minutes: Optional[int]
    door: Optional[str]
    kind: Optional[str]
    depends: List[str]
    opt_human: Optional[float]
    opt: Optional[float]
    opt_auto: Optional[float]

    @staticmethod
    def from_json(payload: Dict[str, Any]) -> "OptionTask":
        """
        Build an OptionTask from Taskwarrior export data.

        Parameters
        ----------
        payload : Dict[str, Any]
            Taskwarrior export dictionary.

        Returns
        -------
        OptionTask
            Parsed move instance.
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

        def parse_text(key: str) -> Optional[str]:
            value = payload.get(key)
            if value is None:
                return None
            text = str(value).strip()
            return text if text else None

        raw_tags = payload.get("tags") or []
        tags: Tuple[str, ...]
        if isinstance(raw_tags, list):
            tags = tuple(str(tag).strip() for tag in raw_tags if str(tag).strip())
        else:
            tags = (str(raw_tags).strip(),) if str(raw_tags).strip() else ()

        return OptionTask(
            uuid=str(payload["uuid"]),
            id=payload.get("id"),
            description=str(payload.get("description", "")).strip(),
            status=str(payload.get("status", "pending")).strip(),
            project=parse_text("project"),
            tags=tags,
            due=parse_task_timestamp(payload.get("due")),
            priority=parse_text("priority"),
            estimate_minutes=parse_int("estimate_minutes"),
            door=parse_text("door"),
            kind=parse_text("kind"),
            depends=parse_dependencies(payload.get("depends")),
            opt_human=parse_float("opt_human"),
            opt=parse_float("opt"),
            opt_auto=parse_float("opt_auto"),
        )


@dataclass(frozen=True)
class Graph:
    """
    Dependency graph representation for option value scoring.

    Attributes
    ----------
    out_edges : Dict[str, Set[str]]
        Mapping from move UUID to dependencies.
    in_edges : Dict[str, Set[str]]
        Mapping from move UUID to dependents.
    """

    out_edges: Dict[str, Set[str]]
    in_edges: Dict[str, Set[str]]


@dataclass(frozen=True)
class Weights:
    """
    Model weights for option value scoring.

    Attributes
    ----------
    w_children : float
        Weight for immediate unlocks.
    w_desc : float
        Weight for downstream reach.
    w_desc_value : float
        Weight for downstream value.
    w_diversity : float
        Weight for branching diversity.
    w_info : float
        Weight for information gain.
    w_rev : float
        Weight for reversibility penalty.
    w_cost : float
        Weight for effort penalty.
    bias : float
        Bias term.
    """

    w_children: float = 2.0
    w_desc: float = 1.0
    w_desc_value: float = 1.2
    w_diversity: float = 0.8
    w_info: float = 1.2
    w_rev: float = 1.0
    w_cost: float = 0.6
    bias: float = 1.0


def now_local() -> datetime:
    """
    Return the current local datetime for scoring.

    Returns
    -------
    datetime
        Current datetime localized to the system timezone.
    """
    return datetime.now().astimezone()


def build_graph(
    tasks: Dict[str, OptionTask],
    deps: Optional[Iterable[Tuple[str, str]]] = None,
) -> Graph:
    """
    Build dependency edges for option value scoring.

    Parameters
    ----------
    tasks : Dict[str, OptionTask]
        Move map keyed by UUID.
    deps : Optional[Iterable[Tuple[str, str]]]
        Optional explicit dependency pairs (task, dep).

    Returns
    -------
    Graph
        Dependency graph data.
    """
    out_edges: Dict[str, Set[str]] = {uuid: set() for uuid in tasks}
    in_edges: Dict[str, Set[str]] = {uuid: set() for uuid in tasks}
    id_map = {
        str(task.id): task.uuid
        for task in tasks.values()
        if task.id is not None
    }

    def resolve_dep(value: str) -> Optional[str]:
        if value in tasks:
            return value
        return id_map.get(str(value))

    if deps is None:
        for task in tasks.values():
            for dep in task.depends:
                resolved = resolve_dep(dep)
                if not resolved:
                    continue
                out_edges[task.uuid].add(resolved)
                in_edges[resolved].add(task.uuid)
        return Graph(out_edges=out_edges, in_edges=in_edges)

    for task_id, dep in deps:
        resolved_task = resolve_dep(task_id)
        resolved_dep = resolve_dep(dep)
        if not resolved_task or not resolved_dep:
            continue
        out_edges[resolved_task].add(resolved_dep)
        in_edges[resolved_dep].add(resolved_task)
    return Graph(out_edges=out_edges, in_edges=in_edges)


def descendants(start: str, in_edges: Dict[str, Set[str]], limit: int = 5000) -> Set[str]:
    """
    Return downstream moves reachable from a start node.

    Parameters
    ----------
    start : str
        Start move UUID.
    in_edges : Dict[str, Set[str]]
        Mapping from move UUID to dependent moves.
    limit : int, optional
        Maximum nodes to visit (default: 5000).

    Returns
    -------
    Set[str]
        Downstream move UUIDs.
    """
    seen: Set[str] = set()
    stack = list(in_edges.get(start, set()))
    while stack and len(seen) < limit:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(in_edges.get(node, set()))
    return seen


def due_soon_bonus(task: OptionTask, now: datetime) -> float:
    """
    Assign a bonus for due dates that are soon.

    Parameters
    ----------
    task : OptionTask
        Move to inspect.
    now : datetime
        Reference time.

    Returns
    -------
    float
        Due date bonus.
    """
    if task.due is None:
        return 0.0
    due = normalize_option_datetime(task.due)
    now_local = normalize_option_datetime(now)
    days = (due - now_local).total_seconds() / 86400.0
    if days <= 0:
        return 2.0
    if days <= 3:
        return 1.0
    if days <= 7:
        return 0.5
    return 0.0


def priority_weight(task: OptionTask) -> float:
    """
    Map Taskwarrior priority values to weights.

    Parameters
    ----------
    task : OptionTask
        Move to inspect.

    Returns
    -------
    float
        Priority weight.

    Examples
    --------
    >>> priority_weight(OptionTask("u1", None, "", "pending", None, (), None, "H", None, None, None, [], None, None, None))
    1.0
    >>> priority_weight(OptionTask("u1", None, "", "pending", None, (), None, None, None, None, None, [], None, None, None))
    0.0
    """
    if task.priority == "H":
        return 1.0
    if task.priority == "M":
        return 0.4
    if task.priority == "L":
        return 0.0
    return 0.0


def info_gain_bonus(task: OptionTask) -> float:
    """
    Return an information-gain bonus for probe-like moves.

    Parameters
    ----------
    task : OptionTask
        Move to inspect.

    Returns
    -------
    float
        Information-gain bonus.
    """
    tags = {tag.lower() for tag in task.tags}
    if task.kind and task.kind.lower() == "probe":
        return 1.5
    if INFO_TAGS & tags:
        return 1.0
    return 0.0


def reversibility_penalty(task: OptionTask) -> float:
    """
    Penalize one-way door moves.

    Parameters
    ----------
    task : OptionTask
        Move to inspect.

    Returns
    -------
    float
        Reversibility penalty.
    """
    tags = {tag.lower() for tag in task.tags}
    if task.door and task.door.lower() == "oneway":
        return 1.5
    if ONEWAY_TAGS & tags:
        return 1.5
    return 0.0


def cost_penalty(task: OptionTask) -> float:
    """
    Penalize larger estimated effort.

    Parameters
    ----------
    task : OptionTask
        Move to inspect.

    Returns
    -------
    float
        Cost penalty.
    """
    if task.estimate_minutes is None:
        return 0.0
    hours = max(0.0, task.estimate_minutes / 60.0)
    return math.log(1.0 + hours)


def entropy_of_projects(task_ids: Iterable[str], tasks: Dict[str, OptionTask]) -> float:
    """
    Compute Shannon entropy of project labels among moves.

    Parameters
    ----------
    task_ids : Iterable[str]
        Move UUIDs to consider.
    tasks : Dict[str, OptionTask]
        Move map.

    Returns
    -------
    float
        Project entropy.

    Examples
    --------
    >>> tasks = {"a": OptionTask("a", None, "", "pending", "p1", (), None, None, None, None, None, [], None, None, None),
    ...          "b": OptionTask("b", None, "", "pending", "p2", (), None, None, None, None, None, [], None, None, None)}
    >>> entropy_of_projects(["a", "b"], tasks) == math.log(2.0)
    True
    """
    labels = [
        tasks[task_id].project
        for task_id in task_ids
        if task_id in tasks and tasks[task_id].project
    ]
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = sum(counts.values())
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
    )


def feature_vector(
    tid: str,
    tasks: Dict[str, OptionTask],
    deps: Optional[Iterable[Tuple[str, str]]] = None,
    now: Optional[datetime] = None,
) -> List[float]:
    """
    Compute the option value feature vector for a move.

    Parameters
    ----------
    tid : str
        Move UUID.
    tasks : Dict[str, OptionTask]
        Move map keyed by UUID.
    deps : Optional[Iterable[Tuple[str, str]]]
        Optional dependency pairs (task, dep).
    now : Optional[datetime]
        Reference time (default: current time).

    Returns
    -------
    List[float]
        Feature vector in model order.
    """
    if now is None:
        now = now_local()
    graph = build_graph(tasks, deps=deps)
    task = tasks[tid]
    children = list(graph.in_edges.get(tid, set()))
    desc = descendants(tid, graph.in_edges)

    f1 = math.log(1.0 + len(children))
    f2 = math.log(1.0 + len(desc))

    downstream_value = 0.0
    for node in desc:
        descendant = tasks[node]
        if descendant.status != "pending":
            continue
        downstream_value += 1.0 + priority_weight(descendant) + due_soon_bonus(descendant, now)
    f3 = math.log(1.0 + downstream_value)

    f4 = entropy_of_projects(children, tasks)
    f6 = info_gain_bonus(task)

    p_rev = reversibility_penalty(task)
    p_cost = cost_penalty(task)

    return [f1, f2, f3, f4, f6, p_rev, p_cost]


def option_value_score(
    tid: str,
    tasks: Dict[str, OptionTask],
    deps: Optional[Iterable[Tuple[str, str]]] = None,
    weights: Weights = Weights(),
    now: Optional[datetime] = None,
) -> float:
    """
    Compute the option value score for a move.

    Parameters
    ----------
    tid : str
        Move UUID.
    tasks : Dict[str, OptionTask]
        Move map keyed by UUID.
    deps : Optional[Iterable[Tuple[str, str]]]
        Optional dependency pairs (task, dep).
    weights : Weights, optional
        Model weights.
    now : Optional[datetime]
        Reference time (default: current time).

    Returns
    -------
    float
        Option value score, clamped to 0-10.
    """
    task = tasks[tid]
    if task.status != "pending":
        return 0.0
    features = feature_vector(tid, tasks, deps=deps, now=now)
    raw = (
        weights.bias
        + weights.w_children * features[0]
        + weights.w_desc * features[1]
        + weights.w_desc_value * features[2]
        + weights.w_diversity * features[3]
        + weights.w_info * features[4]
        - weights.w_rev * features[5]
        - weights.w_cost * features[6]
    )
    return max(0.0, min(10.0, raw))


def format_option_value(value: float) -> str:
    """
    Format an option value for display and storage.

    Parameters
    ----------
    value : float
        Option value.

    Returns
    -------
    str
        Formatted option value.

    Examples
    --------
    >>> format_option_value(6.25)
    '6.2'
    """
    return f"{value:.1f}"


def manual_option_value(task: OptionTask) -> Optional[float]:
    """
    Return the manual option value for a move when available.

    Parameters
    ----------
    task : OptionTask
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


def _solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """
    Solve a linear system using Gaussian elimination.

    Parameters
    ----------
    matrix : List[List[float]]
        Coefficient matrix.
    vector : List[float]
        Right-hand side vector.

    Returns
    -------
    List[float]
        Solution vector.
    """
    size = len(vector)
    aug = [row[:] + [vector[idx]] for idx, row in enumerate(matrix)]

    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix in ridge regression.")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for k in range(col, size + 1):
            aug[col][k] /= pivot_val
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            for k in range(col, size + 1):
                aug[row][k] -= factor * aug[col][k]

    return [aug[row][size] for row in range(size)]


def fit_weights_ridge(
    training: List[Tuple[str, float]],
    tasks: Dict[str, OptionTask],
    deps: Optional[Iterable[Tuple[str, str]]] = None,
    lam: float = 1.0,
) -> Weights:
    """
    Fit option value weights using ridge regression.

    Parameters
    ----------
    training : List[Tuple[str, float]]
        Training pairs of (move UUID, manual option value).
    tasks : Dict[str, OptionTask]
        Move map keyed by UUID.
    deps : Optional[Iterable[Tuple[str, str]]]
        Optional dependency pairs (task, dep).
    lam : float, optional
        Ridge regularization strength (default: 1.0).

    Returns
    -------
    Weights
        Fitted weights.
    """
    if not training:
        return Weights()
    now = now_local()
    features: List[List[float]] = []
    targets: List[float] = []
    for tid, rating in training:
        if tid not in tasks:
            continue
        vector = feature_vector(tid, tasks, deps=deps, now=now)
        features.append([1.0] + vector)
        targets.append(float(rating))
    if not features:
        return Weights()

    dim = len(features[0])
    matrix = [[0.0 for _ in range(dim)] for _ in range(dim)]
    vector = [0.0 for _ in range(dim)]
    for row, target in zip(features, targets):
        for i in range(dim):
            vector[i] += row[i] * target
            for j in range(dim):
                matrix[i][j] += row[i] * row[j]
    for i in range(1, dim):
        matrix[i][i] += lam

    solved = _solve_linear_system(matrix, vector)
    return Weights(
        bias=solved[0],
        w_children=solved[1],
        w_desc=solved[2],
        w_desc_value=solved[3],
        w_diversity=solved[4],
        w_info=solved[5],
        w_rev=solved[6],
        w_cost=solved[7],
    )


def predict_option_values(
    tasks: Dict[str, OptionTask],
    deps: Optional[Iterable[Tuple[str, str]]] = None,
    weights: Weights = Weights(),
) -> Dict[str, float]:
    """
    Compute option value predictions for all moves.

    Parameters
    ----------
    tasks : Dict[str, OptionTask]
        Move map keyed by UUID.
    deps : Optional[Iterable[Tuple[str, str]]]
        Optional dependency pairs (task, dep).
    weights : Weights, optional
        Model weights.

    Returns
    -------
    Dict[str, float]
        Predicted option values keyed by UUID.
    """
    now = now_local()
    return {
        uuid: option_value_score(uuid, tasks, deps=deps, weights=weights, now=now)
        for uuid in tasks
    }


def load_option_tasks(filters: Optional[Sequence[str]] = None) -> List[OptionTask]:
    """
    Load pending moves for option value scoring.

    Parameters
    ----------
    filters : Optional[Sequence[str]]
        Additional Taskwarrior filter tokens.

    Returns
    -------
    List[OptionTask]
        Pending moves.
    """
    filter_tokens = list(filters) if filters else []
    result = run_task_command([*filter_tokens, "status:pending", "export"], capture_output=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or "Taskwarrior export failed.")
    tasks = read_tasks_from_json(result.stdout or "")
    return [
        OptionTask.from_json(task)
        for task in tasks
        if isinstance(task, dict) and task.get("uuid")
    ]


def run_task_command(
    args: Sequence[str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """
    Execute a Taskwarrior command.

    Parameters
    ----------
    args : Sequence[str]
        Taskwarrior arguments excluding the executable.
    capture_output : bool, optional
        Whether to capture stdout/stderr (default: False).

    Returns
    -------
    subprocess.CompletedProcess
        Process completion data.
    """
    kwargs: Dict[str, Any] = {"check": False}
    if capture_output:
        kwargs.update({"capture_output": True, "text": True})
    return subprocess.run(["task", *args], **kwargs)


def apply_option_values(
    tasks: Iterable[OptionTask],
    predictions: Dict[str, float],
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> int:
    """
    Apply auto option value predictions to Taskwarrior.

    Parameters
    ----------
    tasks : Iterable[OptionTask]
        Moves to update.
    predictions : Dict[str, float]
        Predicted option values by UUID.
    runner : Callable[..., subprocess.CompletedProcess], optional
        Runner for task commands (default: run_task_command).

    Returns
    -------
    int
        Exit code.
    """
    if runner is None:
        runner = run_task_command

    updates: List[Tuple[str, Dict[str, str]]] = []
    required_udas: Set[str] = set()
    for task in tasks:
        task_updates: Dict[str, str] = {}
        predicted = predictions.get(task.uuid)
        if predicted is not None:
            if task.opt_auto is None or abs(task.opt_auto - predicted) >= 0.05:
                task_updates["opt_auto"] = format_option_value(predicted)
                required_udas.add("opt_auto")
        if task.opt_human is None and task.opt is not None:
            task_updates["opt_human"] = format_option_value(task.opt)
            required_udas.add("opt_human")
        if task_updates:
            updates.append((task.uuid, task_updates))

    if not updates:
        return 0

    missing = missing_udas(sorted(required_udas))
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            "Missing Taskwarrior UDA(s): "
            f"{missing_list}. Aborting to avoid modifying move descriptions."
        )

    exit_code = 0
    for uuid, task_updates in updates:
        parts = [f"{key}:{value}" for key, value in task_updates.items()]
        result = runner([uuid, "modify", *parts], capture_output=True)
        for line in filter_modified_zero_lines(result.stdout):
            print(line)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            exit_code = result.returncode
    return exit_code


def format_weights(weights: Weights) -> str:
    """
    Format weights for display.

    Parameters
    ----------
    weights : Weights
        Model weights.

    Returns
    -------
    str
        Formatted weight string.
    """
    return (
        "bias={bias:.2f} children={w_children:.2f} desc={w_desc:.2f} "
        "desc_value={w_desc_value:.2f} diversity={w_diversity:.2f} info={w_info:.2f} "
        "rev={w_rev:.2f} cost={w_cost:.2f}"
    ).format(**weights.__dict__)


def run_option_value(
    *,
    filters: Optional[Sequence[str]] = None,
    apply: bool = False,
    include_rated: bool = False,
    limit: int = 25,
    ridge: float = 1.0,
) -> int:
    """
    Run the option value workflow.

    Parameters
    ----------
    filters : Optional[Sequence[str]]
        Additional Taskwarrior filter tokens.
    apply : bool, optional
        Write opt_auto values when True (default: False).
    include_rated : bool, optional
        Include moves with manual opt ratings in output (default: False).
    limit : int, optional
        Max number of moves to display (default: 25).
    ridge : float, optional
        Ridge regularization strength (default: 1.0).

    Returns
    -------
    int
        Exit code.
    """
    try:
        tasks = load_option_tasks(filters=filters)
    except RuntimeError as exc:
        print(f"twh: option value failed: {exc}")
        return 1

    if not tasks:
        print("No pending moves found.")
        return 0

    task_map = {task.uuid: task for task in tasks}
    training: List[Tuple[str, float]] = []
    for task in tasks:
        manual = manual_option_value(task)
        if manual is not None:
            training.append((task.uuid, manual))
    weights = fit_weights_ridge(training, task_map, lam=ridge)

    if training:
        print(f"Option value calibrated from {len(training)} move(s).")
    else:
        print(
            "Option value using default weights (no manual opt_human ratings found on moves)."
        )
    print(f"Weights: {format_weights(weights)}")

    predictions = predict_option_values(task_map, weights=weights)
    ranked = sorted(
        tasks,
        key=lambda task: (
            -(predictions.get(task.uuid, 0.0)),
            task.id if task.id is not None else 10**9,
            task.uuid,
        ),
    )

    shown = 0
    for task in ranked:
        manual = manual_option_value(task)
        if not include_rated and manual is not None:
            continue
        if shown >= limit:
            break
        move_id = str(task.id) if task.id is not None else task.uuid[:8]
        predicted = predictions.get(task.uuid, 0.0)
        opt_human_value = (
            "-" if task.opt_human is None else format_option_value(task.opt_human)
        )
        legacy_opt_value = ""
        if task.opt is not None and task.opt_human is None:
            legacy_opt_value = f"  opt={format_option_value(task.opt)}"
        print(
            f"[{move_id}] {task.description}  "
            f"opt_auto={format_option_value(predicted)}  "
            f"opt_human={opt_human_value}{legacy_opt_value}"
        )
        shown += 1

    if apply:
        try:
            exit_code = apply_option_values(tasks, predictions)
        except RuntimeError as exc:
            print(f"twh: option value failed: {exc}")
            return 1
        return exit_code
    return 0

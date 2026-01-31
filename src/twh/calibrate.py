#!/usr/bin/env python3
"""
Interactive calibration for precedence and option value scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import calibration
from . import option_value
from . import review


@dataclass(frozen=True)
class CalibrationTask:
    """
    Normalized move features for precedence calibration.

    Attributes
    ----------
    uuid : str
        Move UUID.
    move_id : str
        Move identifier (ID or UUID prefix).
    description : str
        Move description.
    enablement : float
        Normalized enablement score (0-1).
    blocker : float
        Normalized blocker relief score (0-1).
    difficulty : float
        Normalized difficulty ease score (0-1).
    dependency : float
        Dependency centrality score (0-1+).
    mode_multiplier : float
        Mode multiplier applied to precedence.
    """

    uuid: str
    move_id: str
    description: str
    enablement: float
    blocker: float
    difficulty: float
    dependency: float
    mode_multiplier: float


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _logistic(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _format_task_line(task: CalibrationTask, label: str) -> str:
    description = task.description.strip()
    if description:
        base = f"[{label}] Move ID {task.move_id}: {description}"
    else:
        base = f"[{label}] Move ID {task.move_id}"
    return (
        f"{base} "
        f"(E={task.enablement:.2f}, "
        f"B={task.blocker:.2f}, "
        f"D={task.difficulty:.2f}, "
        f"Dep={task.dependency:.2f}, "
        f"mode*{task.mode_multiplier:.2f})"
    )


def _precedence_score(task: CalibrationTask, weights: Dict[str, float]) -> float:
    base = (
        weights["enablement"] * task.enablement
        + weights["blocker"] * task.blocker
        + weights["difficulty"] * task.difficulty
    )
    return base * (1.0 + weights["dependency"] * task.dependency) * task.mode_multiplier


def _precedence_sensitivity(
    task: CalibrationTask,
    weights: Dict[str, float],
) -> Dict[str, float]:
    base = (
        weights["enablement"] * task.enablement
        + weights["blocker"] * task.blocker
        + weights["difficulty"] * task.difficulty
    )
    mult = (1.0 + weights["dependency"] * task.dependency) * task.mode_multiplier
    return {
        "enablement": task.enablement * mult,
        "blocker": task.blocker * mult,
        "difficulty": task.difficulty * mult,
        "dependency": base * task.dependency * task.mode_multiplier,
    }


def update_precedence_weights(
    weights: Dict[str, float],
    left: CalibrationTask,
    right: CalibrationTask,
    *,
    chose_left: bool,
    alpha: float,
) -> Dict[str, float]:
    """
    Update precedence weights from a single pairwise choice.

    Parameters
    ----------
    weights : Dict[str, float]
        Current precedence weights.
    left : CalibrationTask
        Left move features.
    right : CalibrationTask
        Right move features.
    chose_left : bool
        True when the user prefers the left move.
    alpha : float
        Learning rate.

    Returns
    -------
    Dict[str, float]
        Updated, normalized weights.
    """
    score_left = _precedence_score(left, weights)
    score_right = _precedence_score(right, weights)
    prob_left = _logistic(score_left - score_right)
    target = 1.0 if chose_left else 0.0
    gradient = target - prob_left
    left_sens = _precedence_sensitivity(left, weights)
    right_sens = _precedence_sensitivity(right, weights)

    updated = dict(weights)
    for key in weights:
        delta = alpha * gradient * (left_sens[key] - right_sens[key])
        updated[key] = updated[key] + delta
    return calibration.normalize_precedence_weights(
        updated,
        keys=list(weights.keys()),
        fallback=weights,
    )


def _task_has_precedence_inputs(task: review.ReviewTask) -> bool:
    return any(
        value is not None
        for value in (
            task.enablement,
            task.blocker_relief,
            task.estimate_hours,
            task.diff,
        )
    )


def build_calibration_tasks(
    tasks: Iterable[review.ReviewTask],
    stats: review.PrecedenceGraphStats,
) -> List[CalibrationTask]:
    """
    Build normalized precedence calibration tasks.

    Parameters
    ----------
    tasks : Iterable[review.ReviewTask]
        Review tasks to transform.
    stats : review.PrecedenceGraphStats
        Graph stats for dependency centrality.

    Returns
    -------
    List[CalibrationTask]
        Normalized calibration tasks.
    """
    calibration_tasks: List[CalibrationTask] = []
    for task in tasks:
        if not _task_has_precedence_inputs(task):
            continue
        enablement = review.normalize_score(task.enablement or 0.0, 0.0, 10.0)
        blocker = review.normalize_score(task.blocker_relief or 0.0, 0.0, 10.0)
        estimate = task.estimate_hours if task.estimate_hours is not None else task.diff
        difficulty = review.hours_score(estimate)
        out_degree = review.normalize_score(
            float(stats.out_degree.get(task.uuid, 0)),
            0.0,
            float(stats.max_out_degree),
        )
        critical_len = review.normalize_score(
            float(stats.critical_path_len.get(task.uuid, 0)),
            0.0,
            float(stats.max_critical_path_len),
        )
        dependency = out_degree * (1.0 + critical_len)
        move_id = str(task.id) if task.id is not None else task.uuid[:8]
        calibration_tasks.append(
            CalibrationTask(
                uuid=task.uuid,
                move_id=move_id,
                description=task.description,
                enablement=enablement,
                blocker=blocker,
                difficulty=difficulty,
                dependency=dependency,
                mode_multiplier=review.precedence_mode_multiplier(task.mode),
            )
        )
    return calibration_tasks


def _pair_contrast_score(left: CalibrationTask, right: CalibrationTask) -> float:
    return (
        abs(left.enablement - right.enablement)
        + abs(left.blocker - right.blocker)
        + abs(left.difficulty - right.difficulty)
        + abs(left.dependency - right.dependency)
    )


def select_calibration_pairs(
    tasks: Sequence[CalibrationTask],
    count: int,
) -> List[Tuple[CalibrationTask, CalibrationTask]]:
    """
    Select contrastive calibration pairs.

    Parameters
    ----------
    tasks : Sequence[CalibrationTask]
        Candidate calibration tasks.
    count : int
        Maximum number of pairs to return.

    Returns
    -------
    List[Tuple[CalibrationTask, CalibrationTask]]
        Selected calibration pairs.

    Examples
    --------
    >>> tasks = [
    ...     CalibrationTask("a", "1", "", 0.0, 0.0, 0.0, 0.0, 1.0),
    ...     CalibrationTask("b", "2", "", 1.0, 0.0, 0.0, 0.0, 1.0),
    ... ]
    >>> select_calibration_pairs(tasks, 1)[0][0].uuid
    'a'
    """
    if count <= 0 or len(tasks) < 2:
        return []
    scored: List[Tuple[float, int, int]] = []
    for idx, left in enumerate(tasks):
        for jdx in range(idx + 1, len(tasks)):
            right = tasks[jdx]
            scored.append((_pair_contrast_score(left, right), idx, jdx))
    scored.sort(key=lambda item: item[0], reverse=True)
    pairs: List[Tuple[CalibrationTask, CalibrationTask]] = []
    for _score, idx, jdx in scored[:count]:
        pairs.append((tasks[idx], tasks[jdx]))
    return pairs


def _format_weights(weights: Dict[str, float]) -> str:
    return (
        "enablement={enablement:.2f} blocker={blocker:.2f} "
        "difficulty={difficulty:.2f} dependency={dependency:.2f}"
    ).format(**weights)


def run_calibrate(
    *,
    pairs: int,
    alpha: float,
    ridge: float,
    apply: bool,
    filters: Optional[Sequence[str]] = None,
    input_func: Callable[[str], str] = input,
) -> int:
    """
    Run interactive calibration for precedence and option value.

    Parameters
    ----------
    pairs : int
        Number of pairwise comparisons to prompt.
    alpha : float
        Learning rate for precedence calibration.
    ridge : float
        Ridge regularization strength for option value calibration.
    apply : bool
        Apply opt_auto updates when True.
    filters : Optional[Sequence[str]]
        Taskwarrior filters to scope calibration.
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).

    Returns
    -------
    int
        Exit code.
    """
    try:
        pending = review.load_pending_tasks(filters=filters)
    except RuntimeError as exc:
        print(f"twh: calibrate failed: {exc}")
        return 1

    if not pending:
        print("No pending moves found.")
        return 0

    stats = review.build_precedence_graph_stats(pending)
    calibration_tasks = build_calibration_tasks(pending, stats)
    existing = calibration.load_calibration()

    precedence_weights = calibration.load_precedence_weights(review.PRECEDENCE_WEIGHTS)
    precedence_updates = 0

    pairs_to_prompt = select_calibration_pairs(calibration_tasks, pairs)
    if len(calibration_tasks) < 2:
        print("Not enough moves with precedence metadata to calibrate precedence weights.")
    elif not pairs_to_prompt:
        print("No precedence calibration pairs selected.")
    else:
        print(f"Precedence calibration using {len(pairs_to_prompt)} comparison(s).")
        for idx, (left, right) in enumerate(pairs_to_prompt, start=1):
            print("\n---")
            print(f"Comparison {idx} of {len(pairs_to_prompt)}")
            print(_format_task_line(left, "A"))
            print(_format_task_line(right, "B"))
            print("Choose which move should come first.")
            print("[A] Choose move A, [B] choose move B, [T] tie, [S] skip, [Q] quit")
            while True:
                choice = input_func("Selection (A/B/T/S/Q): ").strip().upper()
                if choice in {"A", "B", "T", "S", "Q"}:
                    break
                print("Please enter A, B, T, S, or Q.")
            if choice == "Q":
                break
            if choice in {"T", "S"}:
                continue
            precedence_weights = update_precedence_weights(
                precedence_weights,
                left,
                right,
                chose_left=(choice == "A"),
                alpha=alpha,
            )
            precedence_updates += 1

    if precedence_updates == 0 and existing and existing.precedence:
        precedence_section = existing.precedence
    else:
        precedence_section = calibration.CalibrationSection(
            weights=precedence_weights,
            meta={
                "alpha": alpha,
                "samples": precedence_updates,
                "updated": _iso_utc_now(),
            },
        )

    try:
        option_tasks = option_value.load_option_tasks(filters=filters)
    except RuntimeError as exc:
        print(f"twh: calibrate failed: {exc}")
        return 1

    task_map = {task.uuid: task for task in option_tasks}
    training: List[Tuple[str, float]] = []
    for task in option_tasks:
        manual = option_value.manual_option_value(task)
        if manual is not None:
            training.append((task.uuid, manual))

    option_section: Optional[calibration.CalibrationSection]
    if training:
        option_weights = option_value.fit_weights_ridge(training, task_map, lam=ridge)
        option_section = calibration.CalibrationSection(
            weights=option_value.weights_to_mapping(option_weights),
            meta={
                "ridge": ridge,
                "samples": len(training),
                "updated": _iso_utc_now(),
            },
        )
        print(f"Option value calibrated from {len(training)} move(s).")
    elif existing and existing.option_value:
        option_section = existing.option_value
        option_weights = option_value.weights_from_mapping(option_section.weights)
        print("Option value weights kept from existing calibration.")
    else:
        option_weights = option_value.Weights()
        option_section = calibration.CalibrationSection(
            weights=option_value.weights_to_mapping(option_weights),
            meta={
                "ridge": ridge,
                "samples": 0,
                "updated": _iso_utc_now(),
            },
        )
        print("Option value using default weights (no manual opt_human ratings found).")

    if precedence_updates > 0:
        print(f"Precedence weights updated: {_format_weights(precedence_weights)}")
    elif existing and existing.precedence:
        print("Precedence weights kept from existing calibration.")
    else:
        print(f"Precedence weights using defaults: {_format_weights(precedence_weights)}")

    data = calibration.CalibrationData(
        precedence=precedence_section,
        option_value=option_section,
    )
    path = calibration.save_calibration(data)
    print(f"Calibration saved to {path}.")

    if not apply:
        return 0

    predictions = option_value.predict_option_values(task_map, weights=option_weights)
    try:
        return option_value.apply_option_values(option_tasks, predictions)
    except RuntimeError as exc:
        print(f"twh: calibrate failed: {exc}")
        return 1

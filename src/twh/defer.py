#!/usr/bin/env python3
"""
Defer the top move from the ondeck list by setting a future start time.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Callable, Optional, Sequence, Tuple

from . import run_task_command
from .diagnose import format_move_snapshot, order_ondeck_moves
from .review import ReviewTask, get_local_timezone, load_pending_tasks
from .taskwarrior import filter_modified_zero_lines

DEFER_PROMPT = "Defer for how long (m/h/d/w)? "
DEFER_HINT = "Enter a number followed by m/h/d/w (or minutes/hours/days/weeks)."

UNIT_ALIASES = {
    "m": "minute",
    "minute": "minute",
    "minutes": "minute",
    "h": "hour",
    "hour": "hour",
    "hours": "hour",
    "d": "day",
    "day": "day",
    "days": "day",
    "w": "week",
    "week": "week",
    "weeks": "week",
}
UNIT_FIELDS = {
    "minute": "minutes",
    "hour": "hours",
    "day": "days",
    "week": "weeks",
}


def _localize_datetime(value: datetime) -> datetime:
    """
    Normalize datetimes to the local timezone for display.

    Parameters
    ----------
    value : datetime
        Datetime to normalize.

    Returns
    -------
    datetime
        Localized datetime.

    Examples
    --------
    >>> _localize_datetime(datetime(2026, 2, 2, 18, 45))
    datetime.datetime(2026, 2, 2, 18, 45)
    """
    if value.tzinfo is None:
        return value
    return value.astimezone(get_local_timezone())


def parse_defer_interval(text: str) -> Tuple[int, str, timedelta]:
    """
    Parse a defer interval input string.

    Parameters
    ----------
    text : str
        Raw input string (e.g., "15 m").

    Returns
    -------
    Tuple[int, str, timedelta]
        Parsed amount, canonical unit name, and delta.

    Raises
    ------
    ValueError
        If the input cannot be parsed.

    Examples
    --------
    >>> parse_defer_interval("15 m")
    (15, 'minute', datetime.timedelta(seconds=900))
    >>> parse_defer_interval("2 hours")[1]
    'hour'
    """
    if text is None:
        raise ValueError("Missing interval")
    raw = text.strip().lower()
    if not raw:
        raise ValueError("Missing interval")
    match = re.match(r"^(\d+)\s*([a-z]+)$", raw)
    if not match:
        raise ValueError("Invalid interval")
    amount = int(match.group(1))
    if amount <= 0:
        raise ValueError("Interval must be positive")
    unit_raw = match.group(2)
    unit = UNIT_ALIASES.get(unit_raw)
    if not unit:
        raise ValueError("Unknown interval unit")
    field = UNIT_FIELDS[unit]
    delta = timedelta(**{field: amount})
    return amount, unit, delta


def format_defer_timestamp(value: datetime) -> str:
    """
    Format a datetime for defer annotations.

    Parameters
    ----------
    value : datetime
        Datetime to format.

    Returns
    -------
    str
        Formatted timestamp (YYYY-MM-DD HH:MM).

    Examples
    --------
    >>> format_defer_timestamp(datetime(2026, 2, 2, 18, 45))
    '2026-02-02 18:45'
    """
    localized = _localize_datetime(value)
    return localized.strftime("%Y-%m-%d %H:%M")


def format_task_start_timestamp(value: datetime) -> str:
    """
    Format a datetime for Taskwarrior start updates.

    Parameters
    ----------
    value : datetime
        Datetime to format.

    Returns
    -------
    str
        Taskwarrior-compatible timestamp string.

    Examples
    --------
    >>> format_task_start_timestamp(datetime(2026, 2, 3, 18, 45, 5))
    '2026-02-03T18:45:05'
    """
    localized = _localize_datetime(value)
    return localized.strftime("%Y-%m-%dT%H:%M:%S")


def format_defer_annotation(
    now: datetime,
    target: datetime,
    amount: int,
    unit: str,
) -> str:
    """
    Build the deferral annotation string.

    Parameters
    ----------
    now : datetime
        Current datetime.
    target : datetime
        Deferred-until datetime.
    amount : int
        Deferral magnitude.
    unit : str
        Canonical unit label (minute/hour/day/week).

    Returns
    -------
    str
        Annotation text.

    Examples
    --------
    >>> now = datetime(2026, 2, 2, 18, 45)
    >>> target = datetime(2026, 2, 3, 18, 45)
    >>> format_defer_annotation(now, target, 1, "day")
    '2026-02-02 18:45 -- Deferred for 1 day to 2026-02-03 18:45.'
    """
    label = unit if amount == 1 else f"{unit}s"
    return (
        f"{format_defer_timestamp(now)} -- "
        f"Deferred for {amount} {label} to {format_defer_timestamp(target)}."
    )


def prompt_defer_interval(
    *,
    input_func: Callable[[str], str] = input,
) -> Tuple[int, str, timedelta]:
    """
    Prompt until a valid defer interval is provided.

    Parameters
    ----------
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).

    Returns
    -------
    Tuple[int, str, timedelta]
        Parsed amount, unit label, and delta.
    """
    while True:
        raw = input_func(DEFER_PROMPT)
        try:
            return parse_defer_interval(raw)
        except ValueError:
            print(DEFER_HINT)


def _format_top_move_summary(move: ReviewTask) -> None:
    """
    Print summary information about the top move.

    Parameters
    ----------
    move : ReviewTask
        Move to summarize.

    Returns
    -------
    None
        Prints summary lines.
    """
    for line in format_move_snapshot(move, "Top move"):
        print(line)


def run_defer(
    *,
    mode: Optional[str],
    strict_mode: bool,
    include_dominated: bool,
    filters: Optional[Sequence[str]] = None,
    input_func: Callable[[str], str] = input,
    now: Optional[datetime] = None,
    pending_loader: Callable[..., list[ReviewTask]] = load_pending_tasks,
    orderer: Callable[..., list[ReviewTask]] = order_ondeck_moves,
    task_runner: Callable[..., subprocess.CompletedProcess] = run_task_command,
) -> int:
    """
    Defer the top move on the ondeck list.

    Parameters
    ----------
    mode : Optional[str]
        Current mode context.
    strict_mode : bool
        Require mode match when True.
    include_dominated : bool
        Include dominated moves when True.
    filters : Optional[Sequence[str]]
        Additional Taskwarrior filter tokens.
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).
    now : Optional[datetime], optional
        Override for the current time (default: now in local timezone).
    pending_loader : Callable[..., list[ReviewTask]], optional
        Loader for pending moves (default: review.load_pending_tasks).
    orderer : Callable[..., list[ReviewTask]], optional
        Ordering function for ondeck moves (default: diagnose.order_ondeck_moves).
    task_runner : Callable[..., subprocess.CompletedProcess], optional
        Taskwarrior runner (default: run_task_command).

    Returns
    -------
    int
        Exit code.
    """
    try:
        pending = pending_loader(filters=filters)
    except RuntimeError as exc:
        print(f"twh: defer failed: {exc}", file=sys.stderr)
        return 1

    if not pending:
        print("No pending moves found.")
        return 0

    ordered = orderer(
        pending,
        current_mode=mode,
        strict_mode=strict_mode,
        include_dominated=include_dominated,
    )
    if not ordered:
        print("No ready moves found (or all were filtered).")
        return 0

    top_move = ordered[0]
    _format_top_move_summary(top_move)

    amount, unit, delta = prompt_defer_interval(input_func=input_func)
    now_value = now or datetime.now().astimezone()
    target = now_value + delta

    start_value = format_task_start_timestamp(target)
    note = format_defer_annotation(now_value, target, amount, unit)
    result = task_runner(
        [top_move.uuid, "modify", f"start:{start_value}", f"annotate:{note}"],
        capture_output=True,
    )
    for line in filter_modified_zero_lines(result.stdout):
        print(line)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        return result.returncode

    move_id = str(top_move.id) if top_move.id is not None else top_move.uuid[:8]
    print(f"Deferred move {move_id} to {format_defer_timestamp(target)}.")
    return 0

#!/usr/bin/env python3
"""
Restore Taskwarrior move descriptions from history when overwritten by diff values.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .taskwarrior import apply_taskrc_overrides, get_taskrc_path, read_tasks_from_json


DIFF_PREFIX = "diff:"


@dataclass(frozen=True)
class RestoreCandidate:
    """
    Move data needed to restore a description.

    Attributes
    ----------
    uuid : str
        Move UUID.
    move_id : Optional[int]
        Taskwarrior ID, if available.
    current_description : str
        Current description value.
    previous_description : str
        Previous description to restore.
    """

    uuid: str
    move_id: Optional[int]
    current_description: str
    previous_description: str


def find_previous_description(history_text: str, current_description: str) -> Optional[str]:
    """
    Find the last description value before the current description.

    Parameters
    ----------
    history_text : str
        Taskwarrior history output.
    current_description : str
        Current description value.

    Returns
    -------
    Optional[str]
        Previous description if found.
    """
    prefix = "Description changed from '"
    marker = f"' to '{current_description}'"
    previous: Optional[str] = None
    for line in history_text.splitlines():
        line = line.strip()
        if prefix not in line or marker not in line:
            continue
        start = line.index(prefix) + len(prefix)
        mid = line.rindex(marker)
        previous = line[start:mid]
    return previous


def collect_restore_candidates(
    tasks: Iterable[Dict[str, Any]],
    history_fetcher: Callable[[str], str],
) -> List[RestoreCandidate]:
    """
    Build a list of moves that need description restoration.

    Parameters
    ----------
    tasks : Iterable[Dict[str, Any]]
        Taskwarrior export payloads.
    history_fetcher : Callable[[str], str]
        Callable that returns history text for a UUID.

    Returns
    -------
    List[RestoreCandidate]
        Restore candidates with previous descriptions.
    """
    candidates: List[RestoreCandidate] = []
    for task in tasks:
        uuid = str(task.get("uuid", "")).strip()
        if not uuid:
            continue
        description = str(task.get("description", "")).strip()
        if not description.startswith(DIFF_PREFIX):
            continue
        history = history_fetcher(uuid)
        previous = find_previous_description(history, description)
        if not previous:
            continue
        candidates.append(
            RestoreCandidate(
                uuid=uuid,
                move_id=task.get("id"),
                current_description=description,
                previous_description=previous,
            )
        )
    return candidates


def apply_restores(
    candidates: Sequence[RestoreCandidate],
    apply: bool = False,
    runner: Optional[Callable[..., Any]] = None,
) -> None:
    """
    Apply description restores with an optional dry run.

    Parameters
    ----------
    candidates : Sequence[RestoreCandidate]
        Restore candidates to process.
    apply : bool, optional
        When True, apply modifications (default: False).
    runner : Optional[Callable[..., Any]]
        Runner for Taskwarrior commands.
    """
    if runner is None:
        def task_runner(args, **kwargs):
            if not args or args[0] != "task":
                raise ValueError("Expected Taskwarrior args starting with 'task'.")
            task_args = apply_taskrc_overrides(list(args[1:]))
            return subprocess.run(["task", *task_args], **kwargs)

        runner = task_runner

    for candidate in candidates:
        move_id = (
            str(candidate.move_id)
            if candidate.move_id is not None
            else candidate.uuid[:8]
        )
        print(
            f"[{move_id}] {candidate.current_description} -> {candidate.previous_description}"
        )
        if not apply:
            continue
        runner(
            [
                "task",
                candidate.uuid,
                "modify",
                f"description:{candidate.previous_description}",
            ],
            check=False,
        )


def build_rc_overrides(taskrc: Optional[str], data_location: Optional[str]) -> List[str]:
    """
    Build Taskwarrior rc overrides for the canonical taskrc and data location.
    """
    overrides: List[str] = []
    canonical = get_taskrc_path()
    if taskrc:
        resolved = Path(taskrc).expanduser()
        if canonical and resolved != canonical:
            raise RuntimeError(
                "twh enforces a single Taskwarrior config at ~/.taskrc. "
                "Remove --taskrc or update it to point at ~/.taskrc."
            )
    if canonical:
        overrides.append(f"rc:{canonical}")
    if data_location:
        overrides.append(f"rc.data.location:{data_location}")
    return overrides


def fetch_task_info(uuid: str, rc_overrides: Sequence[str]) -> str:
    """
    Fetch Taskwarrior info output for a move.

    Parameters
    ----------
    uuid : str
        Move UUID.

    Returns
    -------
    str
        Taskwarrior info output.
    """
    task_args = apply_taskrc_overrides(list(rc_overrides))
    result = subprocess.run(
        ["task", *task_args, uuid, "info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        hint = ""
        if "No Taskfile found" in stderr:
            hint = " (ensure ~/.taskrc points at your Taskwarrior data or pass --data)"
        raise RuntimeError(
            (stderr or f"Taskwarrior info failed for {uuid}.") + hint
        )
    return result.stdout or ""


def export_tasks(filters: Sequence[str], rc_overrides: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Export tasks using Taskwarrior filters.

    Parameters
    ----------
    filters : Sequence[str]
        Taskwarrior filter tokens.

    Returns
    -------
    List[Dict[str, Any]]
        Taskwarrior export payloads.
    """
    task_args = apply_taskrc_overrides(list(rc_overrides))
    result = subprocess.run(
        ["task", *task_args, *filters, "export"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        hint = ""
        if "No Taskfile found" in stderr:
            hint = " (ensure ~/.taskrc points at your Taskwarrior data or pass --data)"
        raise RuntimeError((stderr or "Taskwarrior export failed.") + hint)
    return read_tasks_from_json(result.stdout or "")


def build_parser() -> argparse.ArgumentParser:
    """
    Build an argument parser for the restore tool.
    """
    parser = argparse.ArgumentParser(
        prog="twh-restore-descriptions",
        description=(
            "Restore move descriptions from Taskwarrior history when overwritten "
            "by diff values."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply description restores (default: dry run).",
    )
    parser.add_argument(
        "--taskrc",
        help="Path to taskrc file (must be ~/.taskrc).",
    )
    parser.add_argument(
        "--data",
        help="Taskwarrior data location (optional).",
    )
    parser.add_argument(
        "filters",
        nargs="*",
        help="Optional Taskwarrior filter tokens (e.g., project:work).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the restore workflow.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    filters = list(args.filters)
    if "description:diff:" not in filters:
        filters.append("description:diff:")

    rc_overrides = build_rc_overrides(args.taskrc, args.data)
    tasks = export_tasks(filters, rc_overrides)
    candidates = collect_restore_candidates(
        tasks,
        lambda uuid: fetch_task_info(uuid, rc_overrides),
    )

    if not candidates:
        print("No description restores found.")
        return 0

    if not args.apply:
        print("Dry run only. Re-run with --apply to restore descriptions.")

    apply_restores(candidates, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

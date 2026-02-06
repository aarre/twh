#!/usr/bin/env python3
"""
Time logging support for twh start/stop workflows.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .taskwarrior import (
    apply_case_insensitive_overrides,
    apply_taskrc_overrides,
    filter_modified_zero_lines,
    get_task_data_location,
    read_tasks_from_json,
)


@dataclass(frozen=True)
class TaskSnapshot:
    """
    Snapshot of move metadata captured for time logging.

    Attributes
    ----------
    uuid : str
        Move UUID.
    description : str
        Move description at log time.
    project : Optional[str]
        Project name, if any.
    tags : Tuple[str, ...]
        Tags associated with the move.
    mode : Optional[str]
        Mode associated with the move.
    start : Optional[datetime]
        Taskwarrior start timestamp.
    """

    uuid: str
    description: str
    project: Optional[str]
    tags: Tuple[str, ...]
    mode: Optional[str]
    start: Optional[datetime] = None


@dataclass(frozen=True)
class TimeEntry:
    """
    Stored time entry for a move.

    Attributes
    ----------
    id : int
        Entry identifier.
    uuid : str
        Move UUID.
    description : str
        Move description at log time.
    project : Optional[str]
        Project name, if any.
    tags : Tuple[str, ...]
        Tags at log time.
    mode : Optional[str]
        Mode at log time.
    start : datetime
        Start timestamp.
    end : Optional[datetime]
        End timestamp, if any.
    """

    id: int
    uuid: str
    description: str
    project: Optional[str]
    tags: Tuple[str, ...]
    mode: Optional[str]
    start: datetime
    end: Optional[datetime]


def get_task_directory() -> Path:
    """
    Return the Taskwarrior data directory.

    Returns
    -------
    Path
        Taskwarrior data directory path.
    """
    return get_task_data_location()


def get_time_log_path() -> Path:
    """
    Return the time log database path.

    Returns
    -------
    Path
        SQLite database path for time logs.
    """
    return get_task_directory() / "twh-time.db"


def get_local_timezone() -> tzinfo:
    """
    Return the local timezone for formatting timestamps.

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


def normalize_to_utc(value: datetime) -> datetime:
    """
    Normalize a datetime to UTC with tzinfo.

    Parameters
    ----------
    value : datetime
        Datetime to normalize.

    Returns
    -------
    datetime
        UTC-normalized datetime.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def format_storage_timestamp(value: datetime) -> str:
    """
    Format timestamps for storage.

    Parameters
    ----------
    value : datetime
        Datetime to format.

    Returns
    -------
    str
        ISO timestamp in UTC.
    """
    utc_value = normalize_to_utc(value)
    return utc_value.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_storage_timestamp(value: Optional[str]) -> Optional[datetime]:
    """
    Parse stored timestamps.

    Parameters
    ----------
    value : Optional[str]
        Stored timestamp string.

    Returns
    -------
    Optional[datetime]
        Parsed datetime, or None.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_duration_seconds(value: str) -> int:
    """
    Parse duration strings into seconds.

    Parameters
    ----------
    value : str
        Duration string (e.g., "1.5h", "90m", "30s", "2d").

    Returns
    -------
    int
        Duration in seconds.

    Examples
    --------
    >>> parse_duration_seconds("1.5h")
    5400
    >>> parse_duration_seconds("90m")
    5400
    >>> parse_duration_seconds("2")
    7200
    """
    raw = value.strip().lower()
    if not raw:
        raise ValueError("Duration value is required.")
    unit = raw[-1]
    if unit.isdigit() or unit == ".":
        number = float(raw)
        return int(round(number * 3600))
    number = float(raw[:-1])
    if unit == "s":
        return int(round(number))
    if unit == "m":
        return int(round(number * 60))
    if unit == "h":
        return int(round(number * 3600))
    if unit == "d":
        return int(round(number * 86400))
    raise ValueError(f"Unknown duration unit: {unit}")


def serialize_tags(tags: Iterable[str]) -> str:
    return ",".join(tag for tag in tags if tag)


def parse_tags(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return ()
    return tuple(tag.strip() for tag in value.split(",") if tag.strip())


def project_rollups(project: Optional[str]) -> List[str]:
    """
    Return hierarchical project rollups.

    Parameters
    ----------
    project : Optional[str]
        Project name.

    Returns
    -------
    List[str]
        Project rollup list.

    Examples
    --------
    >>> project_rollups("alpha.beta")
    ['alpha', 'alpha.beta']
    >>> project_rollups(None)
    ['-']
    """
    if not project:
        return ["-"]
    parts = [part for part in project.split(".") if part]
    rollups: List[str] = []
    for idx in range(1, len(parts) + 1):
        rollups.append(".".join(parts[:idx]))
    return rollups if rollups else ["-"]


class TimeLogStore:
    """
    SQLite-backed storage for time logs.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or get_time_log_path()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.path)

    def ensure_schema(self) -> None:
        """
        Ensure the SQLite schema exists.
        """
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS time_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL,
                    description TEXT NOT NULL,
                    project TEXT,
                    tags TEXT,
                    mode TEXT,
                    start_ts TEXT NOT NULL,
                    end_ts TEXT,
                    updated_ts TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_time_entries_uuid ON time_entries(uuid)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_time_entries_start ON time_entries(start_ts)"
            )
            conn.commit()

    def add_entry(
        self,
        snapshot: TaskSnapshot,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> int:
        """
        Add a time entry.

        Parameters
        ----------
        snapshot : TaskSnapshot
            Task snapshot to store.
        start : datetime
            Start time.
        end : Optional[datetime], optional
            End time, if known.

        Returns
        -------
        int
            Entry identifier.
        """
        self.ensure_schema()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO time_entries (
                    uuid, description, project, tags, mode, start_ts, end_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.uuid,
                    snapshot.description,
                    snapshot.project,
                    serialize_tags(snapshot.tags),
                    snapshot.mode,
                    format_storage_timestamp(start),
                    format_storage_timestamp(end) if end else None,
                    format_storage_timestamp(datetime.now(timezone.utc)),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def fetch_entries(
        self,
        range_start: Optional[datetime] = None,
        range_end: Optional[datetime] = None,
    ) -> List[TimeEntry]:
        """
        Fetch entries, optionally filtered by a date range.

        Parameters
        ----------
        range_start : Optional[datetime], optional
            Range start (inclusive).
        range_end : Optional[datetime], optional
            Range end (exclusive).

        Returns
        -------
        List[TimeEntry]
            Matching time entries.
        """
        self.ensure_schema()
        query = "SELECT id, uuid, description, project, tags, mode, start_ts, end_ts FROM time_entries"
        params: List[Any] = []
        conditions: List[str] = []
        if range_start:
            conditions.append("end_ts IS NULL OR end_ts >= ?")
            params.append(format_storage_timestamp(range_start))
        if range_end:
            conditions.append("start_ts <= ?")
            params.append(format_storage_timestamp(range_end))
        if conditions:
            query += " WHERE " + " AND ".join(f"({clause})" for clause in conditions)
        query += " ORDER BY start_ts"
        entries: List[TimeEntry] = []
        with self._connect() as conn:
            for row in conn.execute(query, params):
                entries.append(
                    TimeEntry(
                        id=int(row[0]),
                        uuid=str(row[1]),
                        description=str(row[2]),
                        project=row[3],
                        tags=parse_tags(row[4]),
                        mode=row[5],
                        start=parse_storage_timestamp(row[6]) or datetime.now(timezone.utc),
                        end=parse_storage_timestamp(row[7]),
                    )
                )
        return entries

    def close_open_entries(
        self,
        end: datetime,
        uuids: Optional[Iterable[str]] = None,
    ) -> int:
        """
        Close open entries with an end timestamp.

        Parameters
        ----------
        end : datetime
            End timestamp to apply.
        uuids : Optional[Iterable[str]]
            Restrict to UUIDs when provided.

        Returns
        -------
        int
            Number of rows updated.
        """
        self.ensure_schema()
        end_ts = format_storage_timestamp(end)
        with self._connect() as conn:
            if uuids:
                uuid_list = list(uuids)
                placeholders = ",".join("?" for _ in uuid_list)
                cursor = conn.execute(
                    f"""
                    UPDATE time_entries
                    SET end_ts = ?, updated_ts = ?
                    WHERE end_ts IS NULL AND uuid IN ({placeholders})
                    """,
                    [end_ts, end_ts, *uuid_list],
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE time_entries
                    SET end_ts = ?, updated_ts = ?
                    WHERE end_ts IS NULL
                    """,
                    (end_ts, end_ts),
                )
            conn.commit()
            return int(cursor.rowcount or 0)

    def update_entry(
        self,
        entry_id: int,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> None:
        """
        Update a time entry.

        Parameters
        ----------
        entry_id : int
            Entry identifier.
        start : Optional[datetime], optional
            Updated start time.
        end : Optional[datetime], optional
            Updated end time.
        """
        self.ensure_schema()
        updates: List[str] = []
        params: List[Any] = []
        if start:
            updates.append("start_ts = ?")
            params.append(format_storage_timestamp(start))
        if end is not None:
            updates.append("end_ts = ?")
            params.append(format_storage_timestamp(end) if end else None)
        if not updates:
            return
        updates.append("updated_ts = ?")
        params.append(format_storage_timestamp(datetime.now(timezone.utc)))
        params.append(entry_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE time_entries SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

    def update_entry_duration(self, entry_id: int, duration: timedelta) -> None:
        """
        Update an entry's end time by duration.

        Parameters
        ----------
        entry_id : int
            Entry identifier.
        duration : timedelta
            New duration from start time.
        """
        entries = self.fetch_entries()
        entry = next((item for item in entries if item.id == entry_id), None)
        if not entry:
            raise ValueError(f"Entry {entry_id} not found.")
        end_time = entry.start + duration
        self.update_entry(entry_id, end=end_time)


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
    task_args = apply_case_insensitive_overrides(list(args))
    task_args = apply_taskrc_overrides(task_args)
    return subprocess.run(["task", *task_args], **kwargs)


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return text


def snapshot_from_payload(payload: Dict[str, Any]) -> TaskSnapshot:
    """
    Build a TaskSnapshot from Taskwarrior export data.
    """
    raw_tags = payload.get("tags") or []
    if isinstance(raw_tags, list):
        tags = tuple(_normalize_text(tag) for tag in raw_tags if _normalize_text(tag))
    else:
        tags = (str(raw_tags).strip(),) if _normalize_text(raw_tags) else ()
    return TaskSnapshot(
        uuid=str(payload.get("uuid", "")).strip(),
        description=_normalize_text(payload.get("description")),
        project=_normalize_text(payload.get("project")) or None,
        tags=tags,
        mode=_normalize_text(payload.get("mode")) or None,
        start=parse_task_timestamp(payload.get("start")),
    )


def load_task_snapshots(filters: Optional[Sequence[str]] = None) -> List[TaskSnapshot]:
    """
    Load task snapshots from Taskwarrior export.

    Parameters
    ----------
    filters : Optional[Sequence[str]]
        Taskwarrior filter tokens.

    Returns
    -------
    List[TaskSnapshot]
        Loaded task snapshots.
    """
    filter_tokens = list(filters) if filters else []
    result = run_task_command([*filter_tokens, "status:pending", "export"], capture_output=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or "Taskwarrior export failed.")
    tasks = read_tasks_from_json(result.stdout or "")
    snapshots: List[TaskSnapshot] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        snapshot = snapshot_from_payload(task)
        if snapshot.uuid:
            snapshots.append(snapshot)
    return snapshots


def load_active_snapshots() -> List[TaskSnapshot]:
    """
    Load active task snapshots.
    """
    snapshots = load_task_snapshots()
    return [snapshot for snapshot in snapshots if snapshot.start is not None]


def _task_label(entry: TimeEntry) -> str:
    prefix = entry.uuid[:8]
    description = entry.description.strip()
    return f"{prefix} {description}".strip()


def _normalize_local(dt: datetime, tzinfo: tzinfo) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tzinfo)
    return dt.astimezone(tzinfo)


def _period_start(dt: datetime, period: str) -> datetime:
    if period == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        start = dt - timedelta(days=dt.weekday())
        return start.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if period == "year":
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return dt


def _next_period_start(dt: datetime, period: str) -> datetime:
    if period == "day":
        return dt + timedelta(days=1)
    if period == "week":
        return dt + timedelta(days=7)
    if period == "month":
        year = dt.year + (dt.month // 12)
        month = 1 if dt.month == 12 else dt.month + 1
        return dt.replace(year=year, month=month, day=1)
    if period == "year":
        return dt.replace(year=dt.year + 1, month=1, day=1)
    return dt


def _bucket_key(dt: datetime, period: str) -> str:
    if period == "day":
        return dt.strftime("%Y-%m-%d")
    if period == "week":
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if period == "month":
        return dt.strftime("%Y-%m")
    if period == "year":
        return dt.strftime("%Y")
    return "range"


def _iter_entry_segments(
    start: datetime,
    end: datetime,
    period: str,
    tzinfo: tzinfo,
) -> Iterable[Tuple[str, int]]:
    if period == "range":
        seconds = int(max(0, (end - start).total_seconds()))
        if seconds:
            yield ("range", seconds)
        return
    local_start = _normalize_local(start, tzinfo)
    local_end = _normalize_local(end, tzinfo)
    cursor = local_start
    while cursor < local_end:
        bucket_start = _period_start(cursor, period)
        bucket_end = _next_period_start(bucket_start, period)
        segment_end = min(local_end, bucket_end)
        seconds = int(max(0, (segment_end - cursor).total_seconds()))
        if seconds:
            yield (_bucket_key(bucket_start, period), seconds)
        cursor = segment_end


def aggregate_entries(
    entries: Iterable[TimeEntry],
    *,
    group_by: str,
    period: str,
    tzinfo: Optional[tzinfo] = None,
    now: Optional[datetime] = None,
) -> Dict[Tuple[str, str], int]:
    """
    Aggregate entries by group and period.

    Parameters
    ----------
    entries : Iterable[TimeEntry]
        Time entries to aggregate.
    group_by : str
        Grouping dimension (task/project/tag/mode/total).
    period : str
        Period bucket (day/week/month/year/range).
    tzinfo : Optional[tzinfo], optional
        Timezone for bucketing.
    now : Optional[datetime], optional
        Reference time for open entries.

    Returns
    -------
    Dict[Tuple[str, str], int]
        Mapping of (period_key, group_key) to seconds.
    """
    valid_periods = {"day", "week", "month", "year", "range"}
    valid_groups = {"task", "project", "tag", "mode", "total"}
    if period not in valid_periods:
        raise ValueError(f"Unsupported period: {period}")
    if group_by not in valid_groups:
        raise ValueError(f"Unsupported group: {group_by}")
    tzinfo = tzinfo or get_local_timezone()
    now = now or datetime.now(tzinfo)
    totals: Dict[Tuple[str, str], int] = {}
    for entry in entries:
        end = entry.end or now
        if end <= entry.start:
            continue
        segments = _iter_entry_segments(entry.start, end, period, tzinfo)
        if group_by == "project":
            groups = project_rollups(entry.project)
        elif group_by == "tag":
            groups = list(entry.tags) if entry.tags else ["-"]
        elif group_by == "mode":
            groups = [entry.mode or "-"]
        elif group_by == "task":
            groups = [_task_label(entry)]
        else:
            groups = ["total"]
        for bucket, seconds in segments:
            for group in groups:
                key = (bucket, group)
                totals[key] = totals.get(key, 0) + seconds
    return totals


def format_duration(seconds: int) -> str:
    """
    Format seconds as hours with one decimal place.
    """
    hours = seconds / 3600.0
    return f"{hours:.1f}h"


def parse_user_datetime(value: str, *, end_of_day: bool = False) -> datetime:
    """
    Parse a user-provided datetime.

    Parameters
    ----------
    value : str
        Datetime input.
    end_of_day : bool, optional
        Use end-of-day when input is date-only.

    Returns
    -------
    datetime
        Parsed datetime with timezone.
    """
    text = value.strip()
    if not text:
        raise ValueError("Datetime value is required.")
    tzinfo = get_local_timezone()
    if "T" not in text and " " not in text:
        parsed = datetime.strptime(text, "%Y-%m-%d")
        if end_of_day:
            parsed = parsed.replace(hour=23, minute=59, second=59)
        return parsed.replace(tzinfo=tzinfo)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tzinfo)
    return parsed.astimezone(tzinfo)


def _print_taskwarrior_output(result: subprocess.CompletedProcess) -> None:
    for line in filter_modified_zero_lines(result.stdout if hasattr(result, "stdout") else ""):
        print(line)
    if getattr(result, "stderr", ""):
        print(result.stderr, end="", file=sys.stderr)


def run_start(filters: Sequence[str]) -> int:
    """
    Start a move and log time.
    """
    try:
        candidates = load_task_snapshots(filters)
    except RuntimeError as exc:
        print(f"twh: start failed: {exc}")
        return 1
    if len(candidates) != 1:
        print("twh: start requires a single matching move.", file=sys.stderr)
        return 1
    target = candidates[0]
    active = load_active_snapshots()
    target_active = any(task.uuid == target.uuid for task in active)
    other_active = [task for task in active if task.uuid != target.uuid]

    store = TimeLogStore()
    now = datetime.now(timezone.utc)
    open_entries = [entry for entry in store.fetch_entries() if entry.end is None]
    open_uuids = {entry.uuid for entry in open_entries}

    if not target_active:
        store.close_open_entries(now)
    elif open_uuids:
        other_open = [uuid for uuid in open_uuids if uuid != target.uuid]
        if other_open:
            store.close_open_entries(now, uuids=other_open)

    if other_active:
        for task in other_active:
            result = run_task_command([task.uuid, "stop"], capture_output=True)
            _print_taskwarrior_output(result)
        for task in other_active:
            if task.start and task.uuid not in open_uuids:
                store.add_entry(task, task.start, end=now)

    if not target_active:
        result = run_task_command([target.uuid, "start"], capture_output=True)
        _print_taskwarrior_output(result)
        if result.returncode != 0:
            return result.returncode

    if not target_active or target.uuid not in open_uuids:
        start_time = target.start or now
        store.add_entry(target, start_time)
    return 0


def run_stop(filters: Sequence[str]) -> int:
    """
    Stop active moves and log time.
    """
    try:
        candidates = load_task_snapshots(filters) if filters else []
    except RuntimeError as exc:
        print(f"twh: stop failed: {exc}")
        return 1
    active = load_active_snapshots()
    if candidates:
        active = [task for task in active if task.uuid in {c.uuid for c in candidates}]
    if not active:
        result = run_task_command([*filters, "stop"], capture_output=True)
        _print_taskwarrior_output(result)
        return result.returncode

    store = TimeLogStore()
    now = datetime.now(timezone.utc)
    open_entries = [entry for entry in store.fetch_entries() if entry.end is None]
    open_uuids = {entry.uuid for entry in open_entries}
    for task in active:
        result = run_task_command([task.uuid, "stop"], capture_output=True)
        _print_taskwarrior_output(result)
        if result.returncode != 0:
            return result.returncode
    for task in active:
        if task.start and task.uuid not in open_uuids:
            store.add_entry(task, task.start, end=now)
    store.close_open_entries(now, uuids=[task.uuid for task in active])
    return 0


def run_time_entries(
    *,
    range_start: Optional[str],
    range_end: Optional[str],
    limit: int,
) -> int:
    """
    List time entries.
    """
    store = TimeLogStore()
    start_dt = parse_user_datetime(range_start) if range_start else None
    end_dt = (
        parse_user_datetime(range_end, end_of_day=True) if range_end else None
    )
    entries = store.fetch_entries(start_dt, end_dt)
    if not entries:
        print("No time entries found.")
        return 0
    now = datetime.now(timezone.utc)
    shown = 0
    for entry in entries:
        if shown >= limit:
            break
        end = entry.end or now
        seconds = int(max(0, (end - entry.start).total_seconds()))
        print(
            f"[{entry.id}] {entry.uuid[:8]} {entry.description} "
            f"{format_storage_timestamp(entry.start)} -> "
            f"{format_storage_timestamp(end) if entry.end else 'active'} "
            f"{format_duration(seconds)}"
        )
        shown += 1
    return 0


def run_time_report(
    *,
    group_by: str,
    period: str,
    range_start: Optional[str],
    range_end: Optional[str],
) -> int:
    """
    Report aggregated time by group and period.
    """
    store = TimeLogStore()
    start_dt = parse_user_datetime(range_start) if range_start else None
    end_dt = (
        parse_user_datetime(range_end, end_of_day=True) if range_end else None
    )
    entries = store.fetch_entries(start_dt, end_dt)
    if not entries:
        print("No time entries found.")
        return 0
    totals = aggregate_entries(entries, group_by=group_by, period=period)
    if not totals:
        print("No time entries found.")
        return 0
    for (bucket, group), seconds in sorted(
        totals.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
    ):
        print(f"{bucket}  {group}  {format_duration(seconds)}")
    return 0


def run_time_edit(
    *,
    entry_id: int,
    start: Optional[str],
    end: Optional[str],
    duration: Optional[str],
) -> int:
    """
    Edit a time entry.
    """
    store = TimeLogStore()
    start_dt = parse_user_datetime(start) if start else None
    end_dt = parse_user_datetime(end, end_of_day=True) if end else None
    if duration:
        seconds = parse_duration_seconds(duration)
        store.update_entry_duration(entry_id, timedelta(seconds=seconds))
        print(f"Updated entry {entry_id} duration to {duration}.")
        return 0
    if start_dt or end_dt:
        store.update_entry(entry_id, start=start_dt, end=end_dt)
        print(f"Updated entry {entry_id}.")
        return 0
    print("twh: time edit requires --start, --end, or --duration.", file=sys.stderr)
    return 1

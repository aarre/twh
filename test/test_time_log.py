"""
Tests for time logging utilities.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import twh.time_log as time_log


@pytest.mark.parametrize(
    ("value", "expected_seconds"),
    [
        ("1h", 3600),
        ("1.5h", 5400),
        ("90m", 5400),
        ("30s", 30),
        ("2d", 172800),
        ("2", 7200),
    ],
)
@pytest.mark.unit
def test_parse_duration_seconds(value, expected_seconds):
    """
    Ensure duration strings parse into seconds.

    Returns
    -------
    None
        This test asserts duration parsing.
    """
    assert time_log.parse_duration_seconds(value) == expected_seconds


@pytest.mark.parametrize(
    ("project", "expected"),
    [
        ("alpha", ["alpha"]),
        ("alpha.beta", ["alpha", "alpha.beta"]),
        ("alpha.beta.gamma", ["alpha", "alpha.beta", "alpha.beta.gamma"]),
        (None, ["-"]),
        ("", ["-"]),
    ],
)
@pytest.mark.unit
def test_project_rollups(project, expected):
    """
    Ensure project rollups include each hierarchy level.

    Returns
    -------
    None
        This test asserts project rollup behavior.
    """
    assert time_log.project_rollups(project) == expected


@pytest.mark.unit
def test_time_log_store_add_and_close(tmp_path):
    """
    Ensure time entries are stored and closed correctly.

    Returns
    -------
    None
        This test asserts time log storage.
    """
    path = tmp_path / "time.db"
    store = time_log.TimeLogStore(path)
    store.ensure_schema()

    snapshot = time_log.TaskSnapshot(
        uuid="u1",
        description="Move 1",
        project="work.alpha",
        tags=("alpha", "beta"),
        mode="analysis",
        start=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
    )
    entry_id = store.add_entry(snapshot, snapshot.start)

    entries = store.fetch_entries()
    assert len(entries) == 1
    assert entries[0].id == entry_id
    assert entries[0].end is None

    store.close_open_entries(datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc))
    entries = store.fetch_entries()
    assert entries[0].end == datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)


@pytest.mark.unit
def test_update_entry_duration_sets_end(tmp_path):
    """
    Ensure duration edits update the end time.

    Returns
    -------
    None
        This test asserts duration editing.
    """
    path = tmp_path / "time.db"
    store = time_log.TimeLogStore(path)
    store.ensure_schema()

    start = datetime(2024, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
    entry_id = store.add_entry(
        time_log.TaskSnapshot(
            uuid="u1",
            description="Move",
            project=None,
            tags=(),
            mode=None,
            start=start,
        ),
        start,
    )

    store.update_entry_duration(entry_id, timedelta(hours=2))
    entry = store.fetch_entries()[0]
    assert entry.end == datetime(2024, 1, 2, 10, 0, 0, tzinfo=timezone.utc)


@pytest.mark.unit
def test_aggregate_entries_by_project_day():
    """
    Ensure project rollups are counted in daily aggregation.

    Returns
    -------
    None
        This test asserts aggregation rollups.
    """
    start = datetime(2024, 1, 3, 9, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, 10, 0, 0, tzinfo=timezone.utc)
    entry = time_log.TimeEntry(
        id=1,
        uuid="u1",
        description="Move",
        project="work.alpha",
        tags=("tag1",),
        mode="analysis",
        start=start,
        end=end,
    )

    totals = time_log.aggregate_entries(
        [entry],
        group_by="project",
        period="day",
        tzinfo=timezone.utc,
    )

    assert totals[("2024-01-03", "work")] == 3600
    assert totals[("2024-01-03", "work.alpha")] == 3600


@pytest.mark.unit
def test_aggregate_entries_by_tag_week():
    """
    Ensure tag aggregation splits durations per tag.

    Returns
    -------
    None
        This test asserts tag aggregation.
    """
    start = datetime(2024, 1, 4, 9, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 4, 9, 30, 0, tzinfo=timezone.utc)
    entry = time_log.TimeEntry(
        id=1,
        uuid="u1",
        description="Move",
        project=None,
        tags=("alpha", "beta"),
        mode=None,
        start=start,
        end=end,
    )

    totals = time_log.aggregate_entries(
        [entry],
        group_by="tag",
        period="week",
        tzinfo=timezone.utc,
    )

    assert totals[("2024-W01", "alpha")] == 1800
    assert totals[("2024-W01", "beta")] == 1800


@pytest.mark.unit
def test_run_start_stops_other_active_and_logs(monkeypatch):
    """
    Ensure start stops other active moves and logs entries.

    Returns
    -------
    None
        This test asserts start logging behavior.
    """
    target = time_log.TaskSnapshot(
        uuid="u1",
        description="Target",
        project=None,
        tags=(),
        mode=None,
        start=None,
        wip=False,
    )
    other = time_log.TaskSnapshot(
        uuid="u2",
        description="Other",
        project=None,
        tags=(),
        mode=None,
        start=datetime(2024, 1, 5, 9, 0, 0, tzinfo=timezone.utc),
        wip=True,
    )

    monkeypatch.setattr(time_log, "load_task_snapshots", lambda _filters=None: [target])
    monkeypatch.setattr(time_log, "load_active_snapshots", lambda: [other])
    monkeypatch.setattr(time_log, "ensure_wip_uda_present", lambda _cmd: True)

    calls = []

    def fake_run(args, capture_output=False):
        calls.append(args)
        return time_log.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(time_log, "run_task_command", fake_run)

    class FakeStore:
        def __init__(self):
            self.added = []

        def fetch_entries(self):
            return []

        def close_open_entries(self, *_args, **_kwargs):
            return 0

        def add_entry(self, snapshot, start, end=None):
            self.added.append((snapshot.uuid, start, end))
            return 1

    store = FakeStore()
    monkeypatch.setattr(time_log, "TimeLogStore", lambda *_args, **_kwargs: store)

    exit_code = time_log.run_start([])

    assert exit_code == 0
    assert calls == [
        ["u2", "stop"],
        ["u2", "modify", "wip:"],
        ["u1", "start"],
        ["u1", "modify", "wip:1"],
    ]
    assert {entry[0] for entry in store.added} == {"u1", "u2"}


@pytest.mark.unit
def test_load_active_snapshots_filters_wip(monkeypatch):
    """
    Ensure active snapshots are determined by wip status.

    Returns
    -------
    None
        This test asserts wip-based filtering.
    """
    active = time_log.TaskSnapshot(
        uuid="u1",
        description="Active",
        project=None,
        tags=(),
        mode=None,
        start=None,
        wip=True,
    )
    inactive = time_log.TaskSnapshot(
        uuid="u2",
        description="Inactive",
        project=None,
        tags=(),
        mode=None,
        start=None,
        wip=False,
    )
    monkeypatch.setattr(time_log, "load_task_snapshots", lambda _filters=None: [active, inactive])

    snapshots = time_log.load_active_snapshots()

    assert [snapshot.uuid for snapshot in snapshots] == ["u1"]


@pytest.mark.unit
def test_run_stop_closes_active_and_logs(monkeypatch):
    """
    Ensure stop closes active entries and logs missing ones.

    Returns
    -------
    None
        This test asserts stop logging behavior.
    """
    active = time_log.TaskSnapshot(
        uuid="u1",
        description="Active",
        project=None,
        tags=(),
        mode=None,
        start=datetime(2024, 1, 6, 9, 0, 0, tzinfo=timezone.utc),
        wip=True,
    )

    monkeypatch.setattr(time_log, "load_active_snapshots", lambda: [active])
    monkeypatch.setattr(time_log, "load_task_snapshots", lambda _filters=None: [])
    monkeypatch.setattr(time_log, "ensure_wip_uda_present", lambda _cmd: True)

    calls = []

    def fake_run(args, capture_output=False):
        calls.append(args)
        return time_log.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(time_log, "run_task_command", fake_run)

    class FakeStore:
        def __init__(self):
            self.added = []
            self.closed = []

        def fetch_entries(self):
            return []

        def close_open_entries(self, *_args, **_kwargs):
            self.closed.append(True)
            return 0

        def add_entry(self, snapshot, start, end=None):
            self.added.append((snapshot.uuid, start, end))
            return 1

    store = FakeStore()
    monkeypatch.setattr(time_log, "TimeLogStore", lambda *_args, **_kwargs: store)

    exit_code = time_log.run_stop([])

    assert exit_code == 0
    assert calls == [["u1", "stop"], ["u1", "modify", "wip:"]]
    assert {entry[0] for entry in store.added} == {"u1"}


@pytest.mark.unit
def test_time_log_doctest_examples():
    """
    Run doctest examples embedded in time log helpers.

    Returns
    -------
    None
        This test asserts doctest coverage for time log helpers.
    """
    import doctest

    results = doctest.testmod(time_log)
    assert results.failed == 0


@pytest.mark.parametrize(
    ("group_by", "period"),
    [
        ("task", "hour"),
        ("lane", "day"),
    ],
)
@pytest.mark.unit
def test_aggregate_entries_rejects_invalid_inputs(group_by, period):
    """
    Ensure invalid grouping or period values raise errors.

    Returns
    -------
    None
        This test asserts validation behavior.
    """
    entry = time_log.TimeEntry(
        id=1,
        uuid="u1",
        description="Move",
        project=None,
        tags=(),
        mode=None,
        start=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError):
        time_log.aggregate_entries([entry], group_by=group_by, period=period)

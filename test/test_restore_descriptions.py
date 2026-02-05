"""
Tests for restoring move descriptions from Taskwarrior history.
"""

from pathlib import Path

import pytest

import twh.restore_descriptions as restore
import twh.taskwarrior as taskwarrior


@pytest.mark.parametrize(
    ("history", "current", "expected"),
    [
        (
            "2026-01-29 Description changed from 'Old desc' to 'diff:4'.",
            "diff:4",
            "Old desc",
        ),
        (
            "2026-01-29 Description changed from 'First' to 'diff:4'.\n"
            "2026-01-30 Description changed from 'Second' to 'diff:4'.",
            "diff:4",
            "Second",
        ),
        (
            "2026-01-29 Description changed from 'Old' to 'diff:3'.",
            "diff:4",
            None,
        ),
    ],
)
@pytest.mark.unit
def test_find_previous_description(history, current, expected):
    """
    Ensure history parsing returns the latest matching description.

    Returns
    -------
    None
        This test asserts history parsing behavior.
    """
    assert restore.find_previous_description(history, current) == expected


@pytest.mark.unit
def test_collect_restore_candidates_filters_diff():
    """
    Confirm only diff-prefixed descriptions are considered for restore.

    Returns
    -------
    None
        This test asserts candidate selection logic.
    """
    tasks = [
        {"uuid": "u1", "id": 1, "description": "diff:4"},
        {"uuid": "u2", "id": 2, "description": "Normal"},
    ]

    def history_fetcher(_uuid):
        return "2026-01-29 Description changed from 'Old desc' to 'diff:4'."

    candidates = restore.collect_restore_candidates(tasks, history_fetcher)

    assert len(candidates) == 1
    assert candidates[0].uuid == "u1"
    assert candidates[0].previous_description == "Old desc"


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.unit
def test_apply_restores_respects_apply_flag(apply):
    """
    Ensure restore application is gated by the apply flag.

    Returns
    -------
    None
        This test asserts apply gating behavior.
    """
    candidates = [
        restore.RestoreCandidate(
            uuid="u1",
            move_id=1,
            current_description="diff:4",
            previous_description="Old desc",
        )
    ]
    calls = []

    def fake_runner(args, **_kwargs):
        calls.append(args)
        return 0

    restore.apply_restores(candidates, apply=apply, runner=fake_runner)

    if apply:
        assert calls == [["task", "u1", "modify", "description:Old desc"]]
    else:
        assert calls == []


@pytest.mark.unit
def test_build_rc_overrides_uses_canonical(monkeypatch, tmp_path):
    """
    Ensure rc overrides use the canonical taskrc path.

    Returns
    -------
    None
        This test asserts canonical rc overrides.
    """
    canonical = tmp_path / ".taskrc"
    canonical.write_text("", encoding="utf-8")
    monkeypatch.setattr(restore, "get_taskrc_path", lambda: canonical)

    assert restore.build_rc_overrides(None, None) == [f"rc:{canonical}"]
    assert restore.build_rc_overrides(str(canonical), "D:/data") == [
        f"rc:{canonical}",
        "rc.data.location:D:/data",
    ]


@pytest.mark.unit
def test_build_rc_overrides_rejects_noncanonical(monkeypatch, tmp_path):
    """
    Ensure non-canonical taskrc values are rejected.

    Returns
    -------
    None
        This test asserts canonical taskrc enforcement.
    """
    canonical = tmp_path / ".taskrc"
    canonical.write_text("", encoding="utf-8")
    monkeypatch.setattr(restore, "get_taskrc_path", lambda: canonical)

    with pytest.raises(RuntimeError):
        restore.build_rc_overrides("C:/taskrc", None)


@pytest.mark.unit
def test_export_tasks_uses_rc_overrides(monkeypatch):
    """
    Verify rc overrides are passed to Taskwarrior export.

    Returns
    -------
    None
        This test asserts Taskwarrior invocation.
    """
    calls = []

    def fake_run(args, capture_output=False, text=False, check=False):
        calls.append(args)
        return restore.subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")

    monkeypatch.setattr(restore.subprocess, "run", fake_run)

    monkeypatch.setattr(taskwarrior, "get_taskrc_path", lambda: Path("C:/taskrc"))

    restore.export_tasks(["description:diff:"], ["rc:/other"])

    assert calls == [["task", "rc:C:/taskrc", "description:diff:", "export"]]

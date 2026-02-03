"""
Tests for Taskwarrior config helpers.
"""

import subprocess
from pathlib import Path

import pytest

import twh.taskwarrior as taskwarrior


@pytest.mark.unit
def test_filter_modified_zero_lines():
    """
    Ensure "Modified 0 tasks." output lines are suppressed.

    Returns
    -------
    None
        This test asserts output filtering.
    """
    assert taskwarrior.filter_modified_zero_lines("Modified 0 tasks.\n") == []
    assert taskwarrior.filter_modified_zero_lines("Modified 1 task.\n") == [
        "Modified 1 task."
    ]
    assert (
        taskwarrior.filter_modified_zero_lines(
            "Project 'work' is 0% complete (1 task remaining).\n"
        )
        == []
    )
    assert taskwarrior.filter_modified_zero_lines(
        "Warning: nope\nModified 0 tasks.\nExtra"
    ) == ["Warning: nope", "Extra"]


@pytest.mark.unit
def test_missing_udas_falls_back_to_taskrc(tmp_path, monkeypatch):
    """
    Ensure UDA detection uses taskrc when task _get fails.

    Returns
    -------
    None
        This test asserts taskrc fallback behavior.
    """
    taskrc = tmp_path / ".taskrc"
    taskrc.write_text(
        "uda.dominates.type=string\nuda.dominated_by.type=string\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TASKRC", str(taskrc))

    missing = taskwarrior.missing_udas(
        ["dominates", "dominated_by"],
        get_setting=lambda _key: None,
    )

    assert missing == []


@pytest.mark.unit
def test_missing_udas_reports_missing(tmp_path, monkeypatch):
    """
    Ensure missing UDAs are reported when unset.

    Returns
    -------
    None
        This test asserts missing UDA detection.
    """
    taskrc = tmp_path / ".taskrc"
    taskrc.write_text("", encoding="utf-8")
    monkeypatch.setenv("TASKRC", str(taskrc))

    missing = taskwarrior.missing_udas(["diff"], get_setting=lambda _key: None)

    assert missing == ["diff"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["list"], ["rc.search.case.sensitive=no", "list"]),
        (
            ["rc.search.case.sensitive=yes", "list"],
            ["rc.search.case.sensitive=no", "list"],
        ),
    ],
)
@pytest.mark.unit
def test_apply_case_insensitive_overrides(args, expected):
    """
    Ensure case-insensitive overrides are prepended and replace existing values.

    Returns
    -------
    None
        This test asserts override formatting.
    """
    assert taskwarrior.apply_case_insensitive_overrides(args) == expected


@pytest.mark.unit
def test_get_tasks_from_taskwarrior_applies_case_insensitive_override(monkeypatch):
    """
    Ensure Taskwarrior export uses case-insensitive search overrides.

    Returns
    -------
    None
        This test asserts command arguments.
    """
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")

    monkeypatch.setattr(taskwarrior.subprocess, "run", fake_run)

    tasks = taskwarrior.get_tasks_from_taskwarrior(status=None)

    assert tasks == []
    assert calls == [["task", "rc.search.case.sensitive=no", "export"]]

"""
Tests for Taskwarrior config helpers.
"""

from pathlib import Path

import pytest

import twh.taskwarrior as taskwarrior


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

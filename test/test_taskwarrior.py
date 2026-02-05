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
    monkeypatch.setattr(taskwarrior, "get_taskrc_path", lambda: taskrc)
    monkeypatch.setattr(taskwarrior, "get_defined_udas", lambda *_a, **_k: set())

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
    monkeypatch.setattr(taskwarrior, "get_taskrc_path", lambda: taskrc)
    monkeypatch.setattr(taskwarrior, "get_defined_udas", lambda *_a, **_k: set())

    missing = taskwarrior.missing_udas(["diff"], get_setting=lambda _key: None)

    assert missing == ["diff"]


@pytest.mark.unit
def test_missing_udas_handles_none_get_setting(monkeypatch):
    """
    Ensure missing_udas falls back to the default getter when None is passed.

    Returns
    -------
    None
        This test asserts None get_setting values are handled.
    """
    captured = {}

    def fake_get_setting(key):
        captured["key"] = key
        return "string"

    monkeypatch.setattr(taskwarrior, "get_taskwarrior_setting", fake_get_setting)

    missing = taskwarrior.missing_udas(
        ["criticality"],
        get_setting=None,
        allow_taskrc_fallback=False,
    )

    assert missing == []
    assert captured["key"] == "uda.criticality.type"


@pytest.mark.unit
def test_missing_udas_uses_task_udas(monkeypatch):
    """
    Ensure Taskwarrior UDAs are honored when taskrc fallback is disabled.

    Returns
    -------
    None
        This test asserts task UDAs are used.
    """
    monkeypatch.setattr(taskwarrior, "get_defined_udas", lambda *_a, **_k: {"mode"})

    missing = taskwarrior.missing_udas(
        ["mode"],
        get_setting=lambda _key: None,
        allow_taskrc_fallback=False,
    )

    assert missing == []


@pytest.mark.unit
def test_missing_udas_strict_ignores_taskrc(tmp_path, monkeypatch):
    """
    Ensure strict UDA detection ignores taskrc fallback.

    Returns
    -------
    None
        This test asserts strict UDA detection.
    """
    taskrc = tmp_path / ".taskrc"
    taskrc.write_text("uda.diff.type=numeric\n", encoding="utf-8")
    monkeypatch.setattr(taskwarrior, "get_taskrc_path", lambda: taskrc)
    monkeypatch.setattr(taskwarrior, "get_defined_udas", lambda *_a, **_k: set())

    missing = taskwarrior.missing_udas(
        ["diff"],
        get_setting=lambda _key: None,
        allow_taskrc_fallback=False,
    )

    assert missing == ["diff"]


@pytest.mark.unit
def test_parse_columns_output():
    """
    Ensure Taskwarrior column output is parsed into tokens.

    Returns
    -------
    None
        This test asserts column parsing.
    """
    assert taskwarrior._parse_columns_output("id\nproject\n\nstatus\n") == [
        "id",
        "project",
        "status",
    ]


@pytest.mark.unit
def test_get_core_attribute_names_excludes_udas(monkeypatch):
    """
    Ensure core attribute detection excludes UDA columns.

    Returns
    -------
    None
        This test asserts core attribute extraction.
    """
    taskwarrior.get_core_attribute_names.cache_clear()

    def fake_runner(args, **_kwargs):
        if args == ["_columns"]:
            return subprocess.CompletedProcess(args, 0, stdout="id\nmode\nwait\n", stderr="")
        if args == ["udas"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="Name Type\nmode string\n\n1 UDAs defined\n",
                stderr="",
            )
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

    core = taskwarrior.get_core_attribute_names(
        runner=fake_runner,
        udas_runner=fake_runner,
    )

    assert "id" in core
    assert "wait" in core
    assert "mode" not in core


@pytest.mark.unit
def test_get_taskrc_path_ignores_taskrc_env(tmp_path, monkeypatch):
    """
    Ensure taskrc resolution ignores TASKRC environment overrides.

    Returns
    -------
    None
        This test asserts canonical taskrc selection.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    taskrc = fake_home / ".taskrc"
    taskrc.write_text("", encoding="utf-8")
    monkeypatch.setenv("TASKRC", str(tmp_path / "other.taskrc"))
    monkeypatch.setattr(taskwarrior.Path, "home", lambda: fake_home)

    assert taskwarrior.get_taskrc_path() == taskrc


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
def test_apply_taskrc_overrides_inserts_canonical(monkeypatch, tmp_path):
    """
    Ensure taskrc overrides always use the canonical taskrc path.

    Returns
    -------
    None
        This test asserts taskrc override normalization.
    """
    taskrc = tmp_path / ".taskrc"
    taskrc.write_text("", encoding="utf-8")
    monkeypatch.setattr(taskwarrior, "get_taskrc_path", lambda: taskrc)

    adjusted = taskwarrior.apply_taskrc_overrides(
        ["rc:/other/taskrc", "rc.search.case.sensitive=no", "list"]
    )

    assert adjusted == [f"rc:{taskrc}", "rc.search.case.sensitive=no", "list"]


@pytest.mark.unit
def test_get_tasks_from_taskwarrior_applies_case_insensitive_override(monkeypatch, tmp_path):
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

    taskrc = tmp_path / ".taskrc"
    taskrc.write_text("", encoding="utf-8")
    monkeypatch.setattr(taskwarrior, "get_taskrc_path", lambda: taskrc)
    monkeypatch.setattr(taskwarrior.subprocess, "run", fake_run)

    tasks = taskwarrior.get_tasks_from_taskwarrior(status=None)

    assert tasks == []
    assert calls == [["task", f"rc:{taskrc}", "rc.search.case.sensitive=no", "export"]]

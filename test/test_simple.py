"""
Tests for the simple report helper logic.
"""

import subprocess

import pytest
import twh


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ("id,description", "id,description.count"),
        ("description.truncated,project", "description.count,project"),
        ("id,project", "id,project"),
        ("description.count", "description.count"),
    ],
)
@pytest.mark.unit
def test_replace_description_column(columns, expected):
    """
    Ensure description columns are replaced with annotation counts.

    Parameters
    ----------
    columns : str
        Comma-separated columns list.
    expected : str
        Expected updated columns list.

    Returns
    -------
    None
        This test asserts column replacement behavior.
    """
    assert twh.replace_description_column(columns) == expected


@pytest.mark.parametrize(
    ("report_name", "output", "expected"),
    [
        (
            "next",
            "report.next.columns id,description\nreport.next.sort urgency-",
            {"columns": "id,description", "sort": "urgency-"},
        ),
        (
            "work",
            "\nreport.work.labels ID,Description\nother.setting value",
            {"labels": "ID,Description"},
        ),
    ],
)
@pytest.mark.unit
def test_parse_report_settings(report_name, output, expected):
    """
    Validate parsing of task show output for report settings.

    Parameters
    ----------
    report_name : str
        Report name to parse.
    output : str
        Output string to parse.
    expected : dict[str, str]
        Expected settings mapping.

    Returns
    -------
    None
        This test asserts report parsing behavior.
    """
    assert twh.parse_report_settings(report_name, output) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("next +work", ("next", ["+work"])),
        ("next", ("next", [])),
        (None, ("next", [])),
    ],
)
@pytest.mark.unit
def test_parse_default_command_tokens(command, expected):
    """
    Ensure default command tokens are parsed into report and filters.

    Parameters
    ----------
    command : str | None
        Default command string.
    expected : tuple[str, list[str]]
        Expected report name and filters.

    Returns
    -------
    None
        This test asserts parsing behavior.
    """
    assert twh.parse_default_command_tokens(command) == expected


@pytest.mark.unit
def test_ensure_simple_report_creates_report(monkeypatch):
    """
    Ensure the simple report is created from the base report.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching task execution.

    Returns
    -------
    None
        This test asserts configuration calls.
    """
    calls = []

    def fake_get_setting(key):
        if key == "report.simple.columns":
            return None
        return None

    def fake_run(args, capture_output=False):
        calls.append((args, capture_output))
        if args[:2] == ["show", "report.next"]:
            output = (
                "report.next.columns id,description\n"
                "report.next.labels ID,Description\n"
                "report.next.sort urgency-\n"
            )
            return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(twh, "get_taskwarrior_setting", fake_get_setting)
    monkeypatch.setattr(twh, "run_task_command", fake_run)

    assert twh.ensure_simple_report("next") is True
    assert calls == [
        (["show", "report.next"], True),
        (["config", "report.simple.columns", "id,description.count"], True),
        (["config", "report.simple.labels", "ID,Description"], True),
        (["config", "report.simple.sort", "urgency-"], True),
    ]


@pytest.mark.unit
def test_ensure_simple_report_skips_when_present(monkeypatch):
    """
    Ensure an existing simple report is not recreated.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching task execution.

    Returns
    -------
    None
        This test asserts early exit behavior.
    """
    monkeypatch.setattr(
        twh,
        "get_taskwarrior_setting",
        lambda key: "id,description.count" if key == "report.simple.columns" else None,
    )

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("run_task_command should not be called.")

    monkeypatch.setattr(twh, "run_task_command", unexpected_run)

    assert twh.ensure_simple_report("next") is True


@pytest.mark.unit
def test_run_simple_report_invokes_task(monkeypatch):
    """
    Verify simple report runs Taskwarrior with filters.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching task execution.

    Returns
    -------
    None
        This test asserts the task invocation.
    """
    calls = []

    monkeypatch.setattr(twh, "get_default_command_tokens", lambda: ("next", ["+work"]))
    monkeypatch.setattr(twh, "ensure_simple_report", lambda _base: True)
    monkeypatch.setattr(twh, "should_disable_simple_pager", lambda: True)

    def fake_run(args, capture_output=False):
        calls.append((args, capture_output))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(twh, "run_task_command", fake_run)

    assert twh.run_simple_report(["project:work"]) == 0
    assert calls == [(["rc.pager=cat", "+work", "project:work", "simple"], False)]


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("1", False),
        ("true", False),
        ("yes", False),
        ("on", False),
        ("", True),
        (None, True),
    ],
)
@pytest.mark.unit
def test_should_disable_simple_pager(env_value, expected, monkeypatch):
    """
    Confirm pager disabling respects environment overrides.

    Parameters
    ----------
    env_value : str | None
        Environment variable value to test.
    expected : bool
        Expected disable result.
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching environment.

    Returns
    -------
    None
        This test asserts pager logic.
    """
    monkeypatch.setattr(twh, "is_wsl", lambda: True)
    if env_value is None:
        monkeypatch.delenv("TWH_SIMPLE_PAGER", raising=False)
    else:
        monkeypatch.setenv("TWH_SIMPLE_PAGER", env_value)

    assert twh.should_disable_simple_pager() is expected

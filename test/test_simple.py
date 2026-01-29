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
    ("value", "expected"),
    [
        ("status:pending -WAITING limit:page", "status:pending -WAITING"),
        ("limit:page", ""),
        ("status:pending", "status:pending"),
    ],
)
@pytest.mark.unit
def test_strip_limit_page(value, expected):
    """
    Ensure limit:page is removed from report filters.

    Parameters
    ----------
    value : str
        Filter string to update.
    expected : str
        Expected filter output.

    Returns
    -------
    None
        This test asserts filter cleanup behavior.
    """
    assert twh.strip_limit_page(value) == expected


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["status:pending", "limit:page"], ["status:pending"]),
        (["limit:page"], []),
        (["status:pending"], ["status:pending"]),
    ],
)
@pytest.mark.unit
def test_strip_limit_page_tokens(tokens, expected):
    """
    Ensure limit:page tokens are removed from lists.

    Parameters
    ----------
    tokens : list[str]
        Tokens to filter.
    expected : list[str]
        Expected filtered tokens.

    Returns
    -------
    None
        This test asserts token filtering.
    """
    assert twh.strip_limit_page_tokens(tokens) == expected


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

    def fake_get_setting(_key):
        return None

    def fake_run(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        if args[-2:] == ["show", "report.next"]:
            output = (
                "report.next.columns id,description\n"
                "report.next.labels ID,Description\n"
                "report.next.filter status:pending -WAITING limit:page\n"
                "report.next.sort urgency-\n"
            )
            return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(twh, "get_taskwarrior_setting_simple", fake_get_setting)
    monkeypatch.setattr(twh, "run_task_command", fake_run)
    monkeypatch.setattr(twh, "should_disable_simple_pager", lambda: True)

    assert twh.ensure_simple_report("next") is True
    assert calls == [
        (["rc.pager=cat", "rc.confirmation=off", "rc.hooks=off", "show", "report.next"], True),
        (
            ["rc.pager=cat", "rc.confirmation=off", "rc.hooks=off",
             "config", "report.simple.columns", "id,description.count"],
            True,
        ),
        (
            ["rc.pager=cat", "rc.confirmation=off", "rc.hooks=off",
             "config", "report.simple.labels", "ID,Description"],
            True,
        ),
        (
            ["rc.pager=cat", "rc.confirmation=off", "rc.hooks=off",
             "config", "report.simple.filter", "status:pending -WAITING"],
            True,
        ),
        (
            ["rc.pager=cat", "rc.confirmation=off", "rc.hooks=off",
             "config", "report.simple.sort", "urgency-"],
            True,
        ),
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
        "get_taskwarrior_setting_simple",
        lambda key: "id,description.count" if key == "report.simple.columns" else None,
    )
    monkeypatch.setattr(twh, "should_disable_simple_pager", lambda: False)

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("run_task_command should not be called.")

    monkeypatch.setattr(twh, "run_task_command", unexpected_run)

    assert twh.ensure_simple_report("next") is True


@pytest.mark.unit
def test_ensure_simple_report_updates_filter(monkeypatch):
    """
    Ensure existing simple report filter drops limit:page when needed.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching task execution.

    Returns
    -------
    None
        This test asserts filter updates for existing reports.
    """
    calls = []

    def fake_get_setting(key):
        if key == "report.simple.columns":
            return "id,description.count"
        if key == "report.simple.filter":
            return "status:pending limit:page"
        return None

    def fake_run(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(twh, "get_taskwarrior_setting_simple", fake_get_setting)
    monkeypatch.setattr(twh, "run_task_command", fake_run)
    monkeypatch.setattr(twh, "should_disable_simple_pager", lambda: True)

    assert twh.ensure_simple_report("next") is True
    assert calls == [
        (
            ["rc.pager=cat", "rc.confirmation=off", "rc.hooks=off",
             "config", "report.simple.filter", "status:pending"],
            True,
        ),
    ]


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

    monkeypatch.setattr(twh, "should_disable_simple_pager", lambda: True)
    monkeypatch.setattr(twh, "get_taskwarrior_setting_simple", lambda _key: None)

    def fake_run(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(twh, "run_task_command", fake_run)

    assert twh.run_simple_report(["project:work", "limit:page"]) == 0
    assert calls == [
        (
            [
                "rc.pager=cat",
                "rc.confirmation=off",
                "rc.hooks=off",
                "project:work",
                "simple",
            ],
            False,
        ),
    ]


@pytest.mark.unit
def test_run_simple_report_creates_missing_report(monkeypatch):
    """
    Ensure missing simple report is created and retried.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching task execution.

    Returns
    -------
    None
        This test asserts report creation behavior.
    """
    calls = []
    ensure_calls = []

    def fake_run(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        if len(calls) == 1:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(twh, "run_task_command", fake_run)
    monkeypatch.setattr(twh, "should_disable_simple_pager", lambda: False)
    monkeypatch.setattr(twh, "get_taskwarrior_setting_simple", lambda _key: None)
    monkeypatch.setattr(twh, "get_default_report_name", lambda: "next")
    monkeypatch.setattr(
        twh,
        "ensure_simple_report",
        lambda report: ensure_calls.append(report) or True,
    )

    assert twh.run_simple_report([]) == 0
    assert ensure_calls == ["next"]
    assert calls == [
        (["simple"], False),
        (["simple"], False),
    ]


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


@pytest.mark.parametrize(
    ("disable", "expected"),
    [
        (True, ["rc.pager=cat", "rc.confirmation=off", "rc.hooks=off"]),
        (False, []),
    ],
)
@pytest.mark.unit
def test_simple_task_overrides(disable, expected, monkeypatch):
    """
    Ensure simple task overrides are applied when needed.

    Parameters
    ----------
    disable : bool
        Whether pager disabling is enabled.
    expected : list[str]
        Expected override tokens.
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching behavior.

    Returns
    -------
    None
        This test asserts override behavior.
    """
    monkeypatch.setattr(twh, "should_disable_simple_pager", lambda: disable)
    assert twh.simple_task_overrides() == expected

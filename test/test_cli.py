"""
Tests for the twh CLI delegation behavior.
"""

import doctest
import subprocess
import sys

import pytest
import twh


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], True),
        (["project:work"], True),
        (["add", "Next task"], True),
        (["list"], False),
        (["reverse"], False),
        (["tree"], False),
        (["graph"], False),
        (["--help"], False),
    ],
)
@pytest.mark.unit
def test_should_delegate_to_task(argv, expected):
    """
    Verify when commands should be delegated to Taskwarrior.

    Parameters
    ----------
    argv : list[str]
        Argument list excluding the program name.
    expected : bool
        Expected delegation decision.

    Returns
    -------
    None
        This test asserts on delegation behavior.
    """
    assert twh.should_delegate_to_task(argv) is expected


@pytest.mark.parametrize(
    ("argv", "expected_args"),
    [
        (["twh"], ["task"]),
        (["twh", "project:work"], ["task", "project:work"]),
        (["twh", "add", "Next task"], ["task", "add", "Next task"]),
        (["twh", "21", "modify", "project:personal", "depends:20"],
         ["task", "21", "modify", "project:personal", "depends:20"]),
    ],
)
@pytest.mark.unit
def test_main_delegates_to_task(monkeypatch, argv, expected_args):
    """
    Ensure main forwards unknown commands to Taskwarrior.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching subprocess and argv.
    argv : list[str]
        Full argv including the program name.
    expected_args : list[str]
        Expected Taskwarrior invocation.

    Returns
    -------
    None
        This test asserts on subprocess delegation and exit code.
    """
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(twh.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        twh.main()

    assert excinfo.value.code == 0
    assert calls == [expected_args]


@pytest.mark.parametrize(
    "argv",
    [
        ["twh", "list"],
        ["twh", "reverse"],
        ["twh", "tree"],
        ["twh", "graph"],
        ["twh", "--help"],
    ],
)
@pytest.mark.unit
def test_main_uses_twh_commands(monkeypatch, argv):
    """
    Confirm twh commands invoke the Typer app directly.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching the Typer app and subprocess.
    argv : list[str]
        Full argv including the program name.

    Returns
    -------
    None
        This test asserts on app invocation and no delegation.
    """
    called = {"app": 0, "task": 0}

    def fake_app():
        called["app"] += 1

    def fake_run(args, **kwargs):
        called["task"] += 1
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(twh, "app", fake_app)
    monkeypatch.setattr(twh.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", argv)

    twh.main()

    assert called["app"] == 1
    assert called["task"] == 0


@pytest.mark.unit
def test_doctest_examples():
    """
    Run doctest examples embedded in twh docstrings.

    Returns
    -------
    None
        This test asserts that doctest examples succeed.
    """
    results = doctest.testmod(twh)
    assert results.failed == 0

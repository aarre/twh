"""
Tests for case-insensitive Taskwarrior invocations.
"""

import subprocess

import pytest

import twh
import twh.option_value as option_value
import twh.review as review
import twh.time_log as time_log


@pytest.mark.parametrize(
    "module",
    [
        twh,
        review,
        option_value,
        time_log,
    ],
)
@pytest.mark.unit
def test_run_task_command_applies_case_insensitive_override(module, monkeypatch):
    """
    Ensure task commands include the case-insensitive override.

    Returns
    -------
    None
        This test asserts task command arguments.
    """
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.run_task_command(["list"], capture_output=True)

    assert calls == [["task", "rc.search.case.sensitive=no", "list"]]


@pytest.mark.unit
def test_exec_task_command_applies_case_insensitive_override(monkeypatch):
    """
    Ensure delegated task execution uses case-insensitive overrides.

    Returns
    -------
    None
        This test asserts exec arguments.
    """
    calls = []

    def fake_execvp(cmd, args):
        calls.append((cmd, args))

    monkeypatch.setattr(twh.os, "execvp", fake_execvp)

    exit_code = twh.exec_task_command(["Task"])

    assert exit_code == 1
    assert calls == [
        ("task", ["task", "rc.search.case.sensitive=no", "Task"]),
    ]

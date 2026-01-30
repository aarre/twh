"""
Tests for the blocks-to-depends translation logic.
"""

import subprocess

import pytest
import twh


@pytest.mark.parametrize(
    ("args", "expected_cleaned", "expected_blocks"),
    [
        (["blocks:32"], [], ["32"]),
        (["blocks", "32"], [], ["32"]),
        (["blocks:32,33"], [], ["32", "33"]),
        (["project:work", "blocks:32", "+tag"], ["project:work", "+tag"], ["32"]),
        (["blocks:32", "--", "blocks:99"], ["--", "blocks:99"], ["32"]),
        (["blocks"], ["blocks"], []),
        (["blocks:"], ["blocks:"], []),
    ],
)
@pytest.mark.unit
def test_extract_blocks_tokens(args, expected_cleaned, expected_blocks):
    """
    Ensure blocks tokens are parsed and removed correctly.

    Parameters
    ----------
    args : list[str]
        Raw argument list to parse.
    expected_cleaned : list[str]
        Arguments after removing blocks tokens.
    expected_blocks : list[str]
        Parsed block targets.

    Returns
    -------
    None
        This test asserts on blocks parsing.
    """
    cleaned, blocks = twh.extract_blocks_tokens(args)
    assert cleaned == expected_cleaned
    assert blocks == expected_blocks


@pytest.mark.unit
def test_apply_blocks_add(monkeypatch):
    """
    Verify add + blocks performs add then modifies blocked tasks.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching task execution.

    Returns
    -------
    None
        This test asserts on add behavior.
    """
    calls = []

    def fake_run_task(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        if args[0] == "add":
            return subprocess.CompletedProcess(args, 0, stdout="Created task 45.\n", stderr="")
        if args[0] == "45" and args[1] == "export":
            payload = '[{"uuid":"uuid-45","id":45}]'
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")
        if args[0] == "32" and args[1] == "export":
            payload = '[{"uuid":"uuid-32","id":32}]'
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(twh, "run_task_command", fake_run_task)

    exit_code = twh.apply_blocks_relationship(["add", "New task", "blocks:32"])

    assert exit_code == 0
    assert calls == [
        (["add", "New task"], True),
        (["45", "export"], True),
        (["32", "export"], True),
        (["uuid-32", "modify", "depends:uuid-45"], True),
    ]


@pytest.mark.unit
def test_apply_blocks_modify(monkeypatch):
    """
    Verify modify blocks applies depends to blocked tasks.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching task execution.

    Returns
    -------
    None
        This test asserts on modify behavior.
    """
    calls = []

    def fake_run_task(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        if args[-1] == "export":
            if args[0] == "31":
                payload = '[{"uuid":"uuid-1","id":31}]'
            else:
                payload = '[{"uuid":"uuid-32","id":32,"depends":"uuid-9"}]'
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(twh, "run_task_command", fake_run_task)

    exit_code = twh.apply_blocks_relationship(["31", "modify", "blocks", "32"])

    assert exit_code == 0
    assert calls == [
        (["31", "export"], True),
        (["32", "export"], True),
        (["uuid-32", "modify", "depends:uuid-9,uuid-1"], True),
    ]


@pytest.mark.unit
def test_apply_blocks_modify_with_other_changes(monkeypatch):
    """
    Ensure modify still applies other changes and blocks.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching task execution.

    Returns
    -------
    None
        This test asserts on mixed modifications.
    """
    calls = []

    def fake_run_task(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        if args[-1] == "export":
            if args[0] == "31":
                payload = '[{"uuid":"uuid-31","id":31}]'
            else:
                payload = '[{"uuid":"uuid-32","id":32,"depends":"uuid-9"}]'
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(twh, "run_task_command", fake_run_task)

    exit_code = twh.apply_blocks_relationship(
        ["31", "modify", "project:work", "blocks:32"]
    )

    assert exit_code == 0
    assert calls == [
        (["31", "export"], True),
        (["31", "modify", "project:work"], False),
        (["32", "export"], True),
        (["uuid-32", "modify", "depends:uuid-9,uuid-31"], True),
    ]


@pytest.mark.unit
def test_apply_blocks_relationship_execs_without_blocks(monkeypatch):
    """
    Ensure exec fast path is used when no blocks are present.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching task execution.

    Returns
    -------
    None
        This test asserts exec usage for non-blocks commands.
    """
    calls = []

    def fake_exec(args):
        calls.append(args)
        return 0

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("run_task_command should not be used without blocks.")

    monkeypatch.setattr(twh, "run_task_command", unexpected_run)

    exit_code = twh.apply_blocks_relationship(["list"], exec_task=fake_exec)

    assert exit_code == 0
    assert calls == [["list"]]

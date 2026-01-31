"""
Tests for readline-style line editing setup.
"""

import importlib
import sys

import pytest

import twh


@pytest.mark.parametrize(
    ("available_modules", "expected", "expected_calls"),
    [
        ({"readline"}, True, ["readline"]),
        ({"pyreadline3"}, True, ["readline", "pyreadline3"]),
        (set(), False, ["readline", "pyreadline3", "pyreadline"]),
    ],
)
@pytest.mark.unit
def test_enable_line_editing_imports_readline(
    monkeypatch,
    available_modules,
    expected,
    expected_calls,
):
    """
    Confirm readline imports are attempted in the expected order.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch helper.
    available_modules : set[str]
        Modules that should import successfully.
    expected : bool
        Expected result from enable_line_editing.
    expected_calls : list[str]
        Expected import order.

    Returns
    -------
    None
        This test asserts on readline import behavior.
    """
    calls = []

    def fake_import(name):
        calls.append(name)
        if name in available_modules:
            return object()
        raise ImportError(f"Missing {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    result = twh.enable_line_editing(is_interactive=True)

    assert result is expected
    assert calls == expected_calls


@pytest.mark.unit
def test_enable_line_editing_skips_non_interactive(monkeypatch):
    """
    Ensure readline imports are skipped when stdin is not a TTY.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch helper.

    Returns
    -------
    None
        This test asserts on non-interactive behavior.
    """
    def fail_import(_name):
        raise AssertionError("readline import should not be attempted")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    assert twh.enable_line_editing(is_interactive=False) is False


@pytest.mark.unit
def test_main_enables_line_editing(monkeypatch):
    """
    Verify main enables line editing before dispatching commands.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch helper.

    Returns
    -------
    None
        This test asserts that enable_line_editing is invoked.
    """
    called = {"value": False}

    def fake_enable_line_editing():
        called["value"] = True
        return True

    def fake_apply_blocks_relationship(*_args, **_kwargs):
        return 0

    monkeypatch.setattr(twh, "enable_line_editing", fake_enable_line_editing)
    monkeypatch.setattr(twh, "apply_blocks_relationship", fake_apply_blocks_relationship)
    monkeypatch.setattr(sys, "argv", ["twh", "add", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        twh.main()

    assert excinfo.value.code == 0
    assert called["value"] is True

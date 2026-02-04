"""
Tests for mode registry helpers.
"""

import doctest

import pytest

import twh.modes as modes


@pytest.mark.unit
def test_load_known_modes_defaults(monkeypatch, tmp_path):
    """
    Ensure defaults load when no modes file exists.

    Returns
    -------
    None
        This test asserts default mode loading.
    """
    monkeypatch.setenv(modes.MODE_ENV_VAR, str(tmp_path / "modes.json"))

    known = modes.load_known_modes()

    assert known == sorted(modes.DEFAULT_MODES)


@pytest.mark.unit
def test_register_mode_persists_and_orders(monkeypatch, tmp_path):
    """
    Ensure registering a mode persists and keeps recency order.

    Returns
    -------
    None
        This test asserts mode registration.
    """
    path = tmp_path / "modes.json"
    monkeypatch.setenv(modes.MODE_ENV_VAR, str(path))

    updated = modes.register_mode("wait")
    assert updated == sorted(updated)
    assert "wait" in updated
    assert path.exists()

    updated_again = modes.register_mode("WAIT")
    assert updated_again == sorted(updated_again)
    assert updated_again.count("wait") == 1


@pytest.mark.unit
def test_format_mode_prompt_includes_recent_mode():
    """
    Ensure the prompt examples include the most recent mode.

    Returns
    -------
    None
        This test asserts prompt formatting.
    """
    prompt = modes.format_mode_prompt(["wait", "analysis", "research"])

    assert "analysis/research/wait" in prompt


@pytest.mark.parametrize(
    ("prefix", "known", "expected"),
    [
        ("wr", ["writing", "work"], "writing"),
        ("W", ["writing", "work"], "writing"),
        ("x", ["writing"], None),
    ],
)
@pytest.mark.unit
def test_best_mode_completion(prefix, known, expected):
    """
    Ensure best-mode completion respects ordering and prefix matches.

    Parameters
    ----------
    prefix : str
        Input prefix.
    known : list[str]
        Known modes.
    expected : str | None
        Expected completion.

    Returns
    -------
    None
        This test asserts completion behavior.
    """
    assert modes.best_mode_completion(prefix, known) == expected


@pytest.mark.unit
def test_modes_doctest_examples(monkeypatch, tmp_path):
    """
    Run doctest examples embedded in modes helpers.

    Returns
    -------
    None
        This test asserts doctest coverage for modes helpers.
    """
    monkeypatch.setenv(modes.MODE_ENV_VAR, str(tmp_path / "modes.json"))
    results = doctest.testmod(modes)
    assert results.failed == 0


@pytest.mark.parametrize(
    ("value", "reserved", "expected"),
    [
        ("wait", ["wait", "pending"], True),
        ("pending", ["wait", "pending"], True),
        ("analysis", ["wait", "pending"], False),
    ],
)
@pytest.mark.unit
def test_is_reserved_mode_value(value, reserved, expected):
    """
    Ensure reserved mode values are detected.

    Parameters
    ----------
    value : str
        Mode value to check.
    reserved : list[str]
        Reserved values to compare.
    expected : bool
        Expected reservation status.

    Returns
    -------
    None
        This test asserts reserved detection.
    """
    assert modes.is_reserved_mode_value(value, reserved=reserved) is expected


@pytest.mark.parametrize(
    ("raw_values", "mode_value", "expected_updated"),
    [
        (None, "wait", False),
        ("analysis,writing", "analysis", False),
        ("analysis,writing", "wait", True),
    ],
)
@pytest.mark.unit
def test_ensure_taskwarrior_mode_value_updates(raw_values, mode_value, expected_updated):
    """
    Ensure Taskwarrior mode values update when needed.

    Parameters
    ----------
    raw_values : str | None
        Existing uda.mode.values string.
    mode_value : str
        Mode value to ensure.
    expected_updated : bool
        Whether an update should occur.

    Returns
    -------
    None
        This test asserts Taskwarrior config updates.
    """
    calls = []

    def fake_get_setting(_key):
        return raw_values

    def fake_runner(args, **_kwargs):
        calls.append(args)
        return modes.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    updated = modes.ensure_taskwarrior_mode_value(
        mode_value,
        get_setting=fake_get_setting,
        runner=fake_runner,
    )

    assert updated is expected_updated
    if expected_updated:
        assert calls
        assert calls[0][0:2] == ["config", "uda.mode.values"]
    else:
        assert calls == []

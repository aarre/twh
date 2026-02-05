"""
Tests for criticality ranking and updates.
"""

import doctest

import pytest

import twh.criticality as criticality
import twh.review as review


def make_move(
    uuid: str,
    move_id: int | None = None,
    description: str | None = None,
    criticality_value: float | None = None,
):
    return review.ReviewTask(
        uuid=uuid,
        id=move_id,
        description=description or f"Move {uuid}",
        project=None,
        depends=[],
        imp=None,
        urg=None,
        opt=None,
        diff=None,
        criticality=criticality_value,
        mode=None,
        dominates=[],
        raw={},
    )


@pytest.mark.parametrize(
    ("choice", "expected_relation"),
    [
        (criticality.CriticalityChoice.LEFT, "more_critical"),
        (criticality.CriticalityChoice.RIGHT, "less_critical"),
        (criticality.CriticalityChoice.TIE, "tie"),
    ],
)
@pytest.mark.unit
def test_record_criticality_comparison_persists(
    tmp_path,
    monkeypatch,
    choice,
    expected_relation,
):
    """
    Ensure comparison selections are saved and reapplied.

    Returns
    -------
    None
        This test asserts saved comparison reuse.
    """
    store_path = tmp_path / "criticality.json"
    monkeypatch.setenv(criticality.CRITICALITY_PATH_ENV, str(store_path))
    move_a = make_move("a")
    move_b = make_move("b")

    criticality.record_criticality_comparison("a", "b", choice)
    state = criticality.build_criticality_state([move_a, move_b])

    assert store_path.exists()
    assert criticality.criticality_relation(state, "a", "b") == expected_relation


@pytest.mark.unit
def test_ensure_criticality_uda_uses_default_setting_lookup(monkeypatch):
    """
    Ensure the criticality UDA check supplies a default setting lookup.

    Returns
    -------
    None
        This test asserts the missing UDA check is configured.
    """
    captured = {}

    def fake_missing_udas(fields, get_setting=None, allow_taskrc_fallback=False):
        captured["fields"] = fields
        captured["get_setting"] = get_setting
        return []

    monkeypatch.setattr(criticality, "missing_udas", fake_missing_udas)

    criticality.ensure_criticality_uda()

    assert captured["fields"] == ["criticality"]
    assert callable(captured["get_setting"])


@pytest.mark.unit
def test_ensure_criticality_uda_respects_custom_setting_lookup(monkeypatch):
    """
    Ensure the criticality UDA check uses the provided getter.

    Returns
    -------
    None
        This test asserts custom getters are honored.
    """
    sentinel = object()
    captured = {}

    def fake_missing_udas(fields, get_setting=None, allow_taskrc_fallback=False):
        captured["get_setting"] = get_setting
        return []

    def custom_get_setting(_key):
        return sentinel

    monkeypatch.setattr(criticality, "missing_udas", fake_missing_udas)

    criticality.ensure_criticality_uda(get_setting=custom_get_setting)

    assert captured["get_setting"] is custom_get_setting


@pytest.mark.unit
def test_sort_into_tiers_respects_existing_values_without_prompt():
    """
    Ensure stored criticality values avoid prompts.

    Returns
    -------
    None
        This test asserts prompt avoidance for known values.
    """
    move_a = make_move("a", move_id=1, criticality_value=9.0)
    move_b = make_move("b", move_id=2, criticality_value=3.0)
    state = criticality.build_criticality_state([move_a, move_b])

    def chooser(_left, _right):
        raise AssertionError("Chooser should not be called for known values.")

    tiers = criticality.sort_into_tiers([move_a, move_b], state, chooser)

    assert [[move.uuid for move in tier] for tier in tiers] == [["a"], ["b"]]


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("A", criticality.CriticalityChoice.LEFT),
        ("b", criticality.CriticalityChoice.RIGHT),
        ("C", criticality.CriticalityChoice.TIE),
    ],
)
@pytest.mark.unit
def test_prompt_criticality_choice_uses_lettered_labels(
    capsys,
    input_value,
    expected,
):
    """
    Ensure criticality prompts use A/B/C labels and accept letter input.

    Returns
    -------
    None
        This test asserts prompt formatting and selection parsing.
    """
    move_a = make_move("a", move_id=3, description="Pay taxes")
    move_b = make_move("b", move_id=4, description="Write report")
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return input_value

    choice = criticality.prompt_criticality_choice(
        move_a,
        move_b,
        input_func=fake_input,
    )

    assert choice == expected
    assert prompts == ["Selection (A/B/C): "]
    captured = capsys.readouterr()
    assert "[A] Move ID 3: Pay taxes" in captured.out
    assert "[B] Move ID 4: Write report" in captured.out
    assert (
        "Time-criticality measures how fast the value of a move decays if you wait."
        in captured.out
    )
    assert "Which move becomes pointless first if ignored?" in captured.out
    assert "[A] Move A, [B] Move B, [C] Tie" in captured.out


@pytest.mark.unit
def test_build_criticality_updates_assigns_tier_values():
    """
    Ensure tier order maps to descending criticality values.

    Returns
    -------
    None
        This test asserts criticality value assignment.
    """
    top = make_move("a", move_id=1)
    mid = make_move("b", move_id=2)
    low = make_move("c", move_id=3)

    updates = criticality.build_criticality_updates([[top], [mid], [low]])

    assert updates["a"] == pytest.approx(10.0)
    assert updates["b"] == pytest.approx(5.0)
    assert updates["c"] == pytest.approx(0.0)


@pytest.mark.unit
def test_make_progress_chooser_reports_remaining(capsys):
    """
    Ensure progress messaging appears during criticality prompts.

    Returns
    -------
    None
        This test asserts progress output formatting.
    """
    move_a = make_move("a", move_id=1)
    move_b = make_move("b", move_id=2)
    state = criticality.build_criticality_state([move_a, move_b])
    chooser = criticality.make_progress_chooser(
        [move_a, move_b],
        state,
        input_func=lambda _prompt: "A",
    )

    _choice = chooser(move_a, move_b)

    captured = capsys.readouterr()
    assert "Criticality progress:" in captured.out


@pytest.mark.unit
def test_criticality_doctest_examples():
    """
    Run doctest examples embedded in criticality docstrings.

    Returns
    -------
    None
        This test asserts doctest coverage for criticality helpers.
    """
    results = doctest.testmod(criticality)
    assert results.failed == 0

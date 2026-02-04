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

"""
Tests for the twh diagnose workflow helpers.
"""

from __future__ import annotations

import doctest
from typing import Callable

import pytest

from twh import diagnose as diagnose_module
from twh.review import ReviewTask


def make_task(uuid: str, task_id: int | None, **raw) -> ReviewTask:
    """
    Build a ReviewTask with extra raw fields.

    Parameters
    ----------
    uuid : str
        Move UUID.
    task_id : int | None
        Move ID.
    raw : dict[str, object]
        Extra raw fields.

    Returns
    -------
    ReviewTask
        Parsed move instance.
    """
    payload = {"uuid": uuid, "id": task_id, "description": f"Move {uuid}"}
    payload.update(raw)
    return ReviewTask.from_json(payload)


def make_inputs(values: list[str]) -> Callable[[str], str]:
    """
    Build an input function from a list of responses.

    Parameters
    ----------
    values : list[str]
        Responses to return in order.

    Returns
    -------
    Callable[[str], str]
        Input function that returns the queued values.
    """
    iterator = iter(values)

    def _input(_prompt: str) -> str:
        return next(iterator)

    return _input


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("3", 3.0),
        ("4.25", 4.25),
        (7, 7.0),
        ("nope", None),
    ],
)
@pytest.mark.unit
def test_parse_numeric_value(value, expected):
    """
    Verify numeric parsing for dimension values.

    Parameters
    ----------
    value : object
        Raw value to parse.
    expected : float | None
        Expected float result.

    Returns
    -------
    None
        This test asserts on numeric parsing behavior.
    """
    assert diagnose_module.parse_numeric_value(value) == expected


@pytest.mark.parametrize(
    ("selector", "expected_uuid"),
    [
        ("2", "b-uuid"),
        ("A-UUID", "a-uuid"),
    ],
)
@pytest.mark.unit
def test_select_move_by_selector(selector, expected_uuid):
    """
    Ensure selector matching works for IDs and UUID prefixes.

    Parameters
    ----------
    selector : str
        Selector input.
    expected_uuid : str
        Expected UUID of the match.

    Returns
    -------
    None
        This test asserts on selector matching.
    """
    pending = [
        make_task("a-uuid", 1),
        make_task("b-uuid", 2),
    ]

    selected = diagnose_module.select_move_by_selector(pending, selector)

    assert selected.uuid == expected_uuid


@pytest.mark.parametrize(
    "selector",
    [
        "999",
        "missing",
    ],
)
@pytest.mark.unit
def test_select_move_by_selector_missing(selector):
    """
    Ensure missing selectors raise a ValueError.

    Parameters
    ----------
    selector : str
        Selector input that should fail.

    Returns
    -------
    None
        This test asserts on missing selector handling.
    """
    pending = [
        make_task("a-uuid", 1),
    ]

    with pytest.raises(ValueError):
        diagnose_module.select_move_by_selector(pending, selector)


@pytest.mark.unit
def test_missing_dimension_values():
    """
    Ensure missing dimensions are detected for a move.

    Returns
    -------
    None
        This test asserts on missing dimension detection.
    """
    task = make_task("u1", 1, energy="5")

    missing = diagnose_module.missing_dimension_values(task)

    assert "energy" not in missing
    assert "attention" in missing
    assert "emotion" in missing


@pytest.mark.unit
def test_prompt_dimension_updates_retries_invalid():
    """
    Confirm dimension prompts retry invalid input.

    Returns
    -------
    None
        This test asserts on numeric prompt validation.
    """
    input_func = make_inputs(["nope", "5", ""])

    updates = diagnose_module.prompt_dimension_updates(
        ["energy", "attention"],
        input_func=input_func,
    )

    assert updates == {"energy": "5"}


@pytest.mark.parametrize(
    ("dimension", "stuck_raw", "candidate_raw", "expected"),
    [
        ("energy", {"energy": "7"}, {"energy": "3"}, True),
        ("energy", {"energy": "3"}, {"energy": "7"}, False),
        ("attention", {"attention": "6"}, {"attention": "2"}, True),
        ("attention", {"interruptible": "0"}, {"interruptible": "1"}, True),
        ("emotion", {"emotion": "6"}, {"emotion": "2"}, True),
        ("emotion", {"mechanical": "0"}, {"mechanical": "1"}, True),
        ("time", {"estimate_hours": "6"}, {"estimate_hours": "2"}, True),
        ("time", {"diff": "6"}, {"diff": "2"}, True),
    ],
)
@pytest.mark.unit
def test_is_easier_in_dimension(dimension, stuck_raw, candidate_raw, expected):
    """
    Confirm dimension comparisons follow the intended rules.

    Parameters
    ----------
    dimension : str
        Dimension name.
    stuck_raw : dict[str, object]
        Raw values for the stuck move.
    candidate_raw : dict[str, object]
        Raw values for the candidate move.
    expected : bool
        Expected comparison result.

    Returns
    -------
    None
        This test asserts on dimension comparisons.
    """
    stuck = make_task("stuck", 1, **stuck_raw)
    candidate = make_task("candidate", 2, **candidate_raw)

    assert (
        diagnose_module.is_easier_in_dimension(candidate, stuck, dimension)
        is expected
    )


@pytest.mark.parametrize(
    ("lacking", "ordered_raw", "expected_uuid"),
    [
        (
            "energy",
            [
                {"uuid": "stuck", "id": 1, "energy": "7"},
                {"uuid": "hard", "id": 2, "energy": "9"},
                {"uuid": "easy", "id": 3, "energy": "3"},
            ],
            "easy",
        ),
        (
            "emotion",
            [
                {"uuid": "stuck", "id": 1, "emotion": "7"},
                {"uuid": "hard", "id": 2, "emotion": "9"},
            ],
            None,
        ),
    ],
)
@pytest.mark.unit
def test_pick_easier_move(lacking, ordered_raw, expected_uuid):
    """
    Ensure easier moves are selected in ondeck order.

    Parameters
    ----------
    lacking : str
        Lacking dimension.
    ordered_raw : list[dict[str, object]]
        Ordered raw move payloads.
    expected_uuid : str | None
        Expected selected UUID.

    Returns
    -------
    None
        This test asserts on easier move selection.
    """
    ordered = [
        make_task(
            item["uuid"],
            item["id"],
            **{k: v for k, v in item.items() if k not in {"uuid", "id"}},
        )
        for item in ordered_raw
    ]
    stuck = ordered[0]

    selected = diagnose_module.pick_easier_move(ordered, stuck, lacking)

    if expected_uuid is None:
        assert selected is None
    else:
        assert selected is not None
        assert selected.uuid == expected_uuid


@pytest.mark.unit
def test_create_helper_move_uses_add_wizard(monkeypatch, capsys):
    """
    Ensure helper moves are created through the add wizard.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching add runner.
    capsys : pytest.CaptureFixture[str]
        Fixture to capture stdout.

    Returns
    -------
    None
        This test asserts on helper move creation behavior.
    """
    stuck = make_task("stuck", 7, description="Write notes")
    calls = []

    def fake_run_interactive_add(args, input_func=None):
        calls.append(args)
        return 0

    monkeypatch.setattr(diagnose_module, "run_interactive_add", fake_run_interactive_add)

    exit_code = diagnose_module.create_helper_move(
        stuck,
        "activation",
        input_func=make_inputs([""]),
    )

    assert exit_code == 0
    assert calls == [[diagnose_module.HELPER_TEMPLATES["activation"].format(desc="Write notes")]]
    captured = capsys.readouterr().out
    assert "Blocks" in captured


@pytest.mark.unit
def test_format_missing_uda_instructions():
    """
    Verify missing UDA instructions include config lines.

    Returns
    -------
    None
        This test asserts on UDA instruction formatting.
    """
    text = diagnose_module.format_missing_uda_instructions(["energy", "mechanical"])

    assert "Missing Taskwarrior UDA(s): energy, mechanical." in text
    assert "Add the following to ~/.taskrc" in text
    assert "uda.energy.type=numeric" in text
    assert "uda.mechanical.type=numeric" in text


@pytest.mark.unit
def test_doctest_examples():
    """
    Run doctest examples for the diagnose module.

    Returns
    -------
    None
        This test asserts doctest examples succeed.
    """
    results = doctest.testmod(diagnose_module)
    assert results.failed == 0

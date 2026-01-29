"""
Tests for the review command logic.
"""

import doctest

import pytest

import twh.review as review


@pytest.mark.parametrize(
    (
        "payload",
        "expected_depends",
        "expected_dominates",
        "expected_imp",
        "expected_mode",
    ),
    [
        (
            {
                "uuid": "u1",
                "id": 1,
                "description": "  Task A  ",
                "depends": "a,b",
                "dominates": "c, d",
                "imp": "5",
                "mode": " editorial ",
            },
            ["a", "b"],
            ["c", "d"],
            5,
            "editorial",
        ),
        (
            {
                "uuid": "u2",
                "id": 2,
                "description": "Task B",
                "depends": ["x", "y"],
                "dominates": ["z"],
                "imp": None,
                "mode": None,
            },
            ["x", "y"],
            ["z"],
            None,
            None,
        ),
        (
            {
                "uuid": "u3",
                "id": 3,
                "description": "Task C",
                "depends": None,
                "dominates": None,
                "imp": "bad",
                "mode": "",
            },
            [],
            [],
            None,
            None,
        ),
    ],
)
@pytest.mark.unit
def test_review_task_from_json_parses_fields(
    payload,
    expected_depends,
    expected_dominates,
    expected_imp,
    expected_mode,
):
    """
    Ensure review tasks parse depends, dominates, and metadata fields.

    Parameters
    ----------
    payload : dict[str, object]
        Raw Taskwarrior export payload.
    expected_depends : list[str]
        Expected depends list.
    expected_dominates : list[str]
        Expected dominates list.
    expected_imp : int | None
        Expected importance value.
    expected_mode : str | None
        Expected normalized mode.

    Returns
    -------
    None
        This test asserts on field parsing.
    """
    task = review.ReviewTask.from_json(payload)
    assert task.depends == expected_depends
    assert task.dominates == expected_dominates
    assert task.imp == expected_imp
    assert task.mode == expected_mode
    assert task.description == payload["description"].strip()


@pytest.mark.unit
def test_ready_tasks_excludes_pending_dependencies():
    """
    Verify ready tasks exclude items blocked by pending dependencies.

    Returns
    -------
    None
        This test asserts readiness logic.
    """
    tasks = [
        review.ReviewTask(
            uuid="a",
            id=1,
            description="Task A",
            project=None,
            depends=["b"],
            imp=None,
            urg=None,
            opt=None,
            mode=None,
            dominates=[],
            raw={},
        ),
        review.ReviewTask(
            uuid="b",
            id=2,
            description="Task B",
            project=None,
            depends=[],
            imp=None,
            urg=None,
            opt=None,
            mode=None,
            dominates=[],
            raw={},
        ),
        review.ReviewTask(
            uuid="c",
            id=3,
            description="Task C",
            project=None,
            depends=["missing"],
            imp=None,
            urg=None,
            opt=None,
            mode=None,
            dominates=[],
            raw={},
        ),
    ]

    ready = review.ready_tasks(tasks)

    assert {task.uuid for task in ready} == {"b", "c"}


@pytest.mark.unit
def test_collect_missing_metadata_orders_ready_first():
    """
    Confirm missing metadata list sorts ready tasks first.

    Returns
    -------
    None
        This test asserts ordering of review items.
    """
    tasks = [
        review.ReviewTask(
            uuid="a",
            id=1,
            description="Task A",
            project="alpha",
            depends=[],
            imp=None,
            urg=3,
            opt=4,
            mode="editorial",
            dominates=[],
            raw={},
        ),
        review.ReviewTask(
            uuid="b",
            id=2,
            description="Task B",
            project="beta",
            depends=["a"],
            imp=2,
            urg=None,
            opt=4,
            mode="editorial",
            dominates=[],
            raw={},
        ),
        review.ReviewTask(
            uuid="c",
            id=3,
            description="Task C",
            project="beta",
            depends=[],
            imp=1,
            urg=2,
            opt=None,
            mode=None,
            dominates=[],
            raw={},
        ),
    ]

    ready = review.ready_tasks(tasks)
    missing = review.collect_missing_metadata(tasks, ready)

    assert [item.task.uuid for item in missing] == ["a", "c", "b"]
    assert missing[0].missing == ("imp",)
    assert missing[1].missing == ("opt", "mode")
    assert missing[2].missing == ("urg",)


@pytest.mark.parametrize(
    ("current_mode", "required_mode", "expected"),
    [
        (None, None, 1.0),
        ("editorial", "editorial", 1.15),
        ("analysis", "editorial", 0.85),
        (None, "editorial", 0.95),
    ],
)
@pytest.mark.unit
def test_mode_multiplier(current_mode, required_mode, expected):
    """
    Validate the mode multiplier behavior.

    Parameters
    ----------
    current_mode : str | None
        Current mode context.
    required_mode : str | None
        Task-required mode.
    expected : float
        Expected multiplier.

    Returns
    -------
    None
        This test asserts on mode scoring.
    """
    assert review.mode_multiplier(current_mode, required_mode) == expected


@pytest.mark.unit
def test_score_task_prefers_matching_mode():
    """
    Ensure score favors tasks matching the current mode.

    Returns
    -------
    None
        This test asserts on score ordering.
    """
    task = review.ReviewTask(
        uuid="a",
        id=1,
        description="Task A",
        project=None,
        depends=[],
        imp=3,
        urg=1,
        opt=5,
        mode="editorial",
        dominates=[],
        raw={},
    )

    match_score, _ = review.score_task(task, "editorial")
    mismatch_score, _ = review.score_task(task, "analysis")

    assert match_score > mismatch_score


@pytest.mark.unit
def test_filter_candidates_applies_mode_and_dominates():
    """
    Verify candidate filtering respects mode and dominance.

    Returns
    -------
    None
        This test asserts candidate filtering.
    """
    t1 = review.ReviewTask(
        uuid="u1",
        id=1,
        description="Task 1",
        project=None,
        depends=[],
        imp=1,
        urg=1,
        opt=1,
        mode="editorial",
        dominates=["u2"],
        raw={},
    )
    t2 = review.ReviewTask(
        uuid="u2",
        id=2,
        description="Task 2",
        project=None,
        depends=[],
        imp=1,
        urg=1,
        opt=1,
        mode=None,
        dominates=[],
        raw={},
    )
    t3 = review.ReviewTask(
        uuid="u3",
        id=3,
        description="Task 3",
        project=None,
        depends=[],
        imp=1,
        urg=1,
        opt=1,
        mode="analysis",
        dominates=[],
        raw={},
    )

    candidates = review.filter_candidates(
        [t1, t2, t3],
        current_mode="editorial",
        strict_mode=False,
        include_dominated=False,
    )
    assert [task.uuid for task in candidates] == ["u1"]

    candidates = review.filter_candidates(
        [t1, t2, t3],
        current_mode="editorial",
        strict_mode=False,
        include_dominated=True,
    )
    assert [task.uuid for task in candidates] == ["u1", "u2"]

    candidates = review.filter_candidates(
        [t1, t2, t3],
        current_mode="editorial",
        strict_mode=True,
        include_dominated=True,
    )
    assert [task.uuid for task in candidates] == ["u1"]


@pytest.mark.unit
def test_rank_candidates_orders_by_score():
    """
    Ensure ranked candidates are sorted by score.

    Returns
    -------
    None
        This test asserts ordering of scored candidates.
    """
    t1 = review.ReviewTask(
        uuid="u1",
        id=1,
        description="Task 1",
        project=None,
        depends=[],
        imp=1,
        urg=0,
        opt=0,
        mode=None,
        dominates=[],
        raw={},
    )
    t2 = review.ReviewTask(
        uuid="u2",
        id=2,
        description="Task 2",
        project=None,
        depends=[],
        imp=1,
        urg=9,
        opt=0,
        mode=None,
        dominates=[],
        raw={},
    )

    ranked = review.rank_candidates([t2, t1], current_mode=None, top=2)

    assert [item.task.uuid for item in ranked] == ["u1", "u2"]


@pytest.mark.unit
def test_interactive_fill_missing_prompts_only_for_missing():
    """
    Confirm the wizard prompts only for missing fields.

    Returns
    -------
    None
        This test asserts interactive behavior.
    """
    task = review.ReviewTask(
        uuid="u1",
        id=1,
        description="Task 1",
        project=None,
        depends=[],
        imp=None,
        urg=3,
        opt=4,
        mode=None,
        dominates=[],
        raw={},
    )
    responses = iter(["7", "editorial"])

    def fake_input(_prompt):
        return next(responses)

    updates = review.interactive_fill_missing(task, input_func=fake_input)

    assert updates == {"imp": "7", "mode": "editorial"}


@pytest.mark.unit
def test_apply_updates_uses_modify(monkeypatch):
    """
    Ensure updates are applied via task modify.

    Returns
    -------
    None
        This test asserts command construction.
    """
    calls = []

    def fake_runner(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        return review.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(review, "run_task_command", fake_runner)

    review.apply_updates("uuid-1", {"imp": "3", "urg": "5"})

    assert calls == [
        (["uuid-1", "modify", "imp:3", "urg:5"], False),
    ]


@pytest.mark.unit
def test_build_review_report_combines_missing_and_candidates():
    """
    Verify review report includes missing metadata and scored candidates.

    Returns
    -------
    None
        This test asserts report structure.
    """
    tasks = [
        review.ReviewTask(
            uuid="a",
            id=1,
            description="Task A",
            project=None,
            depends=[],
            imp=None,
            urg=1,
            opt=2,
            mode="editorial",
            dominates=[],
            raw={},
        ),
        review.ReviewTask(
            uuid="b",
            id=2,
            description="Task B",
            project=None,
            depends=["a"],
            imp=1,
            urg=2,
            opt=3,
            mode="editorial",
            dominates=[],
            raw={},
        ),
    ]

    report = review.build_review_report(
        tasks,
        current_mode="editorial",
        strict_mode=False,
        include_dominated=True,
        top=3,
    )

    assert [item.task.uuid for item in report.missing] == ["a"]
    assert [item.task.uuid for item in report.candidates] == ["a"]


@pytest.mark.unit
def test_review_doctest_examples():
    """
    Run doctest examples embedded in review docstrings.

    Returns
    -------
    None
        This test asserts doctest coverage for review helpers.
    """
    results = doctest.testmod(review)
    assert results.failed == 0

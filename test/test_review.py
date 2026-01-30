"""
Tests for the review command logic.
"""

import doctest
from datetime import timezone

import pytest

import twh.review as review


@pytest.mark.parametrize(
    (
        "payload",
        "expected_depends",
        "expected_dominates",
        "expected_dominated_by",
        "expected_imp",
        "expected_diff",
        "expected_mode",
        "expected_annotations",
    ),
    [
        (
            {
                "uuid": "u1",
                "id": 1,
                "description": "  Task A  ",
                "depends": "a,b",
                "dominates": "c, d",
                "dominated_by": "e, f",
                "imp": "5",
                "diff": "4.5",
                "mode": " editorial ",
                "annotations": [
                    {"description": "Note one"},
                    {"description": "Note two"},
                ],
            },
            ["a", "b"],
            ["c", "d"],
            ["e", "f"],
            5,
            4.5,
            "editorial",
            ["Note one", "Note two"],
        ),
        (
            {
                "uuid": "u2",
                "id": 2,
                "description": "Task B",
                "depends": ["x", "y"],
                "dominates": ["z"],
                "dominated_by": ["w"],
                "imp": None,
                "diff": None,
                "mode": None,
            },
            ["x", "y"],
            ["z"],
            ["w"],
            None,
            None,
            None,
            [],
        ),
        (
            {
                "uuid": "u3",
                "id": 3,
                "description": "Task C",
                "depends": None,
                "dominates": None,
                "dominated_by": None,
                "imp": "bad",
                "diff": "bad",
                "mode": "",
                "annotations": [{"description": ""}],
            },
            [],
            [],
            [],
            None,
            None,
            None,
            [],
        ),
    ],
)
@pytest.mark.unit
def test_review_task_from_json_parses_fields(
    payload,
    expected_depends,
    expected_dominates,
    expected_dominated_by,
    expected_imp,
    expected_diff,
    expected_mode,
    expected_annotations,
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
    assert task.dominated_by == expected_dominated_by
    assert task.imp == expected_imp
    assert task.diff == expected_diff
    assert task.mode == expected_mode
    assert task.annotations == expected_annotations
    assert task.description == payload["description"].strip()


@pytest.mark.unit
def test_parse_annotations_formats_entry_time(monkeypatch):
    """
    Ensure annotation entries are formatted as local-readable date-times.

    Returns
    -------
    None
        This test asserts annotation formatting.
    """
    monkeypatch.setattr(review, "get_local_timezone", lambda: timezone.utc)

    annotations = review.parse_annotations(
        [{"entry": "20260127T180425Z", "description": "Note"}]
    )

    assert annotations == ["2026-01-27 6:04:25 PM: Note"]


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
            diff=2.0,
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
            diff=1.0,
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
            diff=None,
            mode=None,
            dominates=[],
            raw={},
        ),
    ]

    ready = review.ready_tasks(tasks)
    missing = review.collect_missing_metadata(tasks, ready)

    assert [item.task.uuid for item in missing] == ["a", "c", "b"]
    assert missing[0].missing == ("imp",)
    assert missing[1].missing == ("opt", "diff", "mode")
    assert missing[2].missing == ("urg",)


@pytest.mark.unit
def test_collect_missing_metadata_includes_dominance():
    """
    Ensure dominance incompleteness appears in missing metadata.

    Returns
    -------
    None
        This test asserts dominance metadata tracking.
    """
    tasks = [
        review.ReviewTask(
            uuid="a",
            id=1,
            description="Task A",
            project=None,
            depends=[],
            imp=1,
            urg=1,
            opt=1,
            diff=1.0,
            mode="editorial",
            dominates=[],
            raw={},
        ),
        review.ReviewTask(
            uuid="b",
            id=2,
            description="Task B",
            project=None,
            depends=[],
            imp=1,
            urg=1,
            opt=1,
            diff=1.0,
            mode="editorial",
            dominates=[],
            raw={},
        ),
    ]

    ready = review.ready_tasks(tasks)
    missing = review.collect_missing_metadata(tasks, ready, dominance_missing={"a", "b"})

    assert missing[0].missing == ("dominance",)
    assert missing[1].missing == ("dominance",)


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
        diff=2.0,
        mode="editorial",
        dominates=[],
        raw={},
    )

    match_score, _ = review.score_task(task, "editorial")
    mismatch_score, _ = review.score_task(task, "analysis")

    assert match_score > mismatch_score


@pytest.mark.unit
def test_score_task_prioritizes_time_pressure():
    """
    Ensure higher effort is prioritized when urgency is tight.

    Returns
    -------
    None
        This test asserts urgency/difficulty interplay.
    """
    high_effort = review.ReviewTask(
        uuid="u1",
        id=1,
        description="High effort",
        project=None,
        depends=[],
        imp=1,
        urg=1,
        opt=1,
        diff=12.0,
        mode=None,
        dominates=[],
        raw={},
    )
    low_effort = review.ReviewTask(
        uuid="u2",
        id=2,
        description="Low effort",
        project=None,
        depends=[],
        imp=1,
        urg=1,
        opt=1,
        diff=2.0,
        mode=None,
        dominates=[],
        raw={},
    )

    high_score, _ = review.score_task(high_effort, None)
    low_score, _ = review.score_task(low_effort, None)

    assert high_score > low_score


@pytest.mark.unit
def test_score_task_prefers_lower_difficulty_when_less_urgent():
    """
    Confirm lower difficulty is preferred when urgency is low.

    Returns
    -------
    None
        This test asserts low-urgency difficulty bias.
    """
    high_effort = review.ReviewTask(
        uuid="u1",
        id=1,
        description="High effort",
        project=None,
        depends=[],
        imp=1,
        urg=30,
        opt=1,
        diff=12.0,
        mode=None,
        dominates=[],
        raw={},
    )
    low_effort = review.ReviewTask(
        uuid="u2",
        id=2,
        description="Low effort",
        project=None,
        depends=[],
        imp=1,
        urg=30,
        opt=1,
        diff=2.0,
        mode=None,
        dominates=[],
        raw={},
    )

    high_score, _ = review.score_task(high_effort, None)
    low_score, _ = review.score_task(low_effort, None)

    assert low_score > high_score


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
        diff=1.0,
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
        diff=2.0,
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
        diff=3.0,
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
        diff=2.0,
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
        diff=2.0,
        mode=None,
        dominates=[],
        raw={},
    )

    ranked = review.rank_candidates([t2, t1], current_mode=None, top=2)

    assert [item.task.uuid for item in ranked] == ["u1", "u2"]


@pytest.mark.unit
def test_rank_candidates_respects_dominance():
    """
    Ensure dominance ordering overrides score ordering.

    Returns
    -------
    None
        This test asserts dominance-based ranking.
    """
    t1 = review.ReviewTask(
        uuid="u1",
        id=1,
        description="Dominant",
        project=None,
        depends=[],
        imp=1,
        urg=9,
        opt=0,
        diff=2.0,
        mode=None,
        dominates=["u2"],
        raw={},
    )
    t2 = review.ReviewTask(
        uuid="u2",
        id=2,
        description="Dominated",
        project=None,
        depends=[],
        imp=1,
        urg=0,
        opt=0,
        diff=2.0,
        mode=None,
        dominates=[],
        raw={},
    )

    ranked = review.rank_candidates([t2, t1], current_mode=None, top=2)

    assert [item.task.uuid for item in ranked] == ["u1", "u2"]


@pytest.mark.unit
def test_format_candidate_output_includes_annotations_and_dominance():
    """
    Ensure candidate output shows annotations and dominance relations.

    Returns
    -------
    None
        This test asserts review output formatting.
    """
    import twh.dominance as dominance

    t1 = review.ReviewTask(
        uuid="u1",
        id=1,
        description="Move A",
        project=None,
        depends=[],
        imp=1,
        urg=1,
        opt=1,
        diff=1.0,
        mode=None,
        dominates=["u2"],
        annotations=["Annotation one"],
        raw={},
    )
    t2 = review.ReviewTask(
        uuid="u2",
        id=2,
        description="Move B",
        project=None,
        depends=[],
        imp=1,
        urg=1,
        opt=1,
        diff=1.0,
        mode=None,
        dominates=[],
        raw={},
    )
    state = dominance.build_dominance_state([t1, t2])
    score, components = review.score_task(t1, None)
    candidate = review.ScoredTask(task=t1, score=score, components=components)

    lines = review.format_candidate_output(candidate, state, dominance_limit=3)

    assert any("Move A" in line for line in lines)
    assert "  Annotation: Annotation one" in lines
    assert "  Dominates move ID 2: Move B" in lines


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
        diff=None,
        mode=None,
        dominates=[],
        raw={},
    )
    responses = iter(["7", "2.5", "editorial"])

    def fake_input(_prompt):
        return next(responses)

    updates = review.interactive_fill_missing(task, input_func=fake_input)

    assert updates == {"imp": "7", "diff": "2.5", "mode": "editorial"}


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

    review.apply_updates(
        "uuid-1",
        {"imp": "3", "urg": "5"},
        get_setting=lambda _key: "numeric",
    )

    assert calls == [
        (["uuid-1", "modify", "imp:3", "urg:5"], True),
    ]


@pytest.mark.unit
def test_apply_updates_suppresses_modified_zero(monkeypatch, capsys):
    """
    Ensure "Modified 0 tasks." output is suppressed.

    Returns
    -------
    None
        This test asserts output filtering.
    """
    def fake_runner(args, capture_output=False, **_kwargs):
        assert capture_output is True
        return review.subprocess.CompletedProcess(
            args,
            0,
            stdout="Modified 0 tasks.\n",
            stderr="",
        )

    monkeypatch.setattr(review, "run_task_command", fake_runner)

    review.apply_updates(
        "uuid-1",
        {"imp": "3"},
        get_setting=lambda _key: "numeric",
    )

    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.unit
def test_apply_updates_requires_udas(monkeypatch):
    """
    Ensure missing UDAs abort updates to avoid modifying descriptions.

    Returns
    -------
    None
        This test asserts UDA guard behavior.
    """
    calls = []

    def fake_runner(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        return review.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(review, "run_task_command", fake_runner)

    with pytest.raises(RuntimeError):
        review.apply_updates(
            "uuid-1",
            {"diff": "2"},
            get_setting=lambda _key: None,
        )

    assert calls == []


@pytest.mark.unit
def test_load_pending_tasks_uses_filters(monkeypatch):
    """
    Ensure load_pending_tasks includes filter tokens in export call.

    Returns
    -------
    None
        This test asserts filter passthrough.
    """
    calls = []

    def fake_runner(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        return review.subprocess.CompletedProcess(args, 0, stdout="[]", stderr="")

    monkeypatch.setattr(review, "run_task_command", fake_runner)

    tasks = review.load_pending_tasks(filters=["project:work.competitiveness"])

    assert tasks == []
    assert calls == [
        (["project:work.competitiveness", "status:pending", "export"], True),
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
            diff=2.0,
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
            diff=1.0,
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
def test_run_review_wizard_includes_filtered_non_ready(monkeypatch):
    """
    Ensure filtered review runs wizard for non-ready tasks in scope.

    Returns
    -------
    None
        This test asserts wizard scope behavior.
    """
    tasks = [
        review.ReviewTask(
            uuid="u1",
            id=1,
            description="Blocked task",
            project="work",
            depends=["u2"],
            imp=None,
            urg=None,
            opt=None,
            diff=None,
            mode=None,
            dominates=[],
            raw={},
        ),
        review.ReviewTask(
            uuid="u2",
            id=2,
            description="Blocking task",
            project="work",
            depends=[],
            imp=1,
            urg=1,
            opt=1,
            diff=1.0,
            mode="editorial",
            dominates=[],
            raw={},
        ),
    ]

    calls = []

    def fake_load(filters=None):
        return tasks

    def fake_fill(task, input_func=input):
        calls.append(task.uuid)
        return {}

    def fake_report(*_args, **_kwargs):
        return review.ReviewReport(missing=[], candidates=[])

    monkeypatch.setattr(review, "load_pending_tasks", fake_load)
    monkeypatch.setattr(review, "interactive_fill_missing", fake_fill)
    monkeypatch.setattr(review, "build_review_report", fake_report)

    exit_code = review.run_review(
        mode=None,
        limit=20,
        top=5,
        strict_mode=False,
        include_dominated=False,
        wizard=True,
        wizard_once=False,
        filters=["project:work.competitiveness"],
        input_func=lambda _prompt: "",
    )

    assert exit_code == 0
    assert calls == ["u1"]


@pytest.mark.unit
def test_run_review_wizard_includes_non_ready_without_filters(monkeypatch):
    """
    Ensure the wizard prompts for missing metadata on blocked moves by default.

    Returns
    -------
    None
        This test asserts wizard scope behavior without filters.
    """
    tasks = [
        review.ReviewTask(
            uuid="u1",
            id=1,
            description="Blocked move",
            project="work",
            depends=["u2"],
            imp=None,
            urg=None,
            opt=None,
            diff=None,
            mode=None,
            dominates=[],
            raw={},
        ),
        review.ReviewTask(
            uuid="u2",
            id=2,
            description="Ready move",
            project="work",
            depends=[],
            imp=None,
            urg=None,
            opt=None,
            diff=None,
            mode=None,
            dominates=[],
            raw={},
        ),
    ]

    calls = []

    monkeypatch.setattr(review, "load_pending_tasks", lambda filters=None: tasks)
    monkeypatch.setattr(
        review,
        "_build_dominance_context",
        lambda pending: ({}, [], set()),
    )

    def fake_fill(task, input_func=input):
        calls.append(task.uuid)
        return {}

    def fake_report(*_args, **_kwargs):
        return review.ReviewReport(missing=[], candidates=[])

    monkeypatch.setattr(review, "interactive_fill_missing", fake_fill)
    monkeypatch.setattr(review, "build_review_report", fake_report)

    exit_code = review.run_review(
        mode=None,
        limit=20,
        top=5,
        strict_mode=False,
        include_dominated=False,
        wizard=True,
        wizard_once=False,
        filters=None,
        input_func=lambda _prompt: "",
    )

    assert exit_code == 0
    assert calls == ["u2", "u1"]


@pytest.mark.unit
def test_run_review_wizard_collects_dominance(monkeypatch):
    """
    Ensure the wizard triggers dominance collection when ordering is missing.

    Returns
    -------
    None
        This test asserts the dominance stage is executed.
    """
    tasks = [
        review.ReviewTask(
            uuid="u1",
            id=1,
            description="Move 1",
            project=None,
            depends=[],
            imp=1,
            urg=1,
            opt=1,
            diff=1.0,
            mode="editorial",
            dominates=[],
            raw={},
        ),
        review.ReviewTask(
            uuid="u2",
            id=2,
            description="Move 2",
            project=None,
            depends=[],
            imp=1,
            urg=1,
            opt=1,
            diff=1.0,
            mode="editorial",
            dominates=[],
            raw={},
        ),
    ]

    monkeypatch.setattr(review, "load_pending_tasks", lambda filters=None: tasks)
    import twh.dominance as dominance

    state = dominance.build_dominance_state(tasks)

    monkeypatch.setattr(
        review,
        "_build_dominance_context",
        lambda pending: (state, [], {"u1", "u2"}),
    )
    monkeypatch.setattr(
        review,
        "build_review_report",
        lambda *_args, **_kwargs: review.ReviewReport(missing=[], candidates=[]),
    )
    monkeypatch.setattr(review, "interactive_fill_missing", lambda *_args, **_kwargs: {})

    calls = {}

    def fake_sort_into_tiers(pending, state, chooser):
        calls["sort"] = [move.uuid for move in pending]
        return [[pending[0]], [pending[1]]]

    def fake_build_updates(tiers):
        calls["build"] = [[move.uuid for move in tier] for tier in tiers]
        return {}

    def fake_apply_updates(updates):
        calls["apply"] = updates

    monkeypatch.setattr(dominance, "sort_into_tiers", fake_sort_into_tiers)
    monkeypatch.setattr(dominance, "build_dominance_updates", fake_build_updates)
    monkeypatch.setattr(dominance, "apply_dominance_updates", fake_apply_updates)

    exit_code = review.run_review(
        mode=None,
        limit=20,
        top=5,
        strict_mode=False,
        include_dominated=False,
        wizard=True,
        wizard_once=False,
        filters=None,
        input_func=lambda _prompt: "",
    )

    assert exit_code == 0
    assert calls["sort"] == ["u1", "u2"]
    assert calls["build"] == [["u1"], ["u2"]]
    assert calls["apply"] == {}


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

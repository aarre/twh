"""
Tests for dominance collection and ordering.
"""

import pytest

import twh.dominance as dominance
import twh.review as review


def make_move(
    uuid: str,
    move_id: int | None = None,
    description: str | None = None,
    depends=None,
    dominates=None,
    dominated_by=None,
):
    return review.ReviewTask(
        uuid=uuid,
        id=move_id,
        description=description or f"Move {uuid}",
        project=None,
        depends=depends or [],
        imp=None,
        urg=None,
        opt=None,
        diff=None,
        mode=None,
        dominates=dominates or [],
        dominated_by=dominated_by or [],
        raw={},
    )


@pytest.mark.unit
def test_build_dominance_state_respects_dependencies_and_udas():
    """
    Ensure dependency and UDA edges become dominance relations.

    Returns
    -------
    None
        This test asserts dominance relations.
    """
    move_a = make_move("a", depends=["b"])
    move_b = make_move("b")
    move_c = make_move("c", dominates=["a"])

    state = dominance.build_dominance_state([move_a, move_b, move_c])

    assert dominance.dominance_relation(state, "b", "a") == "dominates"
    assert dominance.dominance_relation(state, "a", "b") == "dominated_by"
    assert dominance.dominance_relation(state, "c", "a") == "dominates"


@pytest.mark.unit
def test_dominance_relation_detects_ties():
    """
    Verify mutual edges are treated as ties.

    Returns
    -------
    None
        This test asserts tie detection.
    """
    move_a = make_move("a", dominates=["b"])
    move_b = make_move("b", dominates=["a"])

    state = dominance.build_dominance_state([move_a, move_b])

    assert dominance.dominance_relation(state, "a", "b") == "tie"


@pytest.mark.unit
def test_sort_into_tiers_avoids_prompt_for_dependencies():
    """
    Confirm dependency dominance does not trigger prompts.

    Returns
    -------
    None
        This test asserts prompt avoidance.
    """
    move_a = make_move("a", depends=["b"])
    move_b = make_move("b")
    state = dominance.build_dominance_state([move_a, move_b])

    calls = []

    def chooser(_left, _right):
        calls.append((_left.uuid, _right.uuid))
        return dominance.DominanceChoice.LEFT

    tiers = dominance.sort_into_tiers([move_a, move_b], state, chooser)

    assert calls == []
    assert [[move.uuid for move in tier] for tier in tiers] == [["b"], ["a"]]


@pytest.mark.parametrize(
    "depends",
    [
        ["2"],
        [2],
    ],
)
@pytest.mark.unit
def test_sort_into_tiers_avoids_prompt_for_dependency_ids(depends):
    """
    Ensure dependency IDs are treated as dominance relations.

    Returns
    -------
    None
        This test asserts dependency ID resolution.
    """
    move_a = make_move("a", move_id=1, depends=depends)
    move_b = make_move("b", move_id=2)
    state = dominance.build_dominance_state([move_a, move_b])

    calls = []

    def chooser(_left, _right):
        calls.append((_left.uuid, _right.uuid))
        return dominance.DominanceChoice.LEFT

    tiers = dominance.sort_into_tiers([move_a, move_b], state, chooser)

    assert calls == []
    assert [[move.uuid for move in tier] for tier in tiers] == [["b"], ["a"]]


@pytest.mark.unit
def test_sort_into_tiers_records_tie():
    """
    Ensure ties place moves in the same tier.

    Returns
    -------
    None
        This test asserts tie handling.
    """
    move_a = make_move("a")
    move_b = make_move("b")
    state = dominance.build_dominance_state([move_a, move_b])

    def chooser(_left, _right):
        return dominance.DominanceChoice.TIE

    tiers = dominance.sort_into_tiers([move_a, move_b], state, chooser)

    assert len(tiers) == 1
    assert {move.uuid for move in tiers[0]} == {"a", "b"}


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("A", dominance.DominanceChoice.LEFT),
        ("b", dominance.DominanceChoice.RIGHT),
        ("C", dominance.DominanceChoice.TIE),
    ],
)
@pytest.mark.unit
def test_prompt_dominance_choice_uses_lettered_labels(
    capsys,
    input_value,
    expected,
):
    """
    Ensure dominance prompts use A/B/C labels and accept letter input.

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

    choice = dominance.prompt_dominance_choice(move_a, move_b, input_func=fake_input)

    assert choice == expected
    assert prompts == ["Selection (A/B/C): "]
    captured = capsys.readouterr()
    assert "[A] Move ID 3: Pay taxes" in captured.out
    assert "[B] Move ID 4: Write report" in captured.out
    assert (
        "[A] Move A dominates, [B] Move B dominates, [C] Neither dominates (it is a tie)"
        in captured.out
    )


@pytest.mark.unit
def test_make_progress_chooser_reports_remaining(capsys):
    """
    Ensure progress messaging appears during dominance prompts.

    Returns
    -------
    None
        This test asserts progress output formatting.
    """
    move_a = make_move("a", move_id=1, description="Alpha")
    move_b = make_move("b", move_id=2, description="Beta")
    state = dominance.build_dominance_state([move_a, move_b])
    chooser = dominance.make_progress_chooser(
        [move_a, move_b],
        state,
        input_func=lambda _prompt: "A",
    )

    choice = chooser(move_a, move_b)

    assert choice == dominance.DominanceChoice.LEFT
    captured = capsys.readouterr()
    assert "Dominance progress:" in captured.out
    assert "comparisons complete" in captured.out
    assert "remaining" in captured.out


@pytest.mark.unit
def test_build_dominance_updates_orders_by_tier():
    """
    Verify dominance updates include lower-tier moves only.

    Returns
    -------
    None
        This test asserts update generation.
    """
    move_a = make_move("a")
    move_b = make_move("b")
    move_c = make_move("c")
    tiers = [[move_a], [move_b, move_c]]

    updates = dominance.build_dominance_updates(tiers)

    assert updates["a"].dominates == ["b", "c"]
    assert updates["a"].dominated_by == []
    assert updates["b"].dominates == []
    assert updates["b"].dominated_by == ["a", "c"]
    assert updates["c"].dominated_by == ["a", "b"]


@pytest.mark.unit
def test_apply_dominance_updates_writes_fields(monkeypatch):
    """
    Ensure dominance updates are written via task modify.

    Returns
    -------
    None
        This test asserts command construction.
    """
    updates = {
        "a": dominance.DominanceUpdate(dominates=["b"], dominated_by=[]),
        "b": dominance.DominanceUpdate(dominates=[], dominated_by=["a"]),
    }
    calls = []

    def fake_runner(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        return dominance.subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    dominance.apply_dominance_updates(
        updates,
        runner=fake_runner,
        get_setting=lambda _key: "string",
    )

    assert calls == [
        (["a", "modify", "dominates:b", "dominated_by:"], True),
        (["b", "modify", "dominates:", "dominated_by:a"], True),
    ]


@pytest.mark.unit
def test_apply_dominance_updates_suppresses_modified_zero(monkeypatch, capsys):
    """
    Ensure "Modified 0 tasks." output is suppressed.

    Returns
    -------
    None
        This test asserts output filtering.
    """
    updates = {
        "a": dominance.DominanceUpdate(dominates=["b"], dominated_by=[]),
    }

    def fake_runner(args, capture_output=False, **_kwargs):
        assert capture_output is True
        return dominance.subprocess.CompletedProcess(
            args,
            0,
            stdout="Modified 0 tasks.\n",
            stderr="",
        )

    dominance.apply_dominance_updates(
        updates,
        runner=fake_runner,
        get_setting=lambda _key: "string",
    )

    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.unit
def test_apply_dominance_updates_quiet_suppresses_output(capsys):
    """
    Ensure quiet mode suppresses Taskwarrior stdout.

    Returns
    -------
    None
        This test asserts quiet output handling.
    """
    updates = {
        "a": dominance.DominanceUpdate(dominates=["b"], dominated_by=[]),
    }

    def fake_runner(args, capture_output=False, **_kwargs):
        assert capture_output is True
        return dominance.subprocess.CompletedProcess(
            args,
            0,
            stdout=(
                "Modifying task 1 'Example'.\n"
                "Modified 1 task.\n"
                "Project 'work' is 0% complete (1 task remaining).\n"
            ),
            stderr="",
        )

    dominance.apply_dominance_updates(
        updates,
        runner=fake_runner,
        get_setting=lambda _key: "string",
        quiet=True,
    )

    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.unit
def test_run_dominance_quiet_suppresses_status(monkeypatch, capsys):
    """
    Ensure quiet mode suppresses dominance status output.

    Returns
    -------
    None
        This test asserts quiet run_dominance output handling.
    """
    move = make_move("a")

    monkeypatch.setattr(dominance, "load_pending_tasks", lambda filters=None: [move])
    monkeypatch.setattr(
        dominance,
        "dominance_missing_uuids",
        lambda _tasks, _state: {"a"},
    )
    monkeypatch.setattr(
        dominance,
        "sort_into_tiers",
        lambda _pending, _state, chooser: [[move]],
    )
    monkeypatch.setattr(
        dominance,
        "build_dominance_updates",
        lambda _tiers: {},
    )
    monkeypatch.setattr(
        dominance,
        "apply_dominance_updates",
        lambda *_args, **_kwargs: None,
    )

    exit_code = dominance.run_dominance(quiet=True)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
@pytest.mark.unit
def test_apply_dominance_updates_requires_udas(tmp_path, monkeypatch):
    """
    Ensure missing dominance UDAs abort updates.

    Returns
    -------
    None
        This test asserts UDA guard behavior.
    """
    taskrc = tmp_path / ".taskrc"
    taskrc.write_text("", encoding="utf-8")
    monkeypatch.setenv("TASKRC", str(taskrc))
    updates = {
        "a": dominance.DominanceUpdate(dominates=["b"], dominated_by=[]),
    }

    with pytest.raises(RuntimeError):
        dominance.apply_dominance_updates(
            updates,
            runner=lambda *_args, **_kwargs: None,
            get_setting=lambda _key: None,
        )


@pytest.mark.unit
def test_dominance_missing_uuids_detects_unknown_pairs():
    """
    Ensure unknown dominance pairs are flagged as missing.

    Returns
    -------
    None
        This test asserts missing dominance detection.
    """
    move_a = make_move("a")
    move_b = make_move("b")
    state = dominance.build_dominance_state([move_a, move_b])

    missing = dominance.dominance_missing_uuids([move_a, move_b], state)

    assert missing == {"a", "b"}


@pytest.mark.unit
def test_dominance_missing_uuids_skips_ties():
    """
    Ensure tied moves are not flagged as missing.

    Returns
    -------
    None
        This test asserts tie handling in missing detection.
    """
    move_a = make_move("a", dominated_by=["b"])
    move_b = make_move("b", dominated_by=["a"])
    state = dominance.build_dominance_state([move_a, move_b])

    missing = dominance.dominance_missing_uuids([move_a, move_b], state)

    assert missing == set()

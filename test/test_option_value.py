"""
Tests for option value calculation and calibration.
"""

import subprocess
from datetime import datetime, timezone
from math import log

import pytest

import twh.option_value as option_value


def make_task(
    uuid: str,
    *,
    task_id: int | None = None,
    description: str = "Move",
    status: str = "pending",
    project: str | None = None,
    tags: tuple[str, ...] = (),
    due: datetime | None = None,
    priority: str | None = None,
    estimate_minutes: int | None = None,
    door: str | None = None,
    kind: str | None = None,
    depends: list[str] | None = None,
    opt_human: float | None = None,
    opt: float | None = None,
    opt_auto: float | None = None,
) -> option_value.OptionTask:
    return option_value.OptionTask(
        uuid=uuid,
        id=task_id,
        description=description,
        status=status,
        project=project,
        tags=tags,
        due=due,
        priority=priority,
        estimate_minutes=estimate_minutes,
        door=door,
        kind=kind,
        depends=depends or [],
        opt_human=opt_human,
        opt=opt,
        opt_auto=opt_auto,
    )


@pytest.mark.unit
def test_option_task_from_json_parses_fields():
    """
    Ensure option tasks parse tags, due, and manual values.

    Returns
    -------
    None
        This test asserts option task parsing.
    """
    payload = {
        "uuid": "u1",
        "id": 12,
        "description": "Move A",
        "status": "pending",
        "project": "alpha",
        "tags": ["probe", "oneway"],
        "due": "20240102T120000Z",
        "priority": "H",
        "estimate_minutes": "90",
        "door": "oneway",
        "kind": "probe",
        "depends": "u2,u3",
        "opt_human": "7",
        "opt": "6",
        "opt_auto": "6.5",
    }

    task = option_value.OptionTask.from_json(payload)

    assert task.uuid == "u1"
    assert task.id == 12
    assert task.tags == ("probe", "oneway")
    assert task.due == datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    assert task.priority == "H"
    assert task.estimate_minutes == 90
    assert task.door == "oneway"
    assert task.kind == "probe"
    assert task.depends == ["u2", "u3"]
    assert task.opt_human == 7.0
    assert task.opt == 6.0
    assert task.opt_auto == 6.5


@pytest.mark.unit
def test_feature_vector_zero_when_no_graph():
    """
    Ensure empty graphs yield zero features.

    Returns
    -------
    None
        This test asserts feature vector baseline.
    """
    tasks = {
        "u1": make_task("u1"),
        "u2": make_task("u2"),
    }

    features = option_value.feature_vector("u1", tasks, [])

    assert features == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.unit
def test_feature_vector_counts_children_and_descendants():
    """
    Ensure feature vector counts children and descendants.

    Returns
    -------
    None
        This test asserts downstream feature computations.
    """
    tasks = {
        "a": make_task("a"),
        "b": make_task("b", project="p1"),
        "c": make_task("c", project="p2"),
    }
    deps = [("b", "a"), ("c", "a")]

    features = option_value.feature_vector("a", tasks, deps)

    assert features[0] == pytest.approx(log(3.0))
    assert features[1] == pytest.approx(log(3.0))
    assert features[2] == pytest.approx(log(3.0))
    assert features[3] == pytest.approx(log(2.0))


@pytest.mark.unit
def test_manual_option_value_prefers_opt_human():
    """
    Ensure opt_human takes precedence over legacy opt values.

    Returns
    -------
    None
        This test asserts manual option value selection.
    """
    task = make_task("u1", opt=3.0, opt_human=7.0)
    assert option_value.manual_option_value(task) == 7.0

    task = make_task("u2", opt=3.0)
    assert option_value.manual_option_value(task) == 3.0

    task = make_task("u3")
    assert option_value.manual_option_value(task) is None


@pytest.mark.unit
def test_apply_option_values_copies_opt_to_opt_human(monkeypatch):
    """
    Ensure legacy opt values are copied to opt_human when applying.

    Returns
    -------
    None
        This test asserts opt_human migration.
    """
    tasks = [
        make_task("u1", opt=7.0, opt_auto=1.0),
        make_task("u2", opt_human=5.0, opt=5.0, opt_auto=2.0),
    ]
    predictions = {"u1": 4.2, "u2": 2.0}

    calls = []

    def fake_runner(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(option_value, "missing_udas", lambda _fields: [])

    exit_code = option_value.apply_option_values(
        tasks,
        predictions,
        runner=fake_runner,
    )

    assert exit_code == 0
    assert calls == [
        (["u1", "modify", "opt_auto:4.2", "opt_human:7.0"], True),
    ]


@pytest.mark.unit
def test_fit_weights_ridge_uses_bias_when_features_zero():
    """
    Ensure ridge fitting learns the mean when features are zero.

    Returns
    -------
    None
        This test asserts bias-only fitting.
    """
    tasks = {
        "u1": make_task("u1"),
        "u2": make_task("u2"),
        "u3": make_task("u3"),
    }
    training = [
        ("u1", 2.0),
        ("u2", 4.0),
        ("u3", 6.0),
    ]

    weights = option_value.fit_weights_ridge(training, tasks, [], lam=1.0)

    assert weights.bias == pytest.approx(4.0)
    assert weights.w_children == pytest.approx(0.0)
    assert weights.w_desc == pytest.approx(0.0)
    assert weights.w_desc_value == pytest.approx(0.0)
    assert weights.w_diversity == pytest.approx(0.0)
    assert weights.w_info == pytest.approx(0.0)
    assert weights.w_rev == pytest.approx(0.0)
    assert weights.w_cost == pytest.approx(0.0)


@pytest.mark.unit
def test_option_value_score_clamps_range():
    """
    Ensure option value scores are clamped to 0-10.

    Returns
    -------
    None
        This test asserts score clamping.
    """
    tasks = {"u1": make_task("u1")}
    weights = option_value.Weights(bias=12.0)

    score = option_value.option_value_score("u1", tasks, [], weights=weights)

    assert score == 10.0


@pytest.mark.unit
def test_option_value_doctest_examples():
    """
    Run doctest examples embedded in option value docstrings.

    Returns
    -------
    None
        This test asserts doctest coverage for option value helpers.
    """
    import doctest

    results = doctest.testmod(option_value)
    assert results.failed == 0

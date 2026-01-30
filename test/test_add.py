"""
Tests for the interactive add workflow.
"""

import subprocess

import pytest
import twh
import twh.dominance as dominance


@pytest.mark.unit
def test_prompt_add_input_orders_prompts():
    """
    Ensure the interactive add prompts follow the required order.

    Returns
    -------
    None
        This test asserts prompt ordering and parsed values.
    """
    prompts = []
    responses = iter(
        [
            "Write notes",
            "work",
            "alpha, +beta, alpha",
            "2024-02-01",
            "12, 34",
            "30",
            "10",
            "8",
            "2.5",
            "analysis",
        ]
    )

    def fake_input(prompt):
        prompts.append(prompt)
        return next(responses)

    add_input = twh.prompt_add_input(input_func=fake_input, description_default=None)

    assert prompts == [
        "Move description: ",
        "Project: ",
        "Tags (comma-separated): ",
        "Due date: ",
        "Blocks (move IDs blocked by this move, comma-separated): ",
        "  Importance horizon - how long will you remember whether this move was done? (days): ",
        "  Urgency horizon - how long before acting loses value? (days): ",
        "  Option value - to what extent does doing this move preserve, unlock, or multiply future moves? (0-10): ",
        "  Difficulty, i.e., estimated effort (hours): ",
        "  Mode (e.g., analysis/research/writing/editorial/illustration/programming/teaching/chore/errand): ",
    ]
    assert add_input.description == "Write notes"
    assert add_input.project == "work"
    assert add_input.tags == ["alpha", "beta"]
    assert add_input.due == "2024-02-01"
    assert add_input.blocks == ["12", "34"]
    assert add_input.metadata == {
        "imp": "30",
        "urg": "10",
        "opt": "8",
        "diff": "2.5",
        "mode": "analysis",
    }


@pytest.mark.parametrize(
    ("add_input", "expected_args"),
    [
        (
            twh.AddMoveInput(
                description="Write notes",
                project="work",
                tags=["alpha", "beta"],
                due="2024-02-01",
                blocks=[],
                metadata={"mode": "analysis", "imp": "10", "urg": "5"},
            ),
            [
                "add",
                "Write notes",
                "project:work",
                "+alpha",
                "+beta",
                "due:2024-02-01",
                "imp:10",
                "urg:5",
                "mode:analysis",
            ],
        ),
        (
            twh.AddMoveInput(
                description="Fix issue",
                project=None,
                tags=[],
                due=None,
                blocks=[],
                metadata={"diff": "2"},
            ),
            ["add", "Fix issue", "diff:2"],
        ),
    ],
)
@pytest.mark.unit
def test_build_add_args_orders_metadata(add_input, expected_args):
    """
    Verify add arguments include metadata in the expected order.

    Parameters
    ----------
    add_input : twh.AddMoveInput
        Add input payload.
    expected_args : list[str]
        Expected task add arguments.

    Returns
    -------
    None
        This test asserts add argument formatting.
    """
    assert twh.build_add_args(add_input) == expected_args


@pytest.mark.unit
def test_run_interactive_add_runs_add_blocks_and_dominance(monkeypatch):
    """
    Ensure interactive add runs task add, blocks updates, and dominance.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching Taskwarrior and dominance helpers.

    Returns
    -------
    None
        This test asserts the interactive add flow.
    """
    prompts = iter(
        [
            "Plan work",
            "",
            "",
            "",
            "32,33",
            "7",
            "3",
            "5",
            "1.5",
            "analysis",
        ]
    )

    def fake_input(prompt):
        return next(prompts)

    calls = []

    def fake_run_task_command(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        if args[0] == "add":
            return subprocess.CompletedProcess(
                args,
                0,
                stdout="Created task 45.\n",
                stderr="",
            )
        return subprocess.CompletedProcess(args, 0, stdout="Modified 1 task.\n", stderr="")

    dominance_calls = []

    def fake_run_dominance(*_args, **_kwargs):
        dominance_calls.append(True)
        return 0

    monkeypatch.setattr(twh, "run_task_command", fake_run_task_command)
    monkeypatch.setattr(twh, "missing_udas", lambda _fields: [])
    monkeypatch.setattr(twh, "get_active_context_name", lambda: None)
    monkeypatch.setattr(twh, "get_context_definition", lambda _name: None)
    monkeypatch.setattr(dominance, "run_dominance", fake_run_dominance)

    exit_code = twh.run_interactive_add([], input_func=fake_input)

    assert exit_code == 0
    assert calls == [
        (
            [
                "add",
                "Plan work",
                "imp:7",
                "urg:3",
                "opt:5",
                "diff:1.5",
                "mode:analysis",
            ],
            True,
        ),
        (["32", "modify", "depends:+45"], True),
        (["33", "modify", "depends:+45"], True),
    ]
    assert dominance_calls == [True]


@pytest.mark.unit
def test_run_interactive_add_applies_context(monkeypatch, capsys):
    """
    Verify interactive add applies context project/tags when missing.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching Taskwarrior helpers.
    capsys : pytest.CaptureFixture[str]
        Fixture for capturing output.

    Returns
    -------
    None
        This test asserts context application and messaging.
    """
    prompts = iter(
        [
            "Write move",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    )

    def fake_input(prompt):
        return next(prompts)

    calls = []

    def fake_run_task_command(args, capture_output=False, **_kwargs):
        calls.append((args, capture_output))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="Created task 9.\n",
            stderr="",
        )

    monkeypatch.setattr(twh, "run_task_command", fake_run_task_command)
    monkeypatch.setattr(twh, "missing_udas", lambda _fields: [])
    monkeypatch.setattr(twh, "get_active_context_name", lambda: "work")
    monkeypatch.setattr(
        twh,
        "get_context_definition",
        lambda _name: "project:alpha +tag1 +tag2",
    )
    monkeypatch.setattr(dominance, "run_dominance", lambda *_a, **_k: 0)

    exit_code = twh.run_interactive_add([], input_func=fake_input)

    assert exit_code == 0
    assert calls == [
        (
            [
                "add",
                "Write move",
                "project:alpha",
                "+tag1",
                "+tag2",
            ],
            True,
        )
    ]
    captured = capsys.readouterr()
    assert "twh: project set to alpha; tags set to tag1, tag2 because context is work" in captured.out


@pytest.mark.unit
def test_run_interactive_add_requires_udas_for_metadata(monkeypatch, capsys):
    """
    Ensure missing UDAs abort interactive add before task creation.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching Taskwarrior helpers.
    capsys : pytest.CaptureFixture[str]
        Fixture for capturing output.

    Returns
    -------
    None
        This test asserts missing UDA handling.
    """
    prompts = iter(
        [
            "Write move",
            "",
            "",
            "",
            "",
            "9",
            "",
            "",
            "",
            "",
        ]
    )

    def fake_input(prompt):
        return next(prompts)

    def fake_missing_udas(fields):
        return ["imp"] if "imp" in fields else []

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("Task add should not run when UDAs are missing.")

    monkeypatch.setattr(twh, "run_task_command", unexpected_run)
    monkeypatch.setattr(twh, "missing_udas", fake_missing_udas)
    monkeypatch.setattr(twh, "get_active_context_name", lambda: None)
    monkeypatch.setattr(twh, "get_context_definition", lambda _name: None)

    exit_code = twh.run_interactive_add([], input_func=fake_input)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Missing Taskwarrior UDA(s): imp." in captured.err

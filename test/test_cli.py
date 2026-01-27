"""
Tests for the twh CLI delegation behavior.
"""

import doctest
import subprocess
import sys

import pytest
import twh
import twh.renderer as renderer
import twh.taskwarrior as taskwarrior

GRAPH2_COMMAND = twh.graph2

import twh.graph2 as graph2


@pytest.mark.parametrize(
    ("definition", "expected_project", "expected_tags"),
    [
        ("project:work", "work", []),
        ("+home", None, ["home"]),
        ("project:work +alpha +beta", "work", ["alpha", "beta"]),
        ("tag:alpha project:ops", "ops", ["alpha"]),
        ("-waiting +one +one +two", None, ["one", "two"]),
    ],
)
@pytest.mark.unit
def test_parse_context_filters(definition, expected_project, expected_tags):
    """
    Verify context definitions yield the right project and tags.

    Parameters
    ----------
    definition : str
        Context definition string.
    expected_project : str | None
        Expected project from the context definition.
    expected_tags : list[str]
        Expected tag list from the context definition.

    Returns
    -------
    None
        This test asserts on context parsing behavior.
    """
    project, tags = twh.parse_context_filters(definition)
    assert project == expected_project
    assert tags == expected_tags


@pytest.mark.parametrize(
    ("argv", "context_name", "context_definition", "expected_args", "expected_message"),
    [
        (
            ["add", "Implement feedback"],
            "grin",
            "project:work.competitiveness.gloria.grinsector",
            ["add", "Implement feedback", "project:work.competitiveness.gloria.grinsector"],
            "twh: project set to work.competitiveness.gloria.grinsector because context is grin",
        ),
        (
            ["add", "Fix docs"],
            "docs",
            "+docs",
            ["add", "Fix docs", "+docs"],
            "twh: tag set to docs because context is docs",
        ),
        (
            ["add", "Triage backlog"],
            "mix",
            "project:work +alpha +beta",
            ["add", "Triage backlog", "project:work", "+alpha", "+beta"],
            "twh: project set to work; tags set to alpha, beta because context is mix",
        ),
        (
            ["add", "Already set", "project:other"],
            "grin",
            "project:work",
            ["add", "Already set", "project:other"],
            None,
        ),
        (
            ["add", "Already tagged", "+home"],
            "home",
            "+home",
            ["add", "Already tagged", "+home"],
            None,
        ),
        (
            ["add", "Literal tag", "--", "+home"],
            "home",
            "+home",
            ["add", "Literal tag", "+home", "--", "+home"],
            "twh: tag set to home because context is home",
        ),
        (
            ["list"],
            "work",
            "project:work",
            ["list"],
            None,
        ),
    ],
)
@pytest.mark.unit
def test_apply_context_to_add_args(
    monkeypatch,
    argv,
    context_name,
    context_definition,
    expected_args,
    expected_message,
):
    """
    Ensure context metadata is applied to add arguments when needed.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching context lookups.
    argv : list[str]
        Command args to pass into the helper.
    context_name : str | None
        Active context name.
    context_definition : str | None
        Context definition filter.
    expected_args : list[str]
        Expected argv after applying context.
    expected_message : str | None
        Expected informational message, if any.

    Returns
    -------
    None
        This test asserts on add argument augmentation.
    """
    monkeypatch.setattr(twh, "get_active_context_name", lambda: context_name)
    monkeypatch.setattr(twh, "get_context_definition", lambda name: context_definition)

    updated_args, message = twh.apply_context_to_add_args(argv)

    assert updated_args == expected_args
    assert message == expected_message


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], True),
        (["project:work"], True),
        (["add", "Next task"], True),
        (["list"], False),
        (["reverse"], False),
        (["tree"], False),
        (["graph"], False),
        (["graph2"], False),
        (["--help"], False),
    ],
)
@pytest.mark.unit
def test_should_delegate_to_task(argv, expected):
    """
    Verify when commands should be delegated to Taskwarrior.

    Parameters
    ----------
    argv : list[str]
        Argument list excluding the program name.
    expected : bool
        Expected delegation decision.

    Returns
    -------
    None
        This test asserts on delegation behavior.
    """
    assert twh.should_delegate_to_task(argv) is expected


@pytest.mark.parametrize(
    ("argv", "expected_args"),
    [
        (["twh"], ["task"]),
        (["twh", "project:work"], ["task", "project:work"]),
        (["twh", "add", "Next task"], ["task", "add", "Next task"]),
        (["twh", "21", "modify", "project:personal", "depends:20"],
         ["task", "21", "modify", "project:personal", "depends:20"]),
    ],
)
@pytest.mark.unit
def test_main_delegates_to_task(monkeypatch, argv, expected_args):
    """
    Ensure main forwards unknown commands to Taskwarrior.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching subprocess and argv.
    argv : list[str]
        Full argv including the program name.
    expected_args : list[str]
        Expected Taskwarrior invocation.

    Returns
    -------
    None
        This test asserts on subprocess delegation and exit code.
    """
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(twh, "get_active_context_name", lambda: None)
    monkeypatch.setattr(twh.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        twh.main()

    assert excinfo.value.code == 0
    assert calls == [expected_args]


@pytest.mark.unit
def test_main_applies_context_to_add(monkeypatch, capsys):
    """
    Confirm twh add applies project context before delegating.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching subprocess and context lookups.
    capsys : pytest.CaptureFixture[str]
        Fixture to capture stdout and stderr.

    Returns
    -------
    None
        This test asserts on context-aware add behavior.
    """
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(twh, "get_active_context_name", lambda: "grin")
    monkeypatch.setattr(
        twh,
        "get_context_definition",
        lambda name: "project:work.competitiveness.gloria.grinsector",
    )
    monkeypatch.setattr(twh.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["twh", "add", "Implement feedback"])

    with pytest.raises(SystemExit) as excinfo:
        twh.main()

    assert excinfo.value.code == 0
    assert calls == [
        [
            "task",
            "add",
            "Implement feedback",
            "project:work.competitiveness.gloria.grinsector",
        ]
    ]
    captured = capsys.readouterr()
    assert (
        "twh: project set to work.competitiveness.gloria.grinsector because context is grin"
        in captured.out
    )


@pytest.mark.parametrize(
    ("mode", "expect_reverse"),
    [
        (None, False),
        ("reverse", True),
    ],
)
@pytest.mark.unit
def test_graph2_defaults_to_svg_and_opens(
    monkeypatch,
    tmp_path,
    capsys,
    mode,
    expect_reverse,
):
    """
    Ensure graph2 renders SVG output by default and opens it.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching render helpers.
    tmp_path : pathlib.Path
        Temporary directory for output paths.
    capsys : pytest.CaptureFixture[str]
        Fixture to capture stdout and stderr.
    mode : str | None
        Optional graph2 mode argument.
    expect_reverse : bool
        Expected reverse flag passed to edge builder.

    Returns
    -------
    None
        This test asserts on default graph2 rendering behavior.
    """
    def run_graph2(**overrides):
        params = {
            "mode": None,
            "reverse": False,
            "png": None,
            "svg": None,
            "ascii_only": False,
            "edges": False,
            "rankdir": "LR",
        }
        params.update(overrides)
        GRAPH2_COMMAND(**params)

    tasks = [{"uuid": "a", "description": "Task A"}]
    monkeypatch.setattr(taskwarrior, "get_tasks_from_taskwarrior", lambda: tasks)
    monkeypatch.setattr(twh, "get_graph_output_dir", lambda: tmp_path)

    def fake_build_dependency_edges(tasks_arg, reverse=False):
        assert reverse is expect_reverse
        return [], {"a": tasks_arg[0]}

    monkeypatch.setattr(graph2, "build_dependency_edges", fake_build_dependency_edges)
    monkeypatch.setattr(graph2, "generate_dot", lambda *args, **kwargs: "digraph twh {}")

    calls = {}

    def fake_render(dot_source, png_path, svg_path):
        calls["render"] = (dot_source, png_path, svg_path)
        return True, None

    monkeypatch.setattr(graph2, "render_graphviz", fake_render)

    ascii_called = {"value": False}

    def fake_ascii(*args, **kwargs):
        ascii_called["value"] = True
        return ["ASCII"]

    monkeypatch.setattr(graph2, "ascii_forest", fake_ascii)

    opened = {}

    def fake_open_svg(path):
        opened["svg"] = path

    monkeypatch.setattr(renderer, "open_in_browser", fake_open_svg)
    monkeypatch.setattr(renderer, "open_file", lambda path: opened.setdefault("png", path))

    if mode:
        run_graph2(mode=mode)
    else:
        run_graph2()

    captured = capsys.readouterr()
    expected_svg = tmp_path / "tasks-graph2.svg"
    assert calls["render"][1] is None
    assert calls["render"][2] == expected_svg
    assert opened["svg"] == expected_svg
    assert "ASCII" not in captured.out
    assert ascii_called["value"] is False


@pytest.mark.parametrize(
    "render_error",
    [
        "Graphviz 'dot' not found on PATH.",
    ],
)
@pytest.mark.unit
def test_graph2_falls_back_to_ascii_when_render_fails(
    monkeypatch,
    tmp_path,
    capsys,
    render_error,
):
    """
    Ensure graph2 prints ASCII output if rendering fails.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching render helpers.
    tmp_path : pathlib.Path
        Temporary directory for output paths.
    capsys : pytest.CaptureFixture[str]
        Fixture to capture stdout and stderr.
    render_error : str
        Render error message to simulate.

    Returns
    -------
    None
        This test asserts on ASCII fallback behavior.
    """
    def run_graph2():
        GRAPH2_COMMAND(
            mode=None,
            reverse=False,
            png=None,
            svg=None,
            ascii_only=False,
            edges=False,
            rankdir="LR",
        )

    tasks = [{"uuid": "a", "description": "Task A"}]
    monkeypatch.setattr(taskwarrior, "get_tasks_from_taskwarrior", lambda: tasks)
    monkeypatch.setattr(twh, "get_graph_output_dir", lambda: tmp_path)
    monkeypatch.setattr(graph2, "build_dependency_edges", lambda tasks_arg, reverse=False: ([], {"a": tasks_arg[0]}))
    monkeypatch.setattr(graph2, "generate_dot", lambda *args, **kwargs: "digraph twh {}")
    monkeypatch.setattr(graph2, "render_graphviz", lambda *args, **kwargs: (False, render_error))
    monkeypatch.setattr(graph2, "ascii_forest", lambda *args, **kwargs: ["ASCII fallback"])

    def unexpected_open(*_args, **_kwargs):
        raise AssertionError("Open should not be called when rendering fails.")

    monkeypatch.setattr(renderer, "open_in_browser", unexpected_open)
    monkeypatch.setattr(renderer, "open_file", unexpected_open)

    run_graph2()

    captured = capsys.readouterr()
    assert "ASCII fallback" in captured.out
    assert f"twh: {render_error}" in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["twh", "list"],
        ["twh", "reverse"],
        ["twh", "tree"],
        ["twh", "graph"],
        ["twh", "graph2"],
        ["twh", "--help"],
    ],
)
@pytest.mark.unit
def test_main_uses_twh_commands(monkeypatch, argv):
    """
    Confirm twh commands invoke the Typer app directly.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching the Typer app and subprocess.
    argv : list[str]
        Full argv including the program name.

    Returns
    -------
    None
        This test asserts on app invocation and no delegation.
    """
    called = {"app": 0, "task": 0}

    def fake_app():
        called["app"] += 1

    def fake_run(args, **kwargs):
        called["task"] += 1
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(twh, "app", fake_app)
    monkeypatch.setattr(twh.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", argv)

    twh.main()

    assert called["app"] == 1
    assert called["task"] == 0


@pytest.mark.unit
def test_doctest_examples():
    """
    Run doctest examples embedded in twh docstrings.

    Returns
    -------
    None
        This test asserts that doctest examples succeed.
    """
    results = doctest.testmod(twh)
    assert results.failed == 0

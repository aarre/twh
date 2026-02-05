"""
Tests for the twh CLI delegation behavior.
"""

import doctest
import subprocess
import sys

import pytest
import twh
import twh.renderer as renderer
from typer.testing import CliRunner

GRAPH_COMMAND = twh.graph

import twh.graph as graph_module


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
        (["add", "Next task"], False),
        (["list"], False),
        (["reverse"], False),
        (["tree"], False),
        (["simple"], False),
        (["graph"], False),
        (["help"], False),
        (["ondeck"], False),
        (["option"], False),
        (["dominance"], False),
        (["criticality"], False),
        (["calibrate"], False),
        (["diagnose"], False),
        (["start"], False),
        (["stop"], False),
        (["time"], False),
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


@pytest.fixture(scope="module")
def help_output_lines():
    """
    Capture the twh help output once for reuse across tests.

    Returns
    -------
    list[str]
        Lines of output produced by the help command.
    """
    runner = CliRunner()
    result = runner.invoke(twh.build_app(), ["help"])
    assert result.exit_code == 0
    return [line.rstrip() for line in result.stdout.splitlines()]


@pytest.mark.unit
def test_help_command_includes_header_and_footer(help_output_lines):
    """
    Confirm the help command prints a header and a Taskwarrior reminder.

    Parameters
    ----------
    help_output_lines : list[str]
        Lines produced by the help command.

    Returns
    -------
    None
        This test asserts on the help output framing.
    """
    assert help_output_lines[0] == "twh commands:"
    assert help_output_lines[-1] == "Use task help for Taskwarrior commands."


@pytest.mark.parametrize(
    "command",
    [
        "add",
        "list",
        "reverse",
        "tree",
        "graph",
        "simple",
        "ondeck",
        "defer",
        "diagnose",
        "dominance",
        "criticality",
        "option",
        "calibrate",
        "start",
        "stop",
        "time",
        "help",
    ],
)
@pytest.mark.unit
def test_help_command_lists_twh_commands(help_output_lines, command):
    """
    Ensure the help output lists each twh-specific command.

    Parameters
    ----------
    help_output_lines : list[str]
        Lines produced by the help command.
    command : str
        Command expected in the help output.

    Returns
    -------
    None
        This test asserts on the command listing.
    """
    assert any(line.strip().startswith(command) for line in help_output_lines)


@pytest.mark.unit
def test_ondeck_default_top(monkeypatch):
    """
    Ensure ondeck defaults to showing 25 top candidates.

    Returns
    -------
    None
        This test asserts the CLI default.
    """
    runner = CliRunner()
    captured: dict = {}

    def fake_run_ondeck(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(twh.review, "run_ondeck", fake_run_ondeck)

    result = runner.invoke(twh.build_app(), ["ondeck"])

    assert result.exit_code == 0
    assert captured["top"] == 25


@pytest.mark.parametrize(
    ("argv", "expected_args"),
    [
        (["twh"], []),
        (["twh", "project:work"], ["project:work"]),
        (["twh", "21", "modify", "project:personal", "depends:20"],
         ["21", "modify", "project:personal", "depends:20"]),
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

    def fake_exec(args):
        calls.append(args)
        return 0

    monkeypatch.setattr(twh, "get_active_context_name", lambda: None)
    monkeypatch.setattr(twh, "exec_task_command", fake_exec)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        twh.main()

    assert excinfo.value.code == 0
    assert calls == [expected_args]


@pytest.mark.unit
def test_main_routes_add_to_interactive(monkeypatch):
    """
    Confirm twh add uses the interactive flow.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching subprocess and argv.

    Returns
    -------
    None
        This test asserts interactive add routing.
    """
    calls = []

    def fake_run_interactive_add(args, **_kwargs):
        calls.append(args)
        return 0

    def unexpected_exec(_args):
        raise AssertionError("Delegation should not occur for interactive add.")

    monkeypatch.setattr(twh, "run_interactive_add", fake_run_interactive_add)
    monkeypatch.setattr(twh, "exec_task_command", unexpected_exec)
    monkeypatch.setattr(sys, "argv", ["twh", "add", "Implement feedback"])

    with pytest.raises(SystemExit) as excinfo:
        twh.main()

    assert excinfo.value.code == 0
    assert calls == [["Implement feedback"]]


@pytest.mark.parametrize(
    ("mode", "expect_reverse"),
    [
        (None, False),
        ("reverse", True),
    ],
)
@pytest.mark.unit
def test_graph_defaults_to_svg_and_opens(
    monkeypatch,
    tmp_path,
    capsys,
    mode,
    expect_reverse,
):
    """
    Ensure graph renders SVG output by default and opens it.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching render helpers.
    tmp_path : pathlib.Path
        Temporary directory for output paths.
    capsys : pytest.CaptureFixture[str]
        Fixture to capture stdout and stderr.
    mode : str | None
        Optional graph mode argument.
    expect_reverse : bool
        Expected reverse flag passed to edge builder.

    Returns
    -------
    None
        This test asserts on default graph rendering behavior.
    """
    def run_graph(**overrides):
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
        GRAPH_COMMAND(**params)

    tasks = [{"uuid": "a", "description": "Task A"}]
    monkeypatch.setattr(twh, "get_tasks_from_taskwarrior", lambda: tasks)
    monkeypatch.setattr(twh, "get_graph_output_dir", lambda: tmp_path)

    def fake_build_dependency_edges(tasks_arg, reverse=False):
        assert reverse is expect_reverse
        return [], {"a": tasks_arg[0]}

    monkeypatch.setattr(graph_module, "build_dependency_edges", fake_build_dependency_edges)
    monkeypatch.setattr(graph_module, "generate_dot", lambda *args, **kwargs: "digraph twh {}")

    calls = {}

    def fake_render(dot_source, png_path, svg_path):
        calls["render"] = (dot_source, png_path, svg_path)
        return True, None

    monkeypatch.setattr(graph_module, "render_graphviz", fake_render)

    ascii_called = {"value": False}

    def fake_ascii(*args, **kwargs):
        ascii_called["value"] = True
        return ["ASCII"]

    monkeypatch.setattr(graph_module, "ascii_forest", fake_ascii)

    opened = {}

    def fake_open_svg(path):
        opened["svg"] = path

    monkeypatch.setattr(renderer, "open_in_browser", fake_open_svg)
    monkeypatch.setattr(renderer, "open_file", lambda path: opened.setdefault("png", path))

    if mode:
        run_graph(mode=mode)
    else:
        run_graph()

    captured = capsys.readouterr()
    expected_svg = tmp_path / "tasks-graph.svg"
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
def test_graph_falls_back_to_ascii_when_render_fails(
    monkeypatch,
    tmp_path,
    capsys,
    render_error,
):
    """
    Ensure graph prints ASCII output if rendering fails.

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
    def run_graph():
        GRAPH_COMMAND(
            mode=None,
            reverse=False,
            png=None,
            svg=None,
            ascii_only=False,
            edges=False,
            rankdir="LR",
        )

    tasks = [{"uuid": "a", "description": "Task A"}]
    monkeypatch.setattr(twh, "get_tasks_from_taskwarrior", lambda: tasks)
    monkeypatch.setattr(twh, "get_graph_output_dir", lambda: tmp_path)
    monkeypatch.setattr(graph_module, "build_dependency_edges", lambda tasks_arg, reverse=False: ([], {"a": tasks_arg[0]}))
    monkeypatch.setattr(graph_module, "generate_dot", lambda *args, **kwargs: "digraph twh {}")
    monkeypatch.setattr(graph_module, "render_graphviz", lambda *args, **kwargs: (False, render_error))
    monkeypatch.setattr(graph_module, "ascii_forest", lambda *args, **kwargs: ["ASCII fallback"])

    def unexpected_open(*_args, **_kwargs):
        raise AssertionError("Open should not be called when rendering fails.")

    monkeypatch.setattr(renderer, "open_in_browser", unexpected_open)
    monkeypatch.setattr(renderer, "open_file", unexpected_open)

    run_graph()

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
        ["twh", "ondeck"],
        ["twh", "option"],
        ["twh", "dominance"],
        ["twh", "calibrate"],
        ["twh", "diagnose"],
        ["twh", "start"],
        ["twh", "stop"],
        ["twh", "time"],
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
    called = {"app": 0}

    def fake_build_app():
        def fake_app():
            called["app"] += 1
        return fake_app

    def unexpected_exec(_args):
        raise AssertionError("Delegation should not occur for twh commands.")

    monkeypatch.setattr(twh, "build_app", fake_build_app)
    monkeypatch.setattr(twh, "exec_task_command", unexpected_exec)
    monkeypatch.setattr(sys, "argv", argv)

    twh.main()

    assert called["app"] == 1


@pytest.mark.parametrize(
    ("argv", "expected_args"),
    [
        (["twh", "simple"], []),
        (["twh", "simple", "+work"], ["+work"]),
    ],
)
@pytest.mark.unit
def test_main_fast_path_simple(monkeypatch, argv, expected_args):
    """
    Ensure twh simple bypasses the Typer app and runs directly.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching dependencies.
    argv : list[str]
        Full argv including the program name.
    expected_args : list[str]
        Expected filters passed into the simple report runner.

    Returns
    -------
    None
        This test asserts that the simple fast path is used.
    """
    calls = []

    def fake_run_simple(args):
        calls.append(args)
        return 0

    def unexpected_build_app():
        raise AssertionError("Typer app should not be built for simple fast path.")

    monkeypatch.setattr(twh, "run_simple_report", fake_run_simple)
    monkeypatch.setattr(twh, "build_app", unexpected_build_app)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        twh.main()

    assert excinfo.value.code == 0
    assert calls == [expected_args]


@pytest.mark.unit
def test_main_simple_help_uses_app(monkeypatch):
    """
    Ensure twh simple --help uses the Typer app.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching app creation.

    Returns
    -------
    None
        This test asserts help routing for the simple command.
    """
    called = {"app": 0}

    def fake_build_app():
        def fake_app():
            called["app"] += 1
        return fake_app

    def unexpected_simple(_args):
        raise AssertionError("Simple fast path should be skipped for help.")

    monkeypatch.setattr(twh, "build_app", fake_build_app)
    monkeypatch.setattr(twh, "run_simple_report", unexpected_simple)
    monkeypatch.setattr(sys, "argv", ["twh", "simple", "--help"])

    twh.main()

    assert called["app"] == 1


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

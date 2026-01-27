"""
Unit tests for the Graphviz-based graph2 module.
"""

import doctest
from pathlib import Path

import pytest

import twh.graph2 as graph2


@pytest.mark.parametrize(
    ("reverse", "expected_edges"),
    [
        (
            False,
            {
                ("a", "b"),
                ("a", "c"),
                ("b", "c"),
            },
        ),
        (
            True,
            {
                ("b", "a"),
                ("c", "a"),
                ("c", "b"),
            },
        ),
    ],
)
@pytest.mark.unit
def test_build_dependency_edges(reverse, expected_edges):
    """
    Verify edge direction for dependency parsing.

    Parameters
    ----------
    reverse : bool
        Whether to reverse edge direction.
    expected_edges : set[tuple[str, str]]
        Expected edge tuples.

    Returns
    -------
    None
        This test asserts on edge construction.
    """
    tasks = [
        {"uuid": "a", "id": 1, "description": "Task A", "depends": "b, c"},
        {"uuid": "b", "id": 2, "description": "Task B", "depends": ["c"]},
        {"uuid": "c", "id": 3, "description": "Task C"},
        {"uuid": "d", "id": 4, "description": "Task D", "depends": "missing"},
    ]

    edges, by_uuid = graph2.build_dependency_edges(tasks, reverse=reverse)

    assert set(by_uuid.keys()) == {"a", "b", "c", "d"}
    assert set(edges) == expected_edges


@pytest.mark.parametrize(
    ("tasks", "expected_lines"),
    [
        (
            [
                {
                    "uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "id": 1,
                    "description": "Task A",
                },
                {
                    "uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "id": 2,
                    "description": "Task B",
                    "depends": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                },
                {
                    "uuid": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                    "id": 3,
                    "description": "Task C",
                    "depends": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                },
            ],
            [
                "[3] Task C [cccccccc]",
                "  - [2] Task B [bbbbbbbb]",
                "    - [1] Task A [aaaaaaaa]",
            ],
        ),
    ],
)
@pytest.mark.unit
def test_ascii_forest_outputs(tasks, expected_lines):
    """
    Ensure ASCII output is stable and ordered.

    Parameters
    ----------
    tasks : list[dict]
        Task payloads.
    expected_lines : list[str]
        Expected ASCII output.

    Returns
    -------
    None
        This test asserts on ASCII formatting.
    """
    edges, by_uuid = graph2.build_dependency_edges(tasks, reverse=False)
    lines = graph2.ascii_forest(edges, by_uuid)

    assert lines == expected_lines


@pytest.mark.parametrize(
    ("edges", "expected_substrings"),
    [
        (
            [("a", "b")],
            [
                "digraph twh",
                "rankdir=LR",
                "\"a\" -> \"b\"",
            ],
        ),
    ],
)
@pytest.mark.unit
def test_generate_dot_contains_edges(edges, expected_substrings):
    """
    Confirm DOT output includes graph structure.

    Parameters
    ----------
    edges : list[tuple[str, str]]
        Edge list.
    expected_substrings : list[str]
        Required output fragments.

    Returns
    -------
    None
        This test asserts on DOT content.
    """
    by_uuid = {
        "a": {"uuid": "a", "description": "Task A"},
        "b": {"uuid": "b", "description": "Task B"},
    }

    dot_source = graph2.generate_dot(by_uuid, edges, rankdir="LR")

    for fragment in expected_substrings:
        assert fragment in dot_source


@pytest.mark.unit
def test_render_graphviz_returns_false_when_dot_missing(monkeypatch, tmp_path):
    """
    Render should fail gracefully when dot is unavailable.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching Graphviz lookup.
    tmp_path : pathlib.Path
        Temporary directory for output paths.

    Returns
    -------
    None
        This test asserts on Graphviz fallback behavior.
    """
    monkeypatch.setattr(graph2.shutil, "which", lambda _: None)

    png_path = tmp_path / "graph.png"
    svg_path = tmp_path / "graph.svg"
    rendered, error = graph2.render_graphviz("digraph twh {}", png_path, svg_path)

    assert rendered is False
    assert error


@pytest.mark.unit
def test_graph2_doctest_examples():
    """
    Run doctest examples embedded in graph2 docstrings.

    Returns
    -------
    None
        This test asserts that doctest examples succeed.
    """
    results = doctest.testmod(graph2)
    assert results.failed == 0


@pytest.mark.parametrize(
    ("platform", "dot_path", "expect_conversion"),
    [
        ("cygwin", "/cygdrive/c/Graphviz/bin/dot.exe", True),
        ("cygwin", "/cygdrive/c/Users/aarre/scoop/shims/dot", True),
        ("cygwin", "/usr/bin/dot", False),
        ("win32", "C:\\Graphviz\\bin\\dot.exe", False),
    ],
)
@pytest.mark.unit
def test_dot_output_path_conversion(monkeypatch, platform, dot_path, expect_conversion):
    """
    Ensure DOT output paths are converted for Windows Graphviz on Cygwin.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching platform behavior.
    platform : str
        sys.platform value to simulate.
    dot_path : str
        Dot executable path to evaluate.
    expect_conversion : bool
        Whether conversion should be applied.

    Returns
    -------
    None
        This test asserts on Cygwin path conversion behavior.
    """
    output_path = Path("/tmp/tasks-graph2.svg")
    monkeypatch.setattr(graph2.sys, "platform", platform)

    if expect_conversion:
        monkeypatch.setattr(
            graph2,
            "_cygwin_to_windows_path",
            lambda path: "C:\\cygwin64\\tmp\\tasks-graph2.svg",
        )
        expected = "C:\\cygwin64\\tmp\\tasks-graph2.svg"
    else:
        def unexpected_call(_path):
            raise AssertionError("Conversion should not be called.")

        monkeypatch.setattr(graph2, "_cygwin_to_windows_path", unexpected_call)
        expected = str(output_path)

    assert graph2._dot_output_path(output_path, dot_path) == expected

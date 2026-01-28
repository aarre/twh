"""
Unit tests for renderer utilities.
"""

import doctest
import subprocess

import pytest

import twh.renderer as renderer


@pytest.mark.unit
def test_open_in_browser_uses_edge_on_wsl(
    monkeypatch,
    tmp_path,
):
    """
    Ensure WSL launches Edge using a Windows-compatible file URL.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching subprocess and environment.
    tmp_path : pathlib.Path
        Temporary directory for the SVG file.
    Returns
    -------
    None
        This test asserts on WSL-specific browser invocation.
    """
    svg_path = tmp_path / "tasks-graph.svg"
    svg_path.write_text("<svg></svg>", encoding="utf-8")

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    windows_path = r"C:\Users\aarre\AppData\Local\Temp\tasks-graph.svg"
    expected_url = "file:///C:/Users/aarre/AppData/Local/Temp/tasks-graph.svg"

    edge_exe = tmp_path / "msedge.exe"
    edge_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(renderer, "_wsl_to_windows_path", lambda path: windows_path)
    monkeypatch.setattr(renderer, "_windows_path_to_file_url", lambda path: expected_url)
    monkeypatch.setattr(renderer, "_find_edge_executable", lambda: edge_exe)

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("UNC fallback should not be used for Windows paths.")

    monkeypatch.setattr(renderer, "_get_windows_temp_dir", unexpected_call)
    monkeypatch.setattr(renderer, "_windows_to_wsl_path", unexpected_call)
    monkeypatch.setattr(renderer.shutil, "copy2", unexpected_call)

    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)

    renderer.open_in_browser(svg_path)

    assert calls[0][0] == [str(edge_exe), expected_url]


@pytest.mark.unit
def test_open_in_browser_copies_unc_path_on_wsl(monkeypatch, tmp_path):
    """
    Ensure UNC paths are copied to Windows temp before opening Edge.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching subprocess and environment.
    tmp_path : pathlib.Path
        Temporary directory for the SVG file.

    Returns
    -------
    None
        This test asserts on UNC fallback behavior.
    """
    svg_path = tmp_path / "tasks-graph.svg"
    svg_path.write_text("<svg></svg>", encoding="utf-8")

    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")

    unc_path = r"\\wsl.localhost\Ubuntu-24.04\tmp\tasks-graph.svg"
    windows_temp = r"C:\Users\aarre\AppData\Local\Temp"
    windows_target = r"C:\Users\aarre\AppData\Local\Temp\tasks-graph.svg"
    expected_url = "file:///C:/Users/aarre/AppData/Local/Temp/tasks-graph.svg"
    windows_temp_wsl = tmp_path / "win-temp"
    windows_temp_wsl.mkdir()

    path_calls = {"count": 0}

    def fake_wsl_to_windows_path(path):
        path_calls["count"] += 1
        return unc_path if path_calls["count"] == 1 else windows_target

    edge_exe = tmp_path / "msedge.exe"
    edge_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(renderer, "_wsl_to_windows_path", fake_wsl_to_windows_path)
    monkeypatch.setattr(renderer, "_get_windows_temp_dir", lambda: windows_temp)
    monkeypatch.setattr(renderer, "_windows_to_wsl_path", lambda _: windows_temp_wsl)
    monkeypatch.setattr(renderer, "_windows_path_to_file_url", lambda _: expected_url)
    monkeypatch.setattr(renderer, "_find_edge_executable", lambda: edge_exe)

    copied = {}

    def fake_copy(src, dst):
        copied["src"] = src
        copied["dst"] = dst
        return dst

    monkeypatch.setattr(renderer.shutil, "copy2", fake_copy)

    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)

    renderer.open_in_browser(svg_path)

    assert copied["src"] == svg_path
    assert copied["dst"] == windows_temp_wsl / svg_path.name
    assert calls[0][0] == [str(edge_exe), expected_url]


@pytest.mark.unit
def test_renderer_doctest_examples():
    """
    Run doctest examples embedded in renderer docstrings.

    Returns
    -------
    None
        This test asserts that doctest examples succeed.
    """
    results = doctest.testmod(renderer)
    assert results.failed == 0

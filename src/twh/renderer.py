#!/usr/bin/env python3
"""
Cross-platform file opening utilities.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Union


def open_file(file_path: Union[str, Path]) -> None:
    """
    Open a file in the default system viewer across different platforms.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file to open.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If opening the file fails.

    Notes
    -----
    Supports:
    - Windows (including Cygwin): uses cygstart or start
    - macOS: uses open
    - Linux/Unix: uses xdg-open
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    system = platform.system().lower()

    try:
        if 'cygwin' in os.environ.get('TERM', '').lower() or \
           os.path.exists('/usr/bin/cygstart'):
            subprocess.run(['cygstart', str(file_path)], check=True)
        elif system == 'windows' or sys.platform == 'win32':
            if hasattr(os, 'startfile'):
                os.startfile(str(file_path))
            else:
                subprocess.run(['start', str(file_path)], shell=True, check=True)
        elif system == 'darwin':
            subprocess.run(['open', str(file_path)], check=True)
        else:
            subprocess.run(['xdg-open', str(file_path)], check=True)
    except Exception as e:
        raise RuntimeError(f"Failed to open file {file_path}: {e}")


def _is_wsl() -> bool:
    """
    Detect whether the current environment is WSL.

    Returns
    -------
    bool
        True when running inside WSL, otherwise False.
    """
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    if os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"):
        return True
    try:
        return "microsoft" in platform.release().lower()
    except OSError:
        return False


def _wsl_to_windows_path(file_path: Path) -> str:
    """
    Convert a WSL path to a Windows path using wslpath.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to convert.

    Returns
    -------
    str
        Windows-style path.
    """
    result = subprocess.run(
        ["wslpath", "-w", str(file_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _windows_path_to_file_url(windows_path: str) -> str:
    """
    Build a file:// URL from a Windows path.

    Parameters
    ----------
    windows_path : str
        Windows-style path.

    Returns
    -------
    str
        File URL suitable for browser launch.

    Examples
    --------
    >>> _windows_path_to_file_url(r"C:\\Users\\A\\file.svg")
    'file:///C:/Users/A/file.svg'
    >>> _windows_path_to_file_url(r"\\\\wsl$\\Ubuntu\\tmp\\file.svg")
    'file://wsl$/Ubuntu/tmp/file.svg'
    """
    path = windows_path.replace("\\", "/")
    if path.startswith("//"):
        return f"file:{path}"
    return f"file:///{path}"


def _is_unc_windows_path(windows_path: str) -> bool:
    """
    Determine whether a Windows path is a UNC path.

    Parameters
    ----------
    windows_path : str
        Windows-style path.

    Returns
    -------
    bool
        True when the path is a UNC path.
    """
    return windows_path.startswith("\\\\") or windows_path.startswith("//")


def _get_windows_temp_dir() -> str:
    """
    Resolve the Windows TEMP directory from WSL.

    Returns
    -------
    str
        Windows-style path for the TEMP directory.
    """
    cmd = "cd /d C:\\\\Windows && echo %TEMP%"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", cmd],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _windows_to_wsl_path(windows_path: str) -> Path:
    """
    Convert a Windows path to a WSL path using wslpath.

    Parameters
    ----------
    windows_path : str
        Windows-style path.

    Returns
    -------
    pathlib.Path
        WSL path equivalent.
    """
    result = subprocess.run(
        ["wslpath", "-u", windows_path],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _find_edge_executable() -> Path | None:
    """
    Locate the Windows Edge executable from WSL.

    Returns
    -------
    pathlib.Path | None
        Path to msedge.exe when found, otherwise None.
    """
    candidates = [
        Path("/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path("/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    try:
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "where msedge.exe"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            wsl_path = _windows_to_wsl_path(line)
        except Exception:
            continue
        if wsl_path.exists():
            return wsl_path
    return None


def open_in_browser(file_path: Union[str, Path]) -> None:
    """
    Open a local file in the user's default web browser.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file to open.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If opening the file fails.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    system = platform.system().lower()
    try:
        if _is_wsl():
            windows_path = _wsl_to_windows_path(file_path)
            if _is_unc_windows_path(windows_path):
                windows_temp = _get_windows_temp_dir()
                wsl_temp = _windows_to_wsl_path(windows_temp)
                wsl_temp.mkdir(parents=True, exist_ok=True)
                target_path = wsl_temp / file_path.name
                if target_path.resolve() != file_path:
                    shutil.copy2(file_path, target_path)
                windows_path = _wsl_to_windows_path(target_path)
            url = _windows_path_to_file_url(windows_path)
            edge_exe = _find_edge_executable()
            if edge_exe:
                subprocess.run([str(edge_exe), url], check=True)
            else:
                cmd = f'start "" "microsoft-edge:{url}"'
                subprocess.run(
                    ["cmd.exe", "/d", "/c", cmd],
                    cwd="/mnt/c/Windows",
                    check=True,
                )
        else:
            url = file_path.as_uri()
            if sys.platform.startswith('cygwin') or os.path.exists('/usr/bin/cygstart'):
                subprocess.run(['cygstart', url], check=True)
            elif system == 'windows' or sys.platform == 'win32':
                if hasattr(os, 'startfile'):
                    os.startfile(url)
                else:
                    subprocess.run(['cmd', '/c', 'start', '', url], check=True)
            elif system == 'darwin':
                subprocess.run(['open', url], check=True)
            else:
                subprocess.run(['xdg-open', url], check=True)
    except Exception as e:
        raise RuntimeError(f"Failed to open browser for {file_path}: {e}")

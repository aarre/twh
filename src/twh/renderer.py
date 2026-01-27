#!/usr/bin/env python3
"""
Cross-platform file opening utilities.
"""

import os
import platform
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
    url = file_path.as_uri()
    system = platform.system().lower()
    try:
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

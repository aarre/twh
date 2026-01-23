#!/usr/bin/env python3
"""
Mermaid diagram rendering to PNG and file opening utilities.

This module provides functionality to render Mermaid diagrams to PNG using
pyppeteer (headless Chrome) and to open files in the default viewer across
different platforms.
"""

import asyncio
import atexit
import os
import platform
import subprocess
import sys
import webbrowser
from pathlib import Path
from shutil import which
from typing import Union
import tempfile


def render_mermaid_to_png(mermaid_file: Path, output_file: Path) -> None:
    """
    Render a Mermaid diagram file to PNG using pyppeteer.

    Parameters
    ----------
    mermaid_file : Path
        Path to the Mermaid (.mmd) file.
    output_file : Path
        Path where the PNG file should be saved.

    Raises
    ------
    FileNotFoundError
        If the mermaid_file does not exist.
    RuntimeError
        If rendering fails.
    """
    if not mermaid_file.exists():
        raise FileNotFoundError(f"Mermaid file not found: {mermaid_file}")

    # Read mermaid content
    mermaid_content = mermaid_file.read_text(encoding='utf-8')

    # Run async rendering with a dedicated loop to avoid atexit warnings.
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_render_mermaid_async(mermaid_content, output_file))
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def render_mermaid_to_svg(mermaid_file: Path, output_file: Path) -> None:
    """
    Render a Mermaid diagram file to SVG using pyppeteer.

    Parameters
    ----------
    mermaid_file : Path
        Path to the Mermaid (.mmd) file.
    output_file : Path
        Path where the SVG file should be saved.
    """
    if not mermaid_file.exists():
        raise FileNotFoundError(f"Mermaid file not found: {mermaid_file}")

    mermaid_content = mermaid_file.read_text(encoding='utf-8')
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_render_mermaid_svg_async(mermaid_content, output_file))
    finally:
        asyncio.set_event_loop(None)
        loop.close()


async def _render_mermaid_async(mermaid_content: str, output_file: Path) -> None:
    """
    Asynchronously render Mermaid diagram to PNG.

    Parameters
    ----------
    mermaid_content : str
        The Mermaid diagram definition.
    output_file : Path
        Path where the PNG file should be saved.

    Raises
    ------
    RuntimeError
        If rendering fails.
    """
    from pyppeteer import launch

    # HTML template that includes Mermaid.js from CDN
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
                flowchart: {{
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: 'basis'
                }}
            }});
        </script>
        <style>
            body {{
                margin: 20px;
                background: white;
            }}
            .mermaid {{
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
{mermaid_content}
        </div>
    </body>
    </html>
    """

    html_content = html_template.format(mermaid_content=mermaid_content)

    browser = None
    chromium_path = _detect_chromium_executable()
    user_data_dir = None
    if chromium_path and _is_windows_exe_on_cygwin(chromium_path):
        user_data_dir = _create_windows_user_data_dir()
    try:
        # Launch headless browser
        launch_kwargs = {
            'headless': True,
            'args': [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
            ],
        }
        if chromium_path:
            launch_kwargs['executablePath'] = chromium_path
        if user_data_dir:
            launch_kwargs['userDataDir'] = user_data_dir
        browser = await launch(**launch_kwargs)
        launcher = getattr(browser, "_launcher", None)
        if launcher and hasattr(atexit, "unregister"):
            try:
                atexit.unregister(launcher._close_process)
            except Exception:
                pass
        page = await browser.newPage()

        # Set viewport to a reasonable size
        await page.setViewport({'width': 1920, 'height': 1080})

        # Load HTML content
        await page.setContent(html_content)

        # Wait for Mermaid to render
        await page.waitForSelector('.mermaid svg', {'timeout': 10000})

        # Wait a bit more to ensure rendering is complete
        await asyncio.sleep(1)

        # Get the dimensions of the rendered SVG
        dimensions = await page.evaluate('''() => {
            const svg = document.querySelector('.mermaid svg');
            const bbox = svg.getBBox();
            return {
                width: Math.ceil(bbox.width + 40),
                height: Math.ceil(bbox.height + 40)
            };
        }''')

        # Set viewport to fit the diagram
        await page.setViewport({
            'width': max(800, dimensions['width']),
            'height': max(600, dimensions['height'])
        })

        # Take screenshot of the mermaid div
        element = await page.querySelector('.mermaid')
        await element.screenshot({'path': str(output_file), 'omitBackground': False})

    except Exception as e:
        exe_hint = f" (executable: {chromium_path})" if chromium_path else ""
        raise RuntimeError(f"Failed to render Mermaid diagram: {e}{exe_hint}")
    finally:
        if browser:
            await browser.close()


async def _render_mermaid_svg_async(mermaid_content: str, output_file: Path) -> None:
    """
    Asynchronously render Mermaid diagram to SVG.
    """
    from pyppeteer import launch

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
                flowchart: {{
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: 'basis'
                }}
            }});
        </script>
        <style>
            body {{
                margin: 20px;
                background: white;
            }}
            .mermaid {{
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
{mermaid_content}
        </div>
    </body>
    </html>
    """

    html_content = html_template.format(mermaid_content=mermaid_content)
    browser = None
    chromium_path = _detect_chromium_executable()
    user_data_dir = None
    if chromium_path and _is_windows_exe_on_cygwin(chromium_path):
        user_data_dir = _create_windows_user_data_dir()
    try:
        launch_kwargs = {
            'headless': True,
            'args': [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
            ],
        }
        if chromium_path:
            launch_kwargs['executablePath'] = chromium_path
        if user_data_dir:
            launch_kwargs['userDataDir'] = user_data_dir
        browser = await launch(**launch_kwargs)
        launcher = getattr(browser, "_launcher", None)
        if launcher and hasattr(atexit, "unregister"):
            try:
                atexit.unregister(launcher._close_process)
            except Exception:
                pass
        page = await browser.newPage()
        await page.setViewport({'width': 1920, 'height': 1080})
        await page.setContent(html_content)
        await page.waitForSelector('.mermaid svg', {'timeout': 10000})
        await asyncio.sleep(0.5)
        svg_markup = await page.evaluate('''() => {
            const svg = document.querySelector('.mermaid svg');
            return svg ? svg.outerHTML : null;
        }''')
        if not svg_markup:
            raise RuntimeError("Mermaid SVG not found after rendering.")
        output_file.write_text(svg_markup, encoding='utf-8')
    except Exception as e:
        exe_hint = f" (executable: {chromium_path})" if chromium_path else ""
        raise RuntimeError(f"Failed to render Mermaid diagram: {e}{exe_hint}")
    finally:
        if browser:
            await browser.close()


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
        # Detect Cygwin environment
        if 'cygwin' in os.environ.get('TERM', '').lower() or \
           os.path.exists('/usr/bin/cygstart'):
            # Cygwin - use cygstart
            subprocess.run(['cygstart', str(file_path)], check=True)
        elif system == 'windows' or sys.platform == 'win32':
            # Windows - use os.startfile or start command
            if hasattr(os, 'startfile'):
                os.startfile(str(file_path))
            else:
                subprocess.run(['start', str(file_path)], shell=True, check=True)
        elif system == 'darwin':
            # macOS - use open
            subprocess.run(['open', str(file_path)], check=True)
        else:
            # Linux/Unix - use xdg-open
            subprocess.run(['xdg-open', str(file_path)], check=True)
    except Exception as e:
        raise RuntimeError(f"Failed to open file {file_path}: {e}")


def open_in_browser(file_path: Union[str, Path]) -> None:
    """
    Open a local file in the user's default web browser.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    webbrowser.open(file_path.as_uri())


def _detect_chromium_executable() -> Union[str, None]:
    """
    Find a Chromium/Chrome executable across common environments.
    """
    env_path = (
        os.environ.get('TWH_CHROMIUM_PATH')
        or os.environ.get('PYPPETEER_EXECUTABLE_PATH')
        or os.environ.get('CHROME_PATH')
    )
    if env_path:
        env_path = str(Path(env_path))
        if Path(env_path).exists():
            return env_path
        raise RuntimeError(
            f"Chromium executable not found at {env_path}. "
            "Set TWH_CHROMIUM_PATH to a valid browser path."
        )

    candidates = []
    system = platform.system().lower()
    is_cygwin = sys.platform.startswith('cygwin')
    is_wsl = 'microsoft' in platform.release().lower()

    if system == 'windows' or sys.platform == 'win32':
        local_app = os.environ.get('LOCALAPPDATA', r'C:\Users\%USERNAME%\AppData\Local')
        prog_files = os.environ.get('PROGRAMFILES', r'C:\Program Files')
        prog_files_x86 = os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)')
        candidates.extend([
            str(Path(prog_files) / 'Google/Chrome/Application/chrome.exe'),
            str(Path(prog_files_x86) / 'Google/Chrome/Application/chrome.exe'),
            str(Path(local_app) / 'Google/Chrome/Application/chrome.exe'),
            str(Path(prog_files) / 'Chromium/Application/chrome.exe'),
        ])
    elif is_cygwin:
        candidates.extend([
            '/cygdrive/c/Program Files/Google/Chrome/Application/chrome.exe',
            '/cygdrive/c/Program Files (x86)/Google/Chrome/Application/chrome.exe',
            '/usr/bin/chromium',
            '/usr/bin/chromium-browser',
            '/usr/bin/google-chrome',
        ])
    elif is_wsl:
        candidates.extend([
            '/usr/bin/chromium',
            '/usr/bin/chromium-browser',
            '/usr/bin/google-chrome',
            '/snap/bin/chromium',
            '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
            '/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe',
        ])
    else:
        candidates.extend([
            '/usr/bin/chromium',
            '/usr/bin/chromium-browser',
            '/usr/bin/google-chrome',
            '/snap/bin/chromium',
        ])

    for name in ['chromium', 'chromium-browser', 'google-chrome', 'chrome']:
        found = which(name)
        if found:
            return found

    for path in candidates:
        if Path(path).exists():
            return path

    return None


def _is_windows_exe_on_cygwin(executable_path: str) -> bool:
    return sys.platform.startswith('cygwin') and executable_path.lower().endswith('.exe')


def _create_windows_user_data_dir() -> str:
    """
    Create a temp user data directory with a Windows path for Chrome on Cygwin.
    """
    base = os.environ.get('LOCALAPPDATA') or os.environ.get('TEMP') or os.environ.get('TMP')
    if base and Path(base).exists():
        cyg_base = Path(base)
    else:
        cyg_base = Path('/cygdrive/c/Windows/Temp')
    cyg_base.mkdir(parents=True, exist_ok=True)
    cyg_dir = Path(tempfile.mkdtemp(prefix='twh-chrome-', dir=str(cyg_base)))
    return _cygwin_to_windows_path(str(cyg_dir))


def _cygwin_to_windows_path(path_str: str) -> str:
    """
    Convert /cygdrive/<drive>/path to <Drive>:\\path for Windows executables.
    """
    if path_str.startswith('/cygdrive/'):
        parts = Path(path_str).parts
        if len(parts) >= 4 and parts[1] == 'cygdrive':
            drive = parts[2].upper()
            rest = '\\'.join(parts[3:])
            return f"{drive}:\\{rest}"
    return path_str

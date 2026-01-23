#!/usr/bin/env python3
"""
Mermaid diagram rendering to PNG and file opening utilities.

This module provides functionality to render Mermaid diagrams to PNG using
pyppeteer (headless Chrome) and to open files in the default viewer across
different platforms.
"""

import asyncio
import os
import platform
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Union


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

    # Run async rendering
    asyncio.run(_render_mermaid_async(mermaid_content, output_file))


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
    try:
        # Launch headless browser
        launch_kwargs = {
            'headless': True,
            'args': ['--no-sandbox', '--disable-setuid-sandbox'],
        }
        if chromium_path:
            launch_kwargs['executablePath'] = chromium_path
        browser = await launch(**launch_kwargs)
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
        raise RuntimeError(f"Failed to render Mermaid diagram: {e}")
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
            '/usr/bin/chromium',
            '/usr/bin/chromium-browser',
            '/usr/bin/google-chrome',
            '/cygdrive/c/Program Files/Google/Chrome/Application/chrome.exe',
            '/cygdrive/c/Program Files (x86)/Google/Chrome/Application/chrome.exe',
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

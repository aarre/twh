"""
Manage known move modes and prompt helpers.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

DEFAULT_MODES = [
    "analysis",
    "research",
    "writing",
    "editorial",
    "illustration",
    "programming",
    "teaching",
    "chore",
    "errand",
]
MODE_ENV_VAR = "TWH_MODES_PATH"


def normalize_mode_value(value: Optional[str]) -> str:
    """
    Normalize a mode value for storage and comparison.

    Parameters
    ----------
    value : Optional[str]
        Raw mode value.

    Returns
    -------
    str
        Normalized mode string.

    Examples
    --------
    >>> normalize_mode_value("  Writing ")
    'writing'
    >>> normalize_mode_value(None)
    ''
    """
    if value is None:
        return ""
    return str(value).strip().lower()


def get_modes_path() -> Path:
    """
    Return the path for persisting known modes.

    Returns
    -------
    Path
        Resolved path for the modes JSON file.
    """
    env_path = os.environ.get(MODE_ENV_VAR)
    if env_path:
        return Path(os.path.expandvars(os.path.expanduser(env_path)))
    return Path.home() / ".config" / "twh" / "modes.json"


def _dedupe_modes(modes: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for mode in modes:
        normalized = normalize_mode_value(mode)
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def _split_taskwarrior_values(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]


def load_known_modes(path: Optional[Path] = None) -> List[str]:
    """
    Load known modes from disk, falling back to defaults.

    Parameters
    ----------
    path : Optional[Path], optional
        Optional override for the modes path.

    Returns
    -------
    List[str]
        Ordered list of known modes.
    """
    path = path or get_modes_path()
    modes: List[str] = []
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
        if isinstance(data, list):
            modes = [str(item) for item in data]
    combined = _dedupe_modes([*modes, *DEFAULT_MODES])
    return sorted(combined)


def save_known_modes(modes: Sequence[str], path: Optional[Path] = None) -> None:
    """
    Persist known modes to disk.

    Parameters
    ----------
    modes : Sequence[str]
        Modes to write.
    path : Optional[Path], optional
        Optional override for the modes path.

    Returns
    -------
    None
        Modes are written to disk.
    """
    path = path or get_modes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _dedupe_modes(modes)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def register_mode(
    mode: Optional[str],
    *,
    modes: Optional[Sequence[str]] = None,
    path: Optional[Path] = None,
) -> List[str]:
    """
    Register a mode, moving it to the front of the list.

    Parameters
    ----------
    mode : Optional[str]
        Mode value to register.
    modes : Optional[Sequence[str]], optional
        Existing mode list to update (default: load from disk).
    path : Optional[Path], optional
        Optional override for the modes path.

    Returns
    -------
    List[str]
        Updated mode list.

    Examples
    --------
    >>> register_mode("wait", modes=["analysis", "writing"])[:3]
    ['analysis', 'wait', 'writing']
    """
    normalized = normalize_mode_value(mode)
    if not normalized:
        return list(modes) if modes is not None else load_known_modes(path=path)
    current = list(modes) if modes is not None else load_known_modes(path=path)
    current = [m for m in current if normalize_mode_value(m) != normalized]
    updated = sorted(_dedupe_modes([*current, normalized]))
    save_known_modes(updated, path=path)
    return updated


def format_mode_prompt(modes: Sequence[str]) -> str:
    """
    Format the mode prompt string with examples.

    Parameters
    ----------
    modes : Sequence[str]
        Known modes to surface.

    Returns
    -------
    str
        Prompt string.
    """
    normalized = _dedupe_modes(modes)
    examples = "/".join(sorted(normalized))
    if not examples:
        examples = "/".join(sorted(_dedupe_modes(DEFAULT_MODES)))
    return f"  Mode (e.g., {examples}): "


def ensure_taskwarrior_mode_value(
    mode: Optional[str],
    *,
    get_setting: Optional[Callable[[str], Optional[str]]] = None,
    runner: Optional[Callable[..., subprocess.CompletedProcess]] = None,
) -> bool:
    """
    Ensure Taskwarrior's allowed mode values include the provided mode.

    Parameters
    ----------
    mode : Optional[str]
        Mode value to ensure.
    get_setting : Optional[Callable[[str], Optional[str]]], optional
        Getter for Taskwarrior settings.
    runner : Optional[Callable[..., subprocess.CompletedProcess]], optional
        Runner to update Taskwarrior config (args exclude ``task``).

    Returns
    -------
    bool
        True when Taskwarrior config was updated.

    Examples
    --------
    >>> ensure_taskwarrior_mode_value("", get_setting=lambda _k: None)
    False
    """
    normalized = normalize_mode_value(mode)
    if not normalized:
        return False
    if get_setting is None:
        from .taskwarrior import get_taskwarrior_setting

        get_setting = get_taskwarrior_setting
    raw_values = get_setting("uda.mode.values")
    if raw_values is None:
        return False
    existing = _split_taskwarrior_values(raw_values)
    if any(normalize_mode_value(value) == normalized for value in existing):
        return False
    updated = sorted(_dedupe_modes([*existing, normalized]))
    if runner is None:
        def task_runner(args, **kwargs):
            return subprocess.run(["task", *args], **kwargs)

        runner = task_runner
    result = runner(
        ["config", "uda.mode.values", ",".join(updated)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def best_mode_completion(prefix: str, modes: Sequence[str]) -> Optional[str]:
    """
    Return the best completion for a prefix from known modes.

    Parameters
    ----------
    prefix : str
        User-entered prefix.
    modes : Sequence[str]
        Known modes in preference order.

    Returns
    -------
    Optional[str]
        Best completion, or None when no match exists.

    Examples
    --------
    >>> best_mode_completion("wr", ["writing", "work"])
    'writing'
    >>> best_mode_completion("x", ["writing"]) is None
    True
    """
    normalized = normalize_mode_value(prefix)
    if not normalized:
        return None
    for mode in modes:
        if normalize_mode_value(mode).startswith(normalized):
            return mode
    return None


def prompt_mode_value(
    prompt: str,
    modes: Sequence[str],
    input_func: Callable[[str], str] = input,
) -> str:
    """
    Prompt for a mode with optional autocompletion.

    Parameters
    ----------
    prompt : str
        Prompt string.
    modes : Sequence[str]
        Known modes for completion.
    input_func : Callable[[str], str], optional
        Input function (default: input).

    Returns
    -------
    str
        Raw user input.
    """
    if input_func is not input:
        return input_func(prompt)
    try:
        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError:
        return input_func(prompt)

    class ModeAutoSuggest(AutoSuggest):
        def get_suggestion(self, _buffer, document):
            match = best_mode_completion(document.text, modes)
            if not match:
                return None
            if len(match) <= len(document.text):
                return None
            return Suggestion(match[len(document.text) :])

    completer = WordCompleter(list(modes), ignore_case=True, sentence=True)
    bindings = KeyBindings()

    @bindings.add("enter")
    def accept_or_submit(event):
        buffer = event.current_buffer
        suggestion = buffer.suggestion
        if suggestion:
            buffer.insert_text(suggestion.text)
        buffer.validate_and_handle()

    @bindings.add("right")
    def accept_suggestion(event):
        buffer = event.current_buffer
        suggestion = buffer.suggestion
        if suggestion:
            buffer.insert_text(suggestion.text)
        else:
            buffer.cursor_right(count=1)

    try:
        return pt_prompt(
            prompt,
            completer=completer,
            auto_suggest=ModeAutoSuggest(),
            complete_while_typing=True,
            key_bindings=bindings,
        )
    except Exception:
        return input_func(prompt)

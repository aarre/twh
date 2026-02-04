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
from functools import lru_cache

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
RESERVED_STATUS_VALUES = {
    "pending",
    "completed",
    "deleted",
    "recurring",
    "waiting",
}
FALLBACK_CORE_ATTRIBUTES = {
    "annotation",
    "annotations",
    "depends",
    "description",
    "due",
    "end",
    "entry",
    "id",
    "imask",
    "mask",
    "modified",
    "parent",
    "priority",
    "project",
    "recur",
    "scheduled",
    "start",
    "status",
    "tags",
    "template",
    "until",
    "urgency",
    "uuid",
    "wait",
}


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


def _collect_reserved_mode_values(core_attributes: Iterable[str]) -> set[str]:
    reserved = {normalize_mode_value(attr) for attr in core_attributes}
    reserved.update(RESERVED_STATUS_VALUES)
    return {value for value in reserved if value}


def _fallback_reserved_mode_values() -> set[str]:
    return _collect_reserved_mode_values(FALLBACK_CORE_ATTRIBUTES)


@lru_cache(maxsize=1)
def _default_reserved_mode_values() -> set[str]:
    from . import taskwarrior

    core = taskwarrior.get_core_attribute_names()
    if not core:
        return _fallback_reserved_mode_values()
    return _collect_reserved_mode_values(core)


def get_reserved_mode_values(
    *,
    core_attributes: Optional[Iterable[str]] = None,
) -> set[str]:
    """
    Return reserved mode values derived from core attributes/status keywords.

    Parameters
    ----------
    core_attributes : Optional[Iterable[str]]
        Optional core attribute names to use (default: query Taskwarrior).

    Returns
    -------
    set[str]
        Reserved mode values.
    """
    if core_attributes is None:
        return _default_reserved_mode_values()
    return _collect_reserved_mode_values(core_attributes)


def is_reserved_mode_value(
    value: Optional[str],
    *,
    reserved: Optional[Iterable[str]] = None,
) -> bool:
    """
    Return True when the mode value is reserved by Taskwarrior.

    Parameters
    ----------
    value : Optional[str]
        Mode value to check.
    reserved : Optional[Iterable[str]]
        Optional reserved values list.

    Returns
    -------
    bool
        True if reserved, False otherwise.
    """
    normalized = normalize_mode_value(value)
    if not normalized:
        return False
    reserved_values = (
        {normalize_mode_value(item) for item in reserved}
        if reserved is not None
        else get_reserved_mode_values()
    )
    return normalized in reserved_values


def format_reserved_mode_error(mode: str) -> str:
    """
    Format a helpful error for reserved mode values.

    Parameters
    ----------
    mode : str
        Mode value entered.

    Returns
    -------
    str
        Error message.
    """
    return (
        f"  Mode '{mode}' is reserved by Taskwarrior (core attribute/status). "
        "Please choose a different mode (for example, 'waiting')."
    )


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


def validate_taskwarrior_mode_config(
    mode: Optional[str],
    *,
    get_setting: Optional[Callable[[str], Optional[str]]] = None,
) -> None:
    """
    Validate Taskwarrior mode UDA settings for the provided mode.

    Parameters
    ----------
    mode : Optional[str]
        Mode value to validate.
    get_setting : Optional[Callable[[str], Optional[str]]], optional
        Getter for Taskwarrior settings.

    Returns
    -------
    None
        Validation raises on misconfiguration.

    Raises
    ------
    RuntimeError
        If Taskwarrior cannot store the provided mode value.

    Examples
    --------
    >>> validate_taskwarrior_mode_config(
    ...     "analysis",
    ...     get_setting=lambda key: "string" if key == "uda.mode.type" else None,
    ... )
    """
    normalized = normalize_mode_value(mode)
    if not normalized:
        return
    if is_reserved_mode_value(normalized):
        raise RuntimeError(format_reserved_mode_error(normalized).strip())
    if get_setting is None:
        from .taskwarrior import get_taskwarrior_setting

        get_setting = get_taskwarrior_setting
    mode_type = normalize_mode_value(get_setting("uda.mode.type"))
    if mode_type and mode_type != "string":
        raise RuntimeError(
            "Taskwarrior uda.mode.type must be 'string' to store mode values. "
            f"Found '{mode_type}'. Update your active Taskwarrior config and rerun."
        )
    raw_values = get_setting("uda.mode.values")
    if raw_values:
        existing = _split_taskwarrior_values(raw_values)
        if not any(normalize_mode_value(value) == normalized for value in existing):
            allowed = ", ".join(existing)
            suffix = f" Allowed values: {allowed}." if allowed else ""
            raise RuntimeError(
                f"Mode value '{normalized}' is not listed in uda.mode.values."
                f"{suffix} Add it to your active Taskwarrior config and rerun."
            )


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

#!/usr/bin/env python3
"""
Persist and load calibration weights for twh scoring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import tomllib

CALIBRATION_VERSION = 1

PRECEDENCE_WEIGHT_KEYS = ("enablement", "blocker", "difficulty", "dependency")
OPTION_VALUE_WEIGHT_KEYS = (
    "bias",
    "w_children",
    "w_desc",
    "w_desc_value",
    "w_diversity",
    "w_info",
    "w_rev",
    "w_cost",
)


@dataclass(frozen=True)
class CalibrationSection:
    """
    Calibration section containing weights and metadata.

    Attributes
    ----------
    weights : Dict[str, float]
        Weight values for the section.
    meta : Dict[str, Any]
        Metadata about calibration (samples, alpha, ridge, etc.).
    """

    weights: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationData:
    """
    Parsed calibration data.

    Attributes
    ----------
    precedence : Optional[CalibrationSection]
        Precedence calibration section, if present.
    option_value : Optional[CalibrationSection]
        Option value calibration section, if present.
    """

    precedence: Optional[CalibrationSection] = None
    option_value: Optional[CalibrationSection] = None


def get_calibration_path() -> Path:
    """
    Return the calibration file path.

    Returns
    -------
    Path
        Calibration TOML path.

    Examples
    --------
    >>> isinstance(get_calibration_path(), Path)
    True
    """
    override = os.environ.get("TWH_CALIBRATION_PATH", "").strip()
    if override:
        return Path(os.path.expandvars(os.path.expanduser(override)))
    return Path.home() / ".config" / "twh" / "calibration.toml"


def normalize_precedence_weights(
    weights: Dict[str, float],
    *,
    keys: Optional[Sequence[str]] = None,
    fallback: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Normalize precedence weights to sum to 1.0.

    Parameters
    ----------
    weights : Dict[str, float]
        Weight values to normalize.
    keys : Optional[Sequence[str]], optional
        Keys to include (defaults to weight keys in input order).
    fallback : Optional[Dict[str, float]], optional
        Fallback weights if normalization is impossible.

    Returns
    -------
    Dict[str, float]
        Normalized weights.

    Examples
    --------
    >>> normalize_precedence_weights({"enablement": 1, "blocker": 1})
    {'enablement': 0.5, 'blocker': 0.5}
    """
    if keys is None:
        keys = list(weights.keys())
    cleaned = {key: max(0.0, float(weights.get(key, 0.0))) for key in keys}
    total = sum(cleaned.values())
    if total <= 0:
        return dict(fallback) if fallback is not None else cleaned
    return {key: value / total for key, value in cleaned.items()}


def _parse_section(
    raw: Dict[str, Any],
    weight_keys: Sequence[str],
) -> Optional[CalibrationSection]:
    weights: Dict[str, float] = {}
    meta: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in weight_keys:
            try:
                weights[key] = float(value)
            except (TypeError, ValueError):
                continue
        else:
            meta[key] = value
    if not weights:
        return None
    return CalibrationSection(weights=weights, meta=meta)


def load_calibration(path: Optional[Path] = None) -> Optional[CalibrationData]:
    """
    Load calibration data from disk.

    Parameters
    ----------
    path : Optional[Path], optional
        Path to the calibration file (defaults to standard path).

    Returns
    -------
    Optional[CalibrationData]
        Parsed calibration data, or None if missing.
    """
    calibration_path = path or get_calibration_path()
    if not calibration_path.exists():
        return None
    try:
        raw_text = calibration_path.read_text(encoding="utf-8")
        parsed = tomllib.loads(raw_text)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    precedence_section = None
    option_section = None
    if isinstance(parsed.get("precedence"), dict):
        precedence_section = _parse_section(
            parsed["precedence"],
            PRECEDENCE_WEIGHT_KEYS,
        )
    if isinstance(parsed.get("option_value"), dict):
        option_section = _parse_section(
            parsed["option_value"],
            OPTION_VALUE_WEIGHT_KEYS,
        )
    if not precedence_section and not option_section:
        return None
    return CalibrationData(
        precedence=precedence_section,
        option_value=option_section,
    )


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{text}\""


def format_calibration_toml(data: CalibrationData) -> str:
    """
    Format calibration data as TOML text.

    Parameters
    ----------
    data : CalibrationData
        Calibration data to format.

    Returns
    -------
    str
        TOML text.
    """
    lines = [f"version = {CALIBRATION_VERSION}", ""]
    if data.precedence:
        lines.append("[precedence]")
        for key in PRECEDENCE_WEIGHT_KEYS:
            if key in data.precedence.weights:
                lines.append(f"{key} = {_format_toml_value(data.precedence.weights[key])}")
        for key in sorted(data.precedence.meta.keys()):
            lines.append(f"{key} = {_format_toml_value(data.precedence.meta[key])}")
        lines.append("")
    if data.option_value:
        lines.append("[option_value]")
        for key in OPTION_VALUE_WEIGHT_KEYS:
            if key in data.option_value.weights:
                lines.append(f"{key} = {_format_toml_value(data.option_value.weights[key])}")
        for key in sorted(data.option_value.meta.keys()):
            lines.append(f"{key} = {_format_toml_value(data.option_value.meta[key])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_calibration(
    data: CalibrationData,
    *,
    path: Optional[Path] = None,
) -> Path:
    """
    Save calibration data to disk.

    Parameters
    ----------
    data : CalibrationData
        Calibration data to save.
    path : Optional[Path], optional
        Destination path (defaults to standard path).

    Returns
    -------
    Path
        Path where the calibration data was saved.
    """
    calibration_path = path or get_calibration_path()
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(
        format_calibration_toml(data),
        encoding="utf-8",
    )
    return calibration_path


def load_precedence_weights(
    defaults: Dict[str, float],
    *,
    path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Load precedence weights, falling back to defaults.

    Parameters
    ----------
    defaults : Dict[str, float]
        Default weights to use when no calibration is present.
    path : Optional[Path], optional
        Calibration file path override.

    Returns
    -------
    Dict[str, float]
        Precedence weights.
    """
    data = load_calibration(path=path)
    if not data or not data.precedence:
        return dict(defaults)
    weights = dict(defaults)
    for key, value in data.precedence.weights.items():
        if key in weights:
            weights[key] = float(value)
    return normalize_precedence_weights(
        weights,
        keys=list(weights.keys()),
        fallback=defaults,
    )


def load_option_value_weights(
    defaults: Dict[str, float],
    *,
    path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Load option value weights, falling back to defaults.

    Parameters
    ----------
    defaults : Dict[str, float]
        Default weights to use when no calibration is present.
    path : Optional[Path], optional
        Calibration file path override.

    Returns
    -------
    Dict[str, float]
        Option value weights.
    """
    data = load_calibration(path=path)
    if not data or not data.option_value:
        return dict(defaults)
    weights = dict(defaults)
    for key, value in data.option_value.weights.items():
        if key in weights:
            weights[key] = float(value)
    return weights

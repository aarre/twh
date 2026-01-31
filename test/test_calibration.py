"""
Tests for calibration storage and weight tuning.
"""

from __future__ import annotations

import math

import pytest

import twh.calibration as calibration
import twh.calibrate as calibrate


@pytest.mark.unit
def test_calibration_save_and_load_round_trip(tmp_path):
    """
    Ensure calibration files round-trip precedence and option weights.

    Returns
    -------
    None
        This test asserts calibration serialization.
    """
    data = calibration.CalibrationData(
        precedence=calibration.CalibrationSection(
            weights={
                "enablement": 0.35,
                "blocker": 0.30,
                "difficulty": 0.20,
                "dependency": 0.15,
            },
            meta={"alpha": 0.15, "samples": 8},
        ),
        option_value=calibration.CalibrationSection(
            weights={
                "bias": 1.0,
                "w_children": 2.0,
                "w_desc": 1.0,
                "w_desc_value": 1.2,
                "w_diversity": 0.8,
                "w_info": 1.2,
                "w_rev": 1.0,
                "w_cost": 0.6,
            },
            meta={"ridge": 1.0, "samples": 3},
        ),
    )
    path = tmp_path / "calibration.toml"

    calibration.save_calibration(data, path=path)
    loaded = calibration.load_calibration(path=path)

    assert loaded is not None
    assert loaded.precedence is not None
    assert loaded.option_value is not None
    assert loaded.precedence.weights == data.precedence.weights
    assert loaded.option_value.weights == data.option_value.weights
    assert loaded.precedence.meta["alpha"] == pytest.approx(0.15)
    assert loaded.option_value.meta["ridge"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "loader",
    [
        calibration.load_precedence_weights,
        calibration.load_option_value_weights,
    ],
)
@pytest.mark.unit
def test_load_calibration_weights_falls_back_to_default(tmp_path, loader):
    """
    Ensure missing calibration files return defaults unchanged.

    Returns
    -------
    None
        This test asserts default fallback behavior.
    """
    defaults = {"enablement": 1.0, "blocker": 0.0}

    loaded = loader(defaults, path=tmp_path / "missing.toml")

    assert loaded == defaults


@pytest.mark.unit
def test_update_precedence_weights_biases_toward_choice():
    """
    Ensure calibration nudges weights toward the chosen move.

    Returns
    -------
    None
        This test asserts weight updates and normalization.
    """
    weights = {
        "enablement": 0.25,
        "blocker": 0.25,
        "difficulty": 0.25,
        "dependency": 0.25,
    }
    left = calibrate.CalibrationTask(
        uuid="a",
        move_id="1",
        description="Enablement heavy",
        enablement=1.0,
        blocker=0.0,
        difficulty=0.0,
        dependency=0.0,
        mode_multiplier=1.0,
    )
    right = calibrate.CalibrationTask(
        uuid="b",
        move_id="2",
        description="Blocker heavy",
        enablement=0.0,
        blocker=1.0,
        difficulty=0.0,
        dependency=0.0,
        mode_multiplier=1.0,
    )

    updated = calibrate.update_precedence_weights(
        weights,
        left,
        right,
        chose_left=True,
        alpha=0.2,
    )

    assert updated["enablement"] > updated["blocker"]
    assert math.isclose(sum(updated.values()), 1.0, rel_tol=1e-6)


@pytest.mark.unit
def test_calibration_doctest_examples():
    """
    Run doctest examples embedded in calibration helpers.

    Returns
    -------
    None
        This test asserts doctest coverage for calibration helpers.
    """
    import doctest

    results = doctest.testmod(calibration)
    assert results.failed == 0


@pytest.mark.unit
def test_calibrate_doctest_examples():
    """
    Run doctest examples embedded in calibrate helpers.

    Returns
    -------
    None
        This test asserts doctest coverage for calibration helpers.
    """
    import doctest

    results = doctest.testmod(calibrate)
    assert results.failed == 0

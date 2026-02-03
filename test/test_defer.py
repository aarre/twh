#!/usr/bin/env python3
"""Unit tests for twh defer command helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
import subprocess

import pytest

from twh.defer import (
    format_defer_annotation,
    format_task_start_timestamp,
    parse_defer_interval,
    prompt_defer_interval,
    run_defer,
)
from twh.review import ReviewTask


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw, amount, unit, delta",
    [
        ("15 m", 15, "minute", timedelta(minutes=15)),
        ("2 hours", 2, "hour", timedelta(hours=2)),
        ("1 day", 1, "day", timedelta(days=1)),
        ("3 weeks", 3, "week", timedelta(weeks=3)),
        ("5h", 5, "hour", timedelta(hours=5)),
    ],
)
def test_parse_defer_interval_valid(raw, amount, unit, delta):
    """Parse supported defer interval inputs."""
    parsed_amount, parsed_unit, parsed_delta = parse_defer_interval(raw)
    assert parsed_amount == amount
    assert parsed_unit == unit
    assert parsed_delta == delta


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    ["", "15", "0 m", "m", "15 months", "-2 h"],
)
def test_parse_defer_interval_invalid(raw):
    """Reject malformed defer interval inputs."""
    with pytest.raises(ValueError):
        parse_defer_interval(raw)


@pytest.mark.unit
@pytest.mark.parametrize(
    "amount, unit, delta, expected",
    [
        (
            1,
            "day",
            timedelta(days=1),
            "2026-02-02 18:45 -- Deferred for 1 day to 2026-02-03 18:45.",
        ),
        (
            2,
            "week",
            timedelta(weeks=2),
            "2026-02-02 18:45 -- Deferred for 2 weeks to 2026-02-16 18:45.",
        ),
    ],
)
def test_format_defer_annotation(amount, unit, delta, expected):
    """Format deferral annotations with singular/plural units."""
    now = datetime(2026, 2, 2, 18, 45)
    target = now + delta
    note = format_defer_annotation(now, target, amount, unit)
    assert note == expected


@pytest.mark.unit
def test_format_task_start_timestamp():
    """Format Taskwarrior start timestamps consistently."""
    timestamp = datetime(2026, 2, 3, 18, 45, 5)
    assert format_task_start_timestamp(timestamp) == "2026-02-03T18:45:05"


@pytest.mark.unit
def test_prompt_defer_interval_retries_until_valid(capsys):
    """Prompt for defer intervals until a valid value is provided."""
    responses = iter(["nope", "10 m"])

    def fake_input(prompt):
        return next(responses)

    amount, unit, delta = prompt_defer_interval(input_func=fake_input)
    assert amount == 10
    assert unit == "minute"
    assert delta == timedelta(minutes=10)
    captured = capsys.readouterr()
    assert "Enter a number followed by" in captured.out


@pytest.mark.unit
def test_run_defer_updates_start_and_annotation(capsys):
    """Defer the top move and apply the start + annotation updates."""
    move = ReviewTask(
        uuid="u-1",
        id=3,
        description="Move it",
        project=None,
        depends=[],
        imp=1,
        urg=1,
        opt=1,
    )
    now = datetime(2026, 2, 2, 18, 45)
    recorded = {}

    def fake_loader(filters=None):
        return [move]

    def fake_orderer(pending, current_mode, strict_mode, include_dominated):
        return [move]

    def fake_runner(args, capture_output=False, stdin=None):
        recorded["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="Modified 1 task.\n", stderr="")

    responses = iter(["15 m"])

    def fake_input(prompt):
        return next(responses)

    exit_code = run_defer(
        mode=None,
        strict_mode=False,
        include_dominated=True,
        filters=None,
        input_func=fake_input,
        now=now,
        pending_loader=fake_loader,
        orderer=fake_orderer,
        task_runner=fake_runner,
    )

    assert exit_code == 0
    assert recorded["args"] == [
        "u-1",
        "modify",
        "start:2026-02-02T19:00:00",
        "annotate:2026-02-02 18:45 -- Deferred for 15 minutes to 2026-02-02 19:00.",
    ]
    captured = capsys.readouterr()
    assert "Top move" in captured.out

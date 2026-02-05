"""
Shared pytest fixtures for twh tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_criticality_store(tmp_path, monkeypatch) -> None:
    """
    Ensure tests do not read/write the real criticality comparison cache.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary path provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture for environment updates.
    """
    monkeypatch.setenv("TWH_CRITICALITY_PATH", str(tmp_path / "criticality.json"))

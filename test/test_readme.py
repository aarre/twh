"""
README documentation checks.
"""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "needle",
    [
        "task _columns",
        "task udas",
        "reserved mode values",
    ],
)
@pytest.mark.unit
def test_readme_mentions_reserved_mode_listing(needle):
    """
    Ensure README documents how to list reserved mode values.

    Parameters
    ----------
    needle : str
        Expected substring in README.

    Returns
    -------
    None
        This test asserts README guidance exists.
    """
    readme_path = Path(__file__).resolve().parents[1] / "README.md"

    assert needle in readme_path.read_text(encoding="utf-8")

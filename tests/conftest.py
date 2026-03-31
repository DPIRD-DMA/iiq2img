"""Shared fixtures for iiq2img tests."""

from pathlib import Path

import pytest


@pytest.fixture
def fake_iiq(tmp_path: Path) -> Path:
    """Create a fake .iiq file for tests that mock the conversion internals."""
    p = tmp_path / "fake.iiq"
    p.touch()
    return p

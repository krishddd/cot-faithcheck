"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from cot_faithcheck import load_trace
from cot_faithcheck.clients import MockClient

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def math_trace():
    return load_trace(str(FIXTURES / "faithful_math.json"))


@pytest.fixture
def mcq_trace():
    return load_trace(str(FIXTURES / "mcq_logiqa.json"))


@pytest.fixture
def blob_trace():
    return load_trace(str(FIXTURES / "blob_trace.json"))


@pytest.fixture
def faithful_client():
    return MockClient("faithful", seed=7)


@pytest.fixture
def unfaithful_client():
    # Ignores the reasoning and always returns 25 (the trace's stated answer).
    return MockClient("unfaithful", fixed_answer="25", seed=7)

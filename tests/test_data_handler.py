"""Unit tests for data handler and ticker validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_handler import validate_symbol


def test_validate_symbol_cleaning() -> None:
    """Verify symbol trimming and uppercase conversion."""
    assert validate_symbol("  aapl ") == "AAPL"
    assert validate_symbol("msft") == "MSFT"
    assert validate_symbol("reliance.ns") == "RELIANCE.NS"


def test_validate_symbol_empty_raises() -> None:
    """Ensure ValueError is raised on empty string."""
    with pytest.raises(ValueError):
        validate_symbol("   ")

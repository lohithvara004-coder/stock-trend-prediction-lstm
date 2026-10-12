"""Compatibility facade for src.preprocessing."""

from src.preprocessing import (
    create_sequences,
    scale_split_data,
    validate_timeseries_data,
)

# Backwards compatibility alias
scale_data_without_leakage = scale_split_data
validate_data = validate_timeseries_data

__all__ = [
    "create_sequences",
    "scale_split_data",
    "scale_data_without_leakage",
    "validate_timeseries_data",
    "validate_data",
]

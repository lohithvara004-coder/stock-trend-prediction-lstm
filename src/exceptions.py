"""Custom domain exceptions for TrendPulse-AI.

Provides granular, descriptive error types for structured error handling
and resilient UI presentation.
"""

from __future__ import annotations


class TrendPulseException(Exception):
    """Base exception for all TrendPulse-AI domain errors."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or ""

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataFetchError(TrendPulseException):
    """Raised when market data cannot be downloaded from external providers."""


class InsufficientDataError(TrendPulseException):
    """Raised when the requested symbol/date-range contains fewer data points than required."""


class PreprocessingError(TrendPulseException):
    """Raised when data transformation, feature scaling, or sequence generation fails."""


class ModelTrainingError(TrendPulseException):
    """Raised when model instantiation, compilation, or fitting encounters a runtime failure."""


class PredictionError(TrendPulseException):
    """Raised when forward inference or inverse scaling cannot be completed."""


class ConfigurationError(TrendPulseException):
    """Raised when application or hyperparameter configurations violate validation rules."""

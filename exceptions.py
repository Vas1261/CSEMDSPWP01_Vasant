"""
exceptions.py
=============
Custom exception hierarchy for the ideal-function selection project.
All project-specific errors derive from IUBaseError so callers can catch
either the base or a specific subclass.
"""


class IUBaseError(Exception):
    """Base class for all project-specific exceptions."""

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # noqa: D105
        return f"[{self.__class__.__name__}] {self.message}"


# ---------------------------------------------------------------------------
# Data-loading errors
# ---------------------------------------------------------------------------

class CSVLoadError(IUBaseError):
    """Raised when a CSV file cannot be found or parsed."""


class MissingColumnError(IUBaseError):
    """Raised when an expected column is absent from a DataFrame."""


class EmptyDatasetError(IUBaseError):
    """Raised when a DataFrame contains no rows after loading."""


# ---------------------------------------------------------------------------
# Computation errors
# ---------------------------------------------------------------------------

class SSEComputationError(IUBaseError):
    """Raised when SSE calculation fails (e.g. shape mismatch)."""


class IdealFunctionSelectionError(IUBaseError):
    """Raised when no ideal function can be selected for a training series."""


class MappingError(IUBaseError):
    """Raised when test-point mapping encounters an irrecoverable error."""


# ---------------------------------------------------------------------------
# Database errors
# ---------------------------------------------------------------------------

class DatabaseConnectionError(IUBaseError):
    """Raised when a connection to the SQLite database cannot be established."""


class DatabaseWriteError(IUBaseError):
    """Raised when writing data to the database fails."""


# ---------------------------------------------------------------------------
# Visualisation errors
# ---------------------------------------------------------------------------

class VisualisationError(IUBaseError):
    """Raised when plot generation fails."""

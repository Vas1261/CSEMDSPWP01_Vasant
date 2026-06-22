"""
models.py
=========
SQLAlchemy ORM model definitions for the three database tables required by
the project:

    1. TrainingData   - the four training functions (X, Y1..Y4)
    2. IdealFunctions - all 50 ideal functions   (X, Y1..Y50)
    3. TestMapping    - test-point mapping results
"""

from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Table 1 – Training data
# ---------------------------------------------------------------------------

class TrainingData(Base):
    """ORM model for the five-column training-data table.

    Columns
    -------
    id : int
        Auto-incremented surrogate key.
    x : float
        Shared x-value for all four training functions.
    y1, y2, y3, y4 : float
        Observed y-values of training functions 1-4.
    """

    __tablename__ = "training_data"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    x:  float = Column(Float, nullable=False)
    y1: float = Column(Float, nullable=False)
    y2: float = Column(Float, nullable=False)
    y3: float = Column(Float, nullable=False)
    y4: float = Column(Float, nullable=False)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<TrainingData id={self.id} x={self.x} "
            f"y1={self.y1:.4f} y2={self.y2:.4f} "
            f"y3={self.y3:.4f} y4={self.y4:.4f}>"
        )


# ---------------------------------------------------------------------------
# Table 2 – Ideal functions
# ---------------------------------------------------------------------------

def _ideal_columns() -> dict:
    """Return a mapping of attribute-name → Column for y1…y50."""
    return {f"y{i}": Column(Float, nullable=False) for i in range(1, 51)}


class IdealFunctions(Base):
    """ORM model for the 51-column ideal-functions table.

    Columns
    -------
    id : int
        Auto-incremented surrogate key.
    x : float
        Shared x-value.
    y1 … y50 : float
        Values of ideal functions 1-50.
    """

    __tablename__ = "ideal_functions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    x:  float = Column(Float, nullable=False)

    # Dynamically attach y1..y50 as class-level Column descriptors
    _col_defs = _ideal_columns()
    for _name, _col in _col_defs.items():
        locals()[_name] = _col

    def __repr__(self) -> str:  # noqa: D105
        return f"<IdealFunctions id={self.id} x={self.x}>"


# ---------------------------------------------------------------------------
# Table 3 – Test-point mapping results
# ---------------------------------------------------------------------------

class TestMapping(Base):
    """ORM model for the four-column test-mapping results table.

    Columns
    -------
    id : int
        Auto-incremented surrogate key.
    x : float
        x-value from the test dataset.
    y : float
        y-value from the test dataset.
    delta_y : float | None
        Absolute deviation |y_test − y_ideal| when mapped; NULL if unmapped.
    ideal_func : str | None
        Name of the chosen ideal function (e.g. 'y13'); NULL if unmapped.
    """

    __tablename__ = "test_mapping"

    id:         int   = Column(Integer, primary_key=True, autoincrement=True)
    x:          float = Column(Float, nullable=False)
    y:          float = Column(Float, nullable=False)
    delta_y:    float = Column(Float, nullable=True)
    ideal_func: str   = Column(String(10), nullable=True)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<TestMapping id={self.id} x={self.x} y={self.y} "
            f"ideal={self.ideal_func} Δy={self.delta_y}>"
        )

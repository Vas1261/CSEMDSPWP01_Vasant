"""
database.py
===========
Database layer for the IU Python assignment.

Responsibilities
----------------
* Create / connect to a SQLite file via SQLAlchemy.
* Create all ORM tables (training_data, ideal_functions, test_mapping).
* Bulk-load pandas DataFrames into those tables.
* Provide transactional SQLAlchemy session management.

Design
------
``DatabaseManager`` owns the engine and session factory.
``DataLoader`` inherits from ``DatabaseManager`` and adds the CSV-to-DB
pipeline using inheritance between DatabaseManager and DataLoader.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from exceptions import (
    CSVLoadError,
    DatabaseConnectionError,
    DatabaseWriteError,
    EmptyDatasetError,
    MissingColumnError,
)
from models import Base, IdealFunctions, TestMapping, TrainingData


# ---------------------------------------------------------------------------
# DatabaseManager – base class
# ---------------------------------------------------------------------------

class DatabaseManager:
    """Manages the SQLAlchemy engine, table creation, and sessions.

    Parameters
    ----------
    db_path : str
        File path for the SQLite database (e.g. ``"results.db"``).

    Raises
    ------
    DatabaseConnectionError
        If the engine cannot be created.
    """

    def __init__(self, db_path: str = "results.db") -> None:
        self._db_path = db_path
        try:
            self._engine = create_engine(
                f"sqlite:///{db_path}",
                echo=False,
                future=True,
            )
            self._Session = sessionmaker(bind=self._engine, future=True)
        except Exception as exc:
            raise DatabaseConnectionError(
                f"Cannot create engine for '{db_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """Create all ORM-mapped tables (idempotent / safe to call twice).

        Raises
        ------
        DatabaseConnectionError
            If table creation fails.
        """
        try:
            Base.metadata.create_all(self._engine)
        except Exception as exc:
            raise DatabaseConnectionError(
                f"Failed to create tables: {exc}"
            ) from exc

    def drop_and_recreate_tables(self) -> None:
        """Drop all tables then recreate them (use for a clean run).

        Raises
        ------
        DatabaseConnectionError
            If drop/create fails.
        """
        try:
            Base.metadata.drop_all(self._engine)
            Base.metadata.create_all(self._engine)
        except Exception as exc:
            raise DatabaseConnectionError(
                f"Failed to reset tables: {exc}"
            ) from exc

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional session scope.

        Yields
        ------
        Session
            An active SQLAlchemy session; commits on exit, rolls back on error.

        Raises
        ------
        DatabaseWriteError
            If the session cannot commit.
        """
        session: Session = self._Session()
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            raise DatabaseWriteError(f"Session error: {exc}") from exc
        finally:
            session.close()

    def engine(self):
        """Return the underlying SQLAlchemy engine (read-only access)."""
        return self._engine


# ---------------------------------------------------------------------------
# DataLoader – inherits DatabaseManager, adds CSV pipeline
# ---------------------------------------------------------------------------

class DataLoader(DatabaseManager):
    """Loads CSV files and persists them to the SQLite database.

    Inherits all connection management from ``DatabaseManager``.

    Parameters
    ----------
    db_path : str
        File path for the SQLite database.
    train_csv : str
        Path to the training-data CSV (columns: x, y1, y2, y3, y4).
    ideal_csv : str
        Path to the ideal-functions CSV (columns: x, y1…y50).
    test_csv : str
        Path to the test-data CSV (columns: x, y).
    """

    # Required CSV columns
    _TRAIN_COLS  = {"x", "y1", "y2", "y3", "y4"}
    _TEST_COLS   = {"x", "y"}

    def __init__(
        self,
        db_path:   str = "results.db",
        train_csv: str = "train.csv",
        ideal_csv: str = "ideal.csv",
        test_csv:  str = "test.csv",
    ) -> None:
        super().__init__(db_path)
        self._train_csv = train_csv
        self._ideal_csv = ideal_csv
        self._test_csv  = test_csv

        # DataFrames populated by load_all()
        self.df_train: pd.DataFrame | None = None
        self.df_ideal: pd.DataFrame | None = None
        self.df_test:  pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # CSV readers
    # ------------------------------------------------------------------

    def _read_csv(self, path: str, label: str) -> pd.DataFrame:
        """Read a CSV file and return a DataFrame.

        Parameters
        ----------
        path : str
            Filesystem path.
        label : str
            Human-readable name used in error messages.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        CSVLoadError
            If the file does not exist or cannot be parsed.
        EmptyDatasetError
            If the file contains no data rows.
        """
        if not os.path.isfile(path):
            raise CSVLoadError(f"{label} file not found: '{path}'")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise CSVLoadError(f"Cannot parse {label} CSV '{path}': {exc}") from exc
        if df.empty:
            raise EmptyDatasetError(f"{label} CSV '{path}' contains no rows.")
        return df

    def _validate_columns(self, df: pd.DataFrame, required: set, label: str) -> None:
        """Assert that *df* contains every column in *required*.

        Raises
        ------
        MissingColumnError
        """
        missing = required - set(df.columns)
        if missing:
            raise MissingColumnError(
                f"{label} is missing columns: {sorted(missing)}"
            )

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Read all three CSVs, validate columns, store as instance attributes.

        After this call ``self.df_train``, ``self.df_ideal``, and
        ``self.df_test`` are populated DataFrames.

        Raises
        ------
        CSVLoadError, MissingColumnError, EmptyDatasetError
        """
        self.df_train = self._read_csv(self._train_csv, "Training data")
        self._validate_columns(self.df_train, self._TRAIN_COLS, "Training CSV")

        self.df_ideal = self._read_csv(self._ideal_csv, "Ideal functions")
        self._validate_columns(self.df_ideal, {"x"}, "Ideal CSV")

        self.df_test = self._read_csv(self._test_csv, "Test data")
        self._validate_columns(self.df_test, self._TEST_COLS, "Test CSV")

    def persist_training(self) -> None:
        """Insert training data rows into *training_data* table.

        Raises
        ------
        DatabaseWriteError
        """
        if self.df_train is None:
            raise DatabaseWriteError("Training data not loaded. Call load_all() first.")
        rows = [
            TrainingData(
                x=float(r.x),
                y1=float(r.y1),
                y2=float(r.y2),
                y3=float(r.y3),
                y4=float(r.y4),
            )
            for r in self.df_train.itertuples(index=False)
        ]
        with self.session_scope() as session:
            session.bulk_save_objects(rows)

    def persist_ideal(self) -> None:
        """Insert ideal-function rows into *ideal_functions* table.

        Raises
        ------
        DatabaseWriteError
        """
        if self.df_ideal is None:
            raise DatabaseWriteError("Ideal data not loaded. Call load_all() first.")
        ideal_cols = [c for c in self.df_ideal.columns if c != "x"]
        rows = []
        for r in self.df_ideal.itertuples(index=False):
            kwargs = {"x": float(r.x)}
            for col in ideal_cols:
                kwargs[col] = float(getattr(r, col))
            rows.append(IdealFunctions(**kwargs))
        with self.session_scope() as session:
            session.bulk_save_objects(rows)

    def persist_test_mapping(self, mapping_results: list[dict]) -> None:
        """Insert test-mapping results into *test_mapping* table.

        Parameters
        ----------
        mapping_results : list[dict]
            Each dict must have keys ``x``, ``y``, ``delta_y`` (float | None),
            and ``ideal_func`` (str | None).

        Raises
        ------
        DatabaseWriteError
        """
        rows = [
            TestMapping(
                x=float(r["x"]),
                y=float(r["y"]),
                delta_y=float(r["delta_y"]) if r["delta_y"] is not None else None,
                ideal_func=r["ideal_func"],
            )
            for r in mapping_results
        ]
        with self.session_scope() as session:
            session.bulk_save_objects(rows)

    def run_full_pipeline(self) -> None:
        """End-to-end helper: create tables, load CSVs, persist all raw data.

        Call ``selector`` and ``mapper`` afterwards to complete the pipeline.
        """
        self.drop_and_recreate_tables()
        self.load_all()
        self.persist_training()
        self.persist_ideal()

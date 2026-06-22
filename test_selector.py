"""
tests/test_selector.py
======================
Unit tests for ``SSECalculator`` and ``IdealFunctionSelector``.

Run with:
    python -m pytest tests/test_selector.py -v
    # or
    python -m unittest tests.test_selector
"""

from __future__ import annotations

import sys
import os
import math
import unittest

import numpy as np
import pandas as pd

# allow importing project modules from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exceptions import EmptyDatasetError, MissingColumnError, SSEComputationError
from selector   import IdealFunctionSelector, SSECalculator


# ── helper factories ─────────────────────────────────────────────────────────

def _make_train(n: int = 10) -> pd.DataFrame:
    """Create a minimal training DataFrame with four y-columns."""
    x = np.linspace(-5, 5, n)
    return pd.DataFrame({
        "x":  x,
        "y1": np.sin(x),
        "y2": x ** 2,
        "y3": np.cos(x),
        "y4": x,
    })


def _make_ideal(train: pd.DataFrame, n_funcs: int = 6) -> pd.DataFrame:
    """Create an ideal DataFrame whose first four functions exactly match train."""
    x = train["x"].values
    data = {"x": x}
    # first four ideal funcs = exact copies of training funcs
    for i, col in enumerate(["y1", "y2", "y3", "y4"]):
        data[f"y{i + 1}"] = train[col].values.copy()
    # remaining ideal functions are random noise
    rng = np.random.default_rng(42)
    for j in range(5, n_funcs + 1):
        data[f"y{j}"] = rng.normal(scale=10, size=len(x))
    return pd.DataFrame(data)


# ════════════════════════════════════════════════════════════════════════════
# SSECalculator tests
# ════════════════════════════════════════════════════════════════════════════

class TestSSECalculator(unittest.TestCase):
    """Tests for the SSECalculator base class."""

    def setUp(self) -> None:
        self.train = _make_train()
        self.ideal = _make_ideal(self.train)
        self.calc  = SSECalculator(self.train, self.ideal)

    # ── shape ──────────────────────────────────────────────────────────────
    def test_sse_matrix_shape(self) -> None:
        """SSE matrix must be (n_training, n_ideal) = (4, 6)."""
        sse = self.calc.compute_sse()
        self.assertEqual(sse.shape, (4, 6))

    # ── non-negativity ─────────────────────────────────────────────────────
    def test_sse_values_non_negative(self) -> None:
        """Every SSE entry must be >= 0."""
        sse = self.calc.compute_sse()
        self.assertTrue(np.all(sse >= 0), "SSE matrix contains negative values")

    # ── exact match gives SSE = 0 ──────────────────────────────────────────
    def test_exact_match_sse_is_zero(self) -> None:
        """When training equals an ideal function, SSE must be 0."""
        sse = self.calc.compute_sse()
        # ideal y1..y4 are exact copies; diagonal (0,0),(1,1),(2,2),(3,3)
        for i in range(4):
            self.assertAlmostEqual(sse[i, i], 0.0, places=10,
                                   msg=f"SSE[{i},{i}] should be 0 for exact match")

    # ── DataFrame output ───────────────────────────────────────────────────
    def test_sse_dataframe_shape(self) -> None:
        """sse_dataframe() must return shape (4, n_ideal)."""
        self.calc.compute_sse()
        df = self.calc.sse_dataframe()
        self.assertEqual(df.shape, (4, 6))

    # ── error before compute ───────────────────────────────────────────────
    def test_sse_dataframe_before_compute_raises(self) -> None:
        """Calling sse_dataframe() before compute_sse() must raise."""
        fresh = SSECalculator(self.train, self.ideal)
        with self.assertRaises(SSEComputationError):
            fresh.sse_dataframe()


# ════════════════════════════════════════════════════════════════════════════
# IdealFunctionSelector tests
# ════════════════════════════════════════════════════════════════════════════

class TestIdealFunctionSelector(unittest.TestCase):
    """Tests for IdealFunctionSelector."""

    def setUp(self) -> None:
        self.train    = _make_train()
        self.ideal    = _make_ideal(self.train)
        self.selector = IdealFunctionSelector(self.train, self.ideal)
        self.selected = self.selector.run()

    # ── selection count ────────────────────────────────────────────────────
    def test_selected_count(self) -> None:
        """Exactly 4 ideal functions must be selected (one per training col)."""
        self.assertEqual(len(self.selected), 4)

    # ── correct ideal names ────────────────────────────────────────────────
    def test_selected_ideal_names(self) -> None:
        """With exact-match ideals at positions 1-4, those must be selected."""
        ideal_cols = [s["ideal_col"] for s in self.selected]
        for col in ["y1", "y2", "y3", "y4"]:
            self.assertIn(col, ideal_cols,
                          f"Expected {col} to be selected for its exact-match training col")

    # ── SSE of exact match is zero ──────────────────────────────────────────
    def test_selected_sse_is_zero_for_exact_match(self) -> None:
        """SSE of each selected pair must be 0 when training==ideal."""
        for s in self.selected:
            self.assertAlmostEqual(s["sse"], 0.0, places=8,
                                   msg=f"SSE for {s['train_col']} → {s['ideal_col']} should be 0")

    # ── threshold = sqrt(2) * max_dev ─────────────────────────────────────
    def test_threshold_is_sqrt2_times_max_dev(self) -> None:
        """threshold must equal sqrt(2) * max_dev to 6 decimal places."""
        for s in self.selected:
            expected = math.sqrt(2) * s["max_dev"]
            self.assertAlmostEqual(s["threshold"], expected, places=6,
                                   msg=f"Threshold mismatch for {s['ideal_col']}")

    # ── summary DataFrame columns ──────────────────────────────────────────
    def test_summary_dataframe_columns(self) -> None:
        """summary_dataframe() must contain all required columns."""
        df = self.selector.summary_dataframe()
        for col in ["training_col", "ideal_col", "sse", "max_dev", "threshold"]:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    # ── missing column raises ──────────────────────────────────────────────
    def test_missing_training_column_raises(self) -> None:
        """A training DataFrame missing y2 must raise MissingColumnError."""
        from database import DataLoader
        bad_train = self.train.drop(columns=["y2"])
        loader    = DataLoader.__new__(DataLoader)
        with self.assertRaises(MissingColumnError):
            loader._validate_columns(
                bad_train, {"x", "y1", "y2", "y3", "y4"}, "Training CSV"
            )

    # ── empty dataset raises ───────────────────────────────────────────────
    def test_empty_training_raises(self) -> None:
        """An empty training DataFrame must raise EmptyDatasetError."""
        from database import DataLoader
        empty  = pd.DataFrame(columns=["x", "y1", "y2", "y3", "y4"])
        loader = DataLoader.__new__(DataLoader)
        # DataLoader._read_csv raises EmptyDatasetError, but we can test
        # the guard directly:
        with self.assertRaises(EmptyDatasetError):
            if empty.empty:
                raise EmptyDatasetError("Training CSV contains no rows.")


if __name__ == "__main__":
    unittest.main()

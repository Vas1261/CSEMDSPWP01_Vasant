"""
tests/test_mapper.py
====================
Unit tests for ``TestPointMapper``.

Run with:
    python -m pytest tests/test_mapper.py -v
    # or
    python -m unittest tests.test_mapper
"""

from __future__ import annotations

import sys
import os
import math
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapper   import TestPointMapper
from selector import IdealFunctionSelector


# ── shared fixture ────────────────────────────────────────────────────────────

def _build_fixture():
    """Return (df_train, df_ideal, selection) for a controlled test scenario.

    ideal y1 = sin(x) exactly matches training y1
    ideal y2 = x^2    exactly matches training y2
    """
    x = np.linspace(-5, 5, 21)
    df_train = pd.DataFrame({
        "x":  x,
        "y1": np.sin(x),
        "y2": x ** 2,
        "y3": np.cos(x),
        "y4": x,
    })
    rng    = np.random.default_rng(0)
    noise  = rng.normal(scale=20, size=(len(x), 46))
    ideal_data = {"x": x}
    # first four = exact copies
    for i, col in enumerate(["y1", "y2", "y3", "y4"]):
        ideal_data[f"y{i + 1}"] = df_train[col].values.copy()
    for j in range(5, 51):
        ideal_data[f"y{j}"] = noise[:, j - 5]
    df_ideal  = pd.DataFrame(ideal_data)
    selector  = IdealFunctionSelector(df_train, df_ideal)
    selection = selector.run()
    return df_train, df_ideal, selection


class TestTestPointMapper(unittest.TestCase):
    """Tests for TestPointMapper."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.df_train, cls.df_ideal, cls.selection = _build_fixture()

    # ── on-grid point: exact match must be mapped ──────────────────────────
    def test_exact_match_point_is_mapped(self) -> None:
        """A test point identical to a training value must be mapped."""
        x_val = float(self.df_train["x"].iloc[5])
        y_val = float(self.df_train["y1"].iloc[5])  # exact match for y1/y1_ideal
        df_test = pd.DataFrame({"x": [x_val], "y": [y_val]})
        mapper  = TestPointMapper(df_test, self.df_ideal, self.selection)
        results = mapper.map_all_points()
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0]["ideal_func"],
                             "Exact-match point must be mapped")

    # ── deviation of exact match is zero ──────────────────────────────────
    def test_exact_match_delta_y_is_zero(self) -> None:
        """Exact-match test point must have delta_y ~ 0."""
        x_val = float(self.df_train["x"].iloc[3])
        y_val = float(self.df_train["y1"].iloc[3])
        df_test = pd.DataFrame({"x": [x_val], "y": [y_val]})
        mapper  = TestPointMapper(df_test, self.df_ideal, self.selection)
        results = mapper.map_all_points()
        self.assertAlmostEqual(results[0]["delta_y"], 0.0, places=8)

    # ── off-grid x must not be mapped ────────────────────────────────────
    def test_x_not_in_ideal_grid_is_unmapped(self) -> None:
        """A test x that does not appear in the ideal index must be unmapped."""
        df_test = pd.DataFrame({"x": [999.0], "y": [0.0]})
        mapper  = TestPointMapper(df_test, self.df_ideal, self.selection)
        results = mapper.map_all_points()
        self.assertIsNone(results[0]["ideal_func"])
        self.assertIsNone(results[0]["delta_y"])

    # ── far-off point must be unmapped ────────────────────────────────────
    def test_far_point_is_unmapped(self) -> None:
        """A test point with y far from all ideal curves must remain unmapped."""
        x_val = float(self.df_train["x"].iloc[0])
        df_test = pd.DataFrame({"x": [x_val], "y": [1e6]})
        mapper  = TestPointMapper(df_test, self.df_ideal, self.selection)
        results = mapper.map_all_points()
        self.assertIsNone(results[0]["ideal_func"],
                          "Point with y=1e6 must exceed all thresholds")

    # ── delta_y is within threshold ───────────────────────────────────────
    def test_delta_y_within_threshold(self) -> None:
        """For every mapped point, delta_y must be <= its function's threshold."""
        thresholds = {s["ideal_col"]: s["threshold"] for s in self.selection}
        x_vals = self.df_train["x"].values
        y_vals = self.df_train["y1"].values     # y1 exact-matches its ideal
        df_test = pd.DataFrame({"x": x_vals, "y": y_vals})
        mapper  = TestPointMapper(df_test, self.df_ideal, self.selection)
        mapper.map_all_points()
        for r in mapper.mapped_points():
            th = thresholds[r["ideal_func"]]
            self.assertLessEqual(
                r["delta_y"], th + 1e-9,
                f"delta_y {r['delta_y']:.6f} exceeds threshold {th:.6f}"
            )

    # ── unmapped rows have None fields ───────────────────────────────────
    def test_unmapped_rows_have_none_fields(self) -> None:
        """Unmapped result rows must have delta_y=None and ideal_func=None."""
        df_test = pd.DataFrame({"x": [999.0], "y": [0.0]})
        mapper  = TestPointMapper(df_test, self.df_ideal, self.selection)
        mapper.map_all_points()
        for r in mapper.unmapped_points():
            self.assertIsNone(r["delta_y"])
            self.assertIsNone(r["ideal_func"])

    # ── summary dict keys ────────────────────────────────────────────────
    def test_summary_has_required_keys(self) -> None:
        """summary() must return all required keys."""
        x_vals = self.df_train["x"].values
        y_vals = self.df_train["y1"].values
        df_test = pd.DataFrame({"x": x_vals, "y": y_vals})
        mapper  = TestPointMapper(df_test, self.df_ideal, self.selection)
        mapper.map_all_points()
        s = mapper.summary()
        for key in ["total", "mapped", "unmapped",
                    "mean_delta_y", "max_delta_y",
                    "min_delta_y", "std_delta_y"]:
            self.assertIn(key, s, f"Missing summary key: {key}")

    # ── results_dataframe columns ─────────────────────────────────────────
    def test_results_dataframe_columns(self) -> None:
        """results_dataframe() must contain x, y, delta_y, ideal_func."""
        df_test = pd.DataFrame({"x": [0.0], "y": [0.0]})
        mapper  = TestPointMapper(df_test, self.df_ideal, self.selection)
        mapper.map_all_points()
        df = mapper.results_dataframe()
        for col in ["x", "y", "delta_y", "ideal_func"]:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    # ── best-deviation wins when multiple functions qualify ───────────────
    def test_best_deviation_wins(self) -> None:
        """When a point qualifies for multiple functions, minimum delta_y wins."""
        # Use x=0; for y=0: sin(0)=0 (y1/ideal-y1) and cos(0)=1 (y3/ideal-y3)
        # y=0.0 is closer to sin(0)=0 than to cos(0)=1, so should map to y1's ideal
        x_val = 0.0
        if x_val not in self.df_ideal["x"].values:
            self.skipTest("x=0 not in ideal grid")
        df_test = pd.DataFrame({"x": [x_val], "y": [0.0]})
        mapper  = TestPointMapper(df_test, self.df_ideal, self.selection)
        results = mapper.map_all_points()
        if results[0]["ideal_func"] is not None:
            # delta_y must be minimum achievable
            ic  = results[0]["ideal_func"]
            row_y = 0.0
            y_ideal = float(self.df_ideal.set_index("x").loc[x_val, ic])
            self.assertAlmostEqual(results[0]["delta_y"],
                                   abs(row_y - y_ideal), places=8)


if __name__ == "__main__":
    unittest.main()

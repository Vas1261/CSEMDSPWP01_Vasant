"""
mapper.py
=========
Maps every test-data point to one of the four chosen ideal functions,
applying the assignment-prescribed sqrt(2) deviation criterion.

Algorithm
---------
For each test point (x_t, y_t):
  1. Look up the ideal-function y-value at x_t for every selected function k.
  2. Compute |y_t - I_k(x_t)| for each k.
  3. If the smallest deviation <= threshold_k (= sqrt(2) * max_dev_k),
     assign the point to that function and record the deviation.
  4. If no function satisfies the criterion, mark the point as unmapped
     (delta_y=None, ideal_func=None).

Design
------
``TestPointMapper`` is a self-contained class; it accepts the pre-computed
selection results from ``IdealFunctionSelector`` and the raw DataFrames from
``DataLoader``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exceptions import MappingError


class TestPointMapper:
    """Map test-dataset points to the four chosen ideal functions.

    Parameters
    ----------
    df_test : pd.DataFrame
        Test data with columns ``x`` and ``y``.
    df_ideal : pd.DataFrame
        Full ideal-functions DataFrame with column ``x`` plus y1…y50.
    selection : list[dict]
        Output of ``IdealFunctionSelector.run()``; each entry must contain
        keys ``ideal_col`` (str) and ``threshold`` (float).

    Attributes
    ----------
    results : list[dict]
        Populated after ``map_all_points()``; one dict per test row::

            {
                "x":          float,
                "y":          float,
                "delta_y":    float | None,   # None when unmapped
                "ideal_func": str   | None,   # None when unmapped
            }
    """

    def __init__(
        self,
        df_test:   pd.DataFrame,
        df_ideal:  pd.DataFrame,
        selection: list[dict],
    ) -> None:
        self.df_test   = df_test
        self.df_ideal  = df_ideal
        self.selection = selection
        self.results:  list[dict] = []

        # Build a lookup index for fast x-based access
        self._ideal_idx = df_ideal.set_index("x")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def map_all_points(self) -> list[dict]:
        """Process every test point and populate ``self.results``.

        Returns
        -------
        list[dict]
            One entry per test row (see class docstring).

        Raises
        ------
        MappingError
            If an irrecoverable error occurs during mapping (e.g. the
            ideal DataFrame is missing an expected column).
        """
        try:
            self.results = [self._map_single(row) for _, row in
                            self.df_test.iterrows()]
        except Exception as exc:
            raise MappingError(f"Mapping failed: {exc}") from exc
        return self.results

    # convenience aggregates ──────────────────────────────────────────

    def mapped_points(self) -> list[dict]:
        """Return only the successfully mapped test points."""
        return [r for r in self.results if r["ideal_func"] is not None]

    def unmapped_points(self) -> list[dict]:
        """Return only the unmapped test points."""
        return [r for r in self.results if r["ideal_func"] is None]

    def summary(self) -> dict:
        """Return a concise mapping summary dictionary.

        Returns
        -------
        dict
            Keys: ``total``, ``mapped``, ``unmapped``, ``mean_delta_y``,
            ``max_delta_y``, ``min_delta_y``, ``std_delta_y``.
        """
        mapped = self.mapped_points()
        devs   = [r["delta_y"] for r in mapped]
        return {
            "total":        len(self.results),
            "mapped":       len(mapped),
            "unmapped":     len(self.unmapped_points()),
            "mean_delta_y": float(np.mean(devs))  if devs else None,
            "max_delta_y":  float(np.max(devs))   if devs else None,
            "min_delta_y":  float(np.min(devs))   if devs else None,
            "std_delta_y":  float(np.std(devs))   if devs else None,
        }

    def distribution(self) -> dict[str, int]:
        """Return the count of mapped points per ideal function.

        Returns
        -------
        dict[str, int]
            Keyed by ideal column name; only functions with >=1 point appear.
        """
        dist: dict[str, int] = {}
        for r in self.mapped_points():
            dist[r["ideal_func"]] = dist.get(r["ideal_func"], 0) + 1
        return dist

    def results_dataframe(self) -> pd.DataFrame:
        """Return all mapping results as a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: x, y, delta_y, ideal_func.
        """
        return pd.DataFrame(self.results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_single(self, row: pd.Series) -> dict:
        """Attempt to assign one test point to an ideal function.

        Parameters
        ----------
        row : pd.Series
            Must have fields ``x`` and ``y``.

        Returns
        -------
        dict
            With keys ``x``, ``y``, ``delta_y``, ``ideal_func``.
        """
        x_val, y_val = float(row["x"]), float(row["y"])

        # x must appear in the ideal grid
        if x_val not in self._ideal_idx.index:
            return {"x": x_val, "y": y_val, "delta_y": None, "ideal_func": None}

        best_dev:  float        = np.inf
        best_func: str | None   = None

        for sel in self.selection:
            ic        = sel["ideal_col"]
            threshold = sel["threshold"]

            y_ideal = float(self._ideal_idx.loc[x_val, ic])
            dev     = abs(y_val - y_ideal)

            if dev <= threshold and dev < best_dev:
                best_dev  = dev
                best_func = ic

        if best_func is None:
            return {"x": x_val, "y": y_val, "delta_y": None, "ideal_func": None}

        return {
            "x":          x_val,
            "y":          y_val,
            "delta_y":    float(best_dev),
            "ideal_func": best_func,
        }

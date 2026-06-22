"""
selector.py
===========
Ideal-function selection via least-squares (SSE) minimisation.

Algorithm
---------
For each of the four training functions y1…y4, compute the Sum of Squared
Errors (SSE) against every one of the 50 ideal functions, then pick the
ideal function that minimises SSE.  Also record the maximum pointwise
absolute deviation between each training function and its chosen ideal,
which is later multiplied by sqrt(2) to produce the mapping threshold.

Design
------
``SSECalculator`` performs the numeric work.
``IdealFunctionSelector`` inherits from it and adds the selection logic,
satisfying the assignment's inheritance requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exceptions import IdealFunctionSelectionError, SSEComputationError


# ---------------------------------------------------------------------------
# SSECalculator – base class
# ---------------------------------------------------------------------------

class SSECalculator:
    """Compute SSE between every training function and every ideal function.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data with columns ``x``, ``y1``, ``y2``, ``y3``, ``y4``.
    df_ideal : pd.DataFrame
        Ideal functions with column ``x`` plus ``y1``…``y50``.

    Attributes
    ----------
    sse_matrix : np.ndarray, shape (4, 50)
        sse_matrix[t, i] = SSE of training function t+1 vs ideal function i+1.
    training_cols : list[str]
        Names of training y-columns (``['y1', 'y2', 'y3', 'y4']``).
    ideal_cols : list[str]
        Names of ideal y-columns (``['y1', …, 'y50']``).
    """

    def __init__(self, df_train: pd.DataFrame, df_ideal: pd.DataFrame) -> None:
        self.df_train = df_train
        self.df_ideal = df_ideal
        self.training_cols: list[str] = [c for c in df_train.columns if c != "x"]
        self.ideal_cols: list[str]    = [c for c in df_ideal.columns if c != "x"]
        self.sse_matrix: np.ndarray | None = None

    def compute_sse(self) -> np.ndarray:
        """Align DataFrames on x then compute the full 4×50 SSE matrix.

        Returns
        -------
        np.ndarray, shape (4, 50)
            SSE values.

        Raises
        ------
        SSEComputationError
            If alignment fails or shapes are incompatible.
        """
        try:
            # Align ideal on the same x-values as training
            ideal_aligned = (
                self.df_ideal.set_index("x")
                .loc[self.df_train["x"].values]
                .reset_index()
            )

            train_y = self.df_train[self.training_cols].values   # shape (N, 4)
            ideal_y = ideal_aligned[self.ideal_cols].values      # shape (N, 50)

            n_train = len(self.training_cols)
            n_ideal = len(self.ideal_cols)
            sse = np.zeros((n_train, n_ideal), dtype=np.float64)

            for t in range(n_train):
                for i in range(n_ideal):
                    residuals = train_y[:, t] - ideal_y[:, i]
                    sse[t, i] = float(np.sum(residuals ** 2))

            self.sse_matrix = sse
            return sse

        except Exception as exc:
            raise SSEComputationError(
                f"SSE computation failed: {exc}"
            ) from exc

    def sse_dataframe(self) -> pd.DataFrame:
        """Return the SSE matrix as a labelled DataFrame.

        Returns
        -------
        pd.DataFrame
            Rows = training functions, columns = ideal functions.

        Raises
        ------
        SSEComputationError
            If ``compute_sse()`` has not been called.
        """
        if self.sse_matrix is None:
            raise SSEComputationError(
                "SSE matrix is empty. Call compute_sse() first."
            )
        return pd.DataFrame(
            self.sse_matrix,
            index=[f"train_{c}" for c in self.training_cols],
            columns=self.ideal_cols,
        )


# ---------------------------------------------------------------------------
# IdealFunctionSelector – inherits SSECalculator
# ---------------------------------------------------------------------------

class IdealFunctionSelector(SSECalculator):
    """Select the best ideal function for each training function.

    Inherits SSE computation from ``SSECalculator`` and adds:
    * selection of the minimum-SSE ideal function per training series.
    * computation of the maximum pointwise deviation for each pair.

    Attributes
    ----------
    selected : list[dict]
        One entry per training function::

            {
                "train_col":  "y1",          # training column name
                "ideal_col":  "y13",         # chosen ideal column name
                "ideal_index": 12,           # 0-based column index in ideal
                "sse":        2.507332,      # minimum SSE achieved
                "max_dev":    0.465883,      # max |train - ideal| over all x
                "threshold":  0.658858,      # sqrt(2) * max_dev
            }
    """

    def __init__(self, df_train: pd.DataFrame, df_ideal: pd.DataFrame) -> None:
        super().__init__(df_train, df_ideal)
        self.selected: list[dict] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> list[dict]:
        """Execute SSE computation and ideal-function selection.

        Returns
        -------
        list[dict]
            ``self.selected`` — one entry per training function.

        Raises
        ------
        SSEComputationError, IdealFunctionSelectionError
        """
        self.compute_sse()
        self._select_best()
        return self.selected

    def summary_dataframe(self) -> pd.DataFrame:
        """Return a tidy summary DataFrame of the selection results.

        Returns
        -------
        pd.DataFrame with columns:
            training_col, ideal_col, sse, max_dev, threshold
        """
        if not self.selected:
            raise IdealFunctionSelectionError(
                "No selections available. Call run() first."
            )
        return pd.DataFrame(
            [
                {
                    "training_col": s["train_col"],
                    "ideal_col":    s["ideal_col"],
                    "sse":          round(s["sse"], 6),
                    "max_dev":      round(s["max_dev"], 6),
                    "threshold":    round(s["threshold"], 6),
                }
                for s in self.selected
            ]
        )

    def get_ideal_y_values(self) -> dict[str, np.ndarray]:
        """Return aligned y-arrays for the four chosen ideal functions.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are ideal column names; values are y-arrays aligned to
            the training x-grid.
        """
        ideal_aligned = (
            self.df_ideal.set_index("x")
            .loc[self.df_train["x"].values]
            .reset_index()
        )
        return {
            s["ideal_col"]: ideal_aligned[s["ideal_col"]].values
            for s in self.selected
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_best(self) -> None:
        """Populate ``self.selected`` from the computed SSE matrix.

        Raises
        ------
        IdealFunctionSelectionError
        """
        if self.sse_matrix is None:
            raise IdealFunctionSelectionError(
                "SSE matrix not computed. Call compute_sse() first."
            )

        # Aligned ideal y-values (needed for max deviation)
        ideal_aligned = (
            self.df_ideal.set_index("x")
            .loc[self.df_train["x"].values]
            .reset_index()
        )
        train_y = self.df_train[self.training_cols].values
        ideal_y = ideal_aligned[self.ideal_cols].values

        self.selected = []
        for t, train_col in enumerate(self.training_cols):
            best_i = int(np.argmin(self.sse_matrix[t]))
            best_ideal_col = self.ideal_cols[best_i]
            best_sse = float(self.sse_matrix[t, best_i])

            deviations = np.abs(train_y[:, t] - ideal_y[:, best_i])
            max_dev    = float(deviations.max())
            threshold  = float(np.sqrt(2) * max_dev)

            self.selected.append(
                {
                    "train_col":   train_col,
                    "ideal_col":   best_ideal_col,
                    "ideal_index": best_i,
                    "sse":         best_sse,
                    "max_dev":     max_dev,
                    "threshold":   threshold,
                }
            )

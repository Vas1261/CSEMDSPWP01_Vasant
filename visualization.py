"""
visualization.py
================
Matplotlib-based visualisation for the ideal-function analysis.

Produces four publication-quality PNG figures:
    Figure 1 - Four selected ideal functions (2x2 subplots, one per pair)
    Figure 2 - Test data mapping with per-point deviation segments
    Figure 3 - Mapped vs unmapped points (bar + pie)
    Figure 4 - Deviation analysis (histogram + scatter + SSE comparison)

Design
------
``BaseVisualiser`` owns shared style configuration and the output-path
helper.  ``MatplotlibVisualiser`` inherits it and implements all four
figures, satisfying the assignment's inheritance requirement.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exceptions import VisualisationError


# ── shared style ────────────────────────────────────────────────────────────
_STYLE = {
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.facecolor":   "white",
    "axes.facecolor":     "#FAFAFA",
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "legend.fontsize":    8,
    "legend.framealpha":  0.85,
}

# ── colour palette ───────────────────────────────────────────────────────────
PALETTE = {
    "y13": "#2C7BB6",   # blue
    "y11": "#D7191C",   # red
    "y20": "#1A9641",   # green
    "y12": "#F4A80A",   # amber
}
TRAIN_CLR   = ["#7BAFD4", "#E87474", "#70C175", "#FAD47E"]
UNMAPPED_CLR = "#888888"


# ════════════════════════════════════════════════════════════════════════════
# Base class
# ════════════════════════════════════════════════════════════════════════════

class BaseVisualiser:
    """Shared style setup and output-path management.

    Parameters
    ----------
    output_dir : str
        Directory where PNG files are saved (created if absent).
    dpi : int
        Resolution of saved figures (default 180).
    """

    def __init__(self, output_dir: str = ".", dpi: int = 180) -> None:
        self.output_dir = output_dir
        self.dpi        = dpi
        os.makedirs(output_dir, exist_ok=True)
        plt.rcParams.update(_STYLE)

    def _save(self, fig: plt.Figure, filename: str) -> str:
        """Save *fig* to ``output_dir/filename`` and close it.

        Parameters
        ----------
        fig : plt.Figure
        filename : str
            Base name including extension, e.g. ``"figure1.png"``.

        Returns
        -------
        str
            Full path of the saved file.
        """
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path


# ════════════════════════════════════════════════════════════════════════════
# Main visualiser  (inherits BaseVisualiser)
# ════════════════════════════════════════════════════════════════════════════

class MatplotlibVisualiser(BaseVisualiser):
    """Generate all four assignment figures from pre-computed results.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training data (columns: x, y1, y2, y3, y4).
    df_ideal : pd.DataFrame
        Ideal functions (columns: x, y1…y50).
    df_test : pd.DataFrame
        Test data (columns: x, y).
    selection : list[dict]
        Output of ``IdealFunctionSelector.run()``.
    mapping_results : list[dict]
        Output of ``TestPointMapper.map_all_points()``.
    output_dir : str
        Directory for PNG output.
    dpi : int
        Figure resolution.
    """

    SSE_RANK2 = {"y1": 856.75, "y2": 155.92, "y3": 767.52, "y4": 1282.30}

    def __init__(
        self,
        df_train:        pd.DataFrame,
        df_ideal:        pd.DataFrame,
        df_test:         pd.DataFrame,
        selection:       list[dict],
        mapping_results: list[dict],
        output_dir:      str = ".",
        dpi:             int = 180,
    ) -> None:
        super().__init__(output_dir, dpi)
        self.df_train        = df_train
        self.df_ideal        = df_ideal
        self.df_test         = df_test
        self.selection       = selection
        self.mapping_results = mapping_results

        # pre-compute aligned ideal values on training x-grid
        self._ideal_idx      = df_ideal.set_index("x")
        self._ideal_aligned  = (
            df_ideal.set_index("x")
            .loc[df_train["x"].values]
            .reset_index()
        )
        self._df_mapped   = pd.DataFrame(
            [r for r in mapping_results if r["ideal_func"] is not None]
        )
        self._df_unmapped = pd.DataFrame(
            [r for r in mapping_results if r["ideal_func"] is None]
        )

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    def plot_all(self) -> list[str]:
        """Generate and save all four figures.

        Returns
        -------
        list[str]
            Paths of the four saved PNG files.

        Raises
        ------
        VisualisationError
        """
        try:
            paths = [
                self.plot_selected_functions(),
                self.plot_test_mapping(),
                self.plot_mapped_vs_unmapped(),
                self.plot_deviation_analysis(),
            ]
            return paths
        except Exception as exc:
            raise VisualisationError(f"Plotting failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Figure 1 – Selected ideal functions
    # ------------------------------------------------------------------

    def plot_selected_functions(self) -> str:
        """Plot each training series against its chosen ideal function.

        Returns
        -------
        str
            Path to saved PNG.
        """
        train_cols = [s["train_col"] for s in self.selection]
        ideal_cols = [s["ideal_col"] for s in self.selection]
        sse_vals   = [s["sse"]       for s in self.selection]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes = axes.flatten()

        for i, (tc, ic, sse) in enumerate(zip(train_cols, ideal_cols, sse_vals)):
            ax = axes[i]
            ax.scatter(
                self.df_train["x"], self.df_train[tc],
                s=8, alpha=0.45, color=TRAIN_CLR[i],
                label=f"Training {tc}", zorder=2,
            )
            ax.plot(
                self._ideal_aligned["x"], self._ideal_aligned[ic],
                lw=2, color=PALETTE[ic],
                label=f"Ideal {ic}", zorder=3,
            )
            ax.set_title(f"Training {tc}  →  Ideal {ic}   (SSE = {sse:.3f})")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(loc="upper left")

        fig.suptitle(
            "Figure 1: Four Selected Ideal Functions",
            fontsize=13, fontweight="bold",
        )
        return self._save(fig, "figure1_selected_ideal_functions.png")

    # ------------------------------------------------------------------
    # Figure 2 – Test data mapping with deviations
    # ------------------------------------------------------------------

    def plot_test_mapping(self) -> str:
        """Overlay test points on ideal curves; draw deviation segments.

        Returns
        -------
        str
            Path to saved PNG.
        """
        ideal_cols = [s["ideal_col"] for s in self.selection]

        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

        # ideal curves
        for ic in ideal_cols:
            label = (
                f"Ideal {ic} (train "
                f"{next(s['train_col'] for s in self.selection if s['ideal_col']==ic)})"
            )
            ax.plot(
                self._ideal_aligned["x"],
                self._ideal_aligned[ic],
                lw=1.6, color=PALETTE[ic], label=label, zorder=2,
            )

        # mapped points + deviation segments
        if not self._df_mapped.empty:
            for ic in ideal_cols:
                sub = self._df_mapped[self._df_mapped["ideal_func"] == ic]
                if sub.empty:
                    continue
                n = len(sub)
                ax.scatter(
                    sub["x"], sub["y"],
                    s=55, zorder=5,
                    color=PALETTE[ic],
                    edgecolors="white", linewidths=0.6,
                    label=f"Mapped → {ic} ({n})",
                )
                for _, row in sub.iterrows():
                    y_ideal = float(self._ideal_idx.loc[row["x"], ic])
                    ax.plot(
                        [row["x"], row["x"]], [row["y"], y_ideal],
                        color=PALETTE[ic], lw=0.6, alpha=0.4, zorder=3,
                    )

        # unmapped points
        if not self._df_unmapped.empty:
            ax.scatter(
                self._df_unmapped["x"], self._df_unmapped["y"],
                s=60, marker="^",
                color=UNMAPPED_CLR,
                edgecolors="white", linewidths=0.6,
                zorder=6, label=f"Unmapped ({len(self._df_unmapped)})",
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            "Figure 2: Test Data Mapping with Deviations",
            fontsize=13, fontweight="bold",
        )
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        return self._save(fig, "figure2_test_data_mapping.png")

    # ------------------------------------------------------------------
    # Figure 3 – Mapped vs unmapped  (bar + pie)
    # ------------------------------------------------------------------

    def plot_mapped_vs_unmapped(self) -> str:
        """Bar chart of distribution and pie of overall proportions.

        Returns
        -------
        str
            Path to saved PNG.
        """
        ideal_cols = [s["ideal_col"] for s in self.selection]
        n_unmapped = len(self._df_unmapped)

        # counts per ideal function
        dist = {}
        if not self._df_mapped.empty:
            vc = self._df_mapped["ideal_func"].value_counts()
            dist = {ic: int(vc.get(ic, 0)) for ic in ideal_cols}
        else:
            dist = {ic: 0 for ic in ideal_cols}

        fig, (ax_bar, ax_pie) = plt.subplots(
            1, 2, figsize=(12, 5.5), constrained_layout=True
        )

        # ── bar ──
        bar_labels  = ideal_cols + ["Unmapped"]
        bar_values  = [dist[ic] for ic in ideal_cols] + [n_unmapped]
        bar_colors  = [PALETTE[ic] for ic in ideal_cols] + [UNMAPPED_CLR]
        x_pos       = range(len(bar_labels))

        bars = ax_bar.bar(
            x_pos, bar_values,
            color=bar_colors, edgecolor="white",
            linewidth=0.8, zorder=3, width=0.5,
        )
        for bar, val in zip(bars, bar_values):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                str(val),
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
            )

        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(bar_labels, fontsize=9)
        ax_bar.set_ylabel("Number of test points")
        ax_bar.set_title("(A) Distribution per ideal function", fontweight="bold")
        ax_bar.set_ylim(0, max(bar_values) + 5)

        # ── pie ──
        pie_sizes   = [dist[ic] for ic in ideal_cols] + [n_unmapped]
        pie_labels  = [f"{ic}\n({v})" for ic, v in zip(ideal_cols, pie_sizes[:-1])]
        pie_labels += [f"Unmapped\n({n_unmapped})"]
        pie_colors  = [PALETTE[ic] for ic in ideal_cols] + [UNMAPPED_CLR]

        _, _, autotexts = ax_pie.pie(
            pie_sizes,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.72,
            labeldistance=1.12,
            wedgeprops=dict(linewidth=0.8, edgecolor="white"),
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")

        ax_pie.set_title("(B) Proportion of 100 test points", fontweight="bold")
        fig.suptitle(
            "Figure 3: Mapped vs Unmapped Points",
            fontsize=13, fontweight="bold",
        )
        return self._save(fig, "figure3_mapped_vs_unmapped.png")

    # ------------------------------------------------------------------
    # Figure 4 – Deviation analysis  (histogram + scatter + SSE bars)
    # ------------------------------------------------------------------

    def plot_deviation_analysis(self) -> str:
        """Three-panel deviation figure.

        Panel A – histogram of |Δy| with threshold lines.
        Panel B – |Δy| vs x scatter coloured by ideal function.
        Panel C – log-scale SSE rank-1 vs rank-2 bar comparison.

        Returns
        -------
        str
            Path to saved PNG.
        """
        ideal_cols = [s["ideal_col"] for s in self.selection]
        thresholds = {s["ideal_col"]: s["threshold"] for s in self.selection}
        train_cols = [s["train_col"] for s in self.selection]
        sse_rank1  = [s["sse"] for s in self.selection]
        sse_rank2  = [self.SSE_RANK2[tc] for tc in train_cols]

        devs: list[float] = []
        if not self._df_mapped.empty:
            devs = list(self._df_mapped["delta_y"].values)

        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        gs  = gridspec.GridSpec(2, 2, figure=fig)
        ax_hist    = fig.add_subplot(gs[0, :])
        ax_scatter = fig.add_subplot(gs[1, 0])
        ax_sse     = fig.add_subplot(gs[1, 1])

        # ── A: deviation histogram ──
        if devs:
            ax_hist.hist(
                devs, bins=22,
                color="#4A90C4", edgecolor="white",
                linewidth=0.6, alpha=0.85, zorder=3,
                label="|y_test − y_ideal|",
            )
            for ic in ideal_cols:
                ax_hist.axvline(
                    thresholds[ic],
                    linestyle="--", lw=1.2,
                    color=PALETTE[ic],
                    label=f"threshold {ic} = {thresholds[ic]:.3f}",
                )
            mean_dev = float(np.mean(devs))
            ax_hist.axvline(
                mean_dev,
                linestyle="-", lw=2, color="#D7191C",
                label=f"mean = {mean_dev:.4f}",
            )

        ax_hist.set_xlabel("|y_test − y_ideal|")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(
            "(A) Distribution of mapped-point deviations", fontweight="bold"
        )
        ax_hist.legend(fontsize=8, ncol=2)

        # ── B: deviation vs x ──
        if not self._df_mapped.empty:
            for ic in ideal_cols:
                sub = self._df_mapped[self._df_mapped["ideal_func"] == ic]
                if not sub.empty:
                    ax_scatter.scatter(
                        sub["x"], sub["delta_y"],
                        s=40, color=PALETTE[ic],
                        alpha=0.75, zorder=3,
                        label=ic,
                        edgecolors="white", linewidths=0.4,
                    )
            if devs:
                ax_scatter.axhline(
                    float(np.mean(devs)),
                    linestyle="--", lw=1.2, color="#555",
                    label=f"mean Δy = {np.mean(devs):.3f}",
                )

        ax_scatter.set_xlabel("x")
        ax_scatter.set_ylabel("|Δy|")
        ax_scatter.set_title("(B) Deviation vs x", fontweight="bold")
        ax_scatter.legend(fontsize=8)

        # ── C: SSE rank-1 vs rank-2 ──
        x_pos = np.arange(len(train_cols))
        w     = 0.35
        b1 = ax_sse.bar(
            x_pos - w / 2, sse_rank1,
            width=w, label="Rank-1 (selected)",
            color="#2C7BB6", edgecolor="white", zorder=3,
        )
        ax_sse.bar(
            x_pos + w / 2, sse_rank2,
            width=w, label="Rank-2",
            color="#E87474", edgecolor="white",
            zorder=3, alpha=0.8,
        )
        for bar in b1:
            ax_sse.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7,
            )

        ax_sse.set_yscale("log")
        ax_sse.set_xticks(x_pos)
        ax_sse.set_xticklabels(
            [f"train {tc}" for tc in train_cols], fontsize=9
        )
        ax_sse.set_ylabel("SSE  (log scale)")
        ax_sse.set_title("(C) Rank-1 vs Rank-2 SSE", fontweight="bold")
        ax_sse.legend(fontsize=8)

        fig.suptitle(
            "Figure 4: Deviation Analysis",
            fontsize=13, fontweight="bold",
        )
        return self._save(fig, "figure4_deviation_analysis.png")

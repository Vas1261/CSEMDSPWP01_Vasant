"""
main.py
=======
Entry point for the IU Programming with Python assignment.

Execution order
---------------
1.  DataLoader  - read and validate all three CSV files.
2.  DataLoader  - create SQLite tables; persist training + ideal data.
3.  IdealFunctionSelector - compute 4x50 SSE matrix; select best ideal.
4.  TestPointMapper       - apply sqrt(2) threshold; classify test points.
5.  DataLoader  - persist mapping results to test_mapping table.
6.  MatplotlibVisualiser  - generate and save all four figures.
7.  Print summary to stdout.

Usage
-----
    python main.py

All CSV files (train.csv, ideal.csv, test.csv) must reside in the same
directory as main.py, or absolute paths must be supplied via the constants
below.  The SQLite database and PNG figures are written to the working
directory.
"""

from __future__ import annotations

import os
import sys

from database      import DataLoader
from exceptions    import IUBaseError
from mapper        import TestPointMapper
from selector      import IdealFunctionSelector
from visualization import MatplotlibVisualiser


# ── file paths (edit if data lives elsewhere) ─────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
IDEAL_CSV = os.path.join(BASE_DIR, "ideal.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test.csv")
DB_PATH   = os.path.join(BASE_DIR, "results.db")
OUT_DIR   = os.path.join(BASE_DIR, "output_figures")


def main() -> None:
    """Run the complete IU Python assignment pipeline."""

    print("=" * 62)
    print("  IU CSEMDSPWP01 – Ideal Function Selection Pipeline")
    print("=" * 62)

    # ── step 1 & 2 : load CSVs and persist raw data ───────────────────
    print("\n[1/5] Loading CSV files and initialising database …")
    try:
        loader = DataLoader(
            db_path   = DB_PATH,
            train_csv = TRAIN_CSV,
            ideal_csv = IDEAL_CSV,
            test_csv  = TEST_CSV,
        )
        loader.run_full_pipeline()          # drop/create tables; load CSVs; persist
    except IUBaseError as exc:
        print(f"ERROR during data loading: {exc}")
        sys.exit(1)

    df_train = loader.df_train
    df_ideal = loader.df_ideal
    df_test  = loader.df_test

    print(f"    Training rows  : {len(df_train)}")
    print(f"    Ideal rows     : {len(df_ideal)}")
    print(f"    Test rows      : {len(df_test)}")

    # ── step 3 : select ideal functions ──────────────────────────────
    print("\n[2/5] Computing SSE matrix and selecting ideal functions …")
    try:
        selector = IdealFunctionSelector(df_train, df_ideal)
        selection = selector.run()
    except IUBaseError as exc:
        print(f"ERROR during ideal-function selection: {exc}")
        sys.exit(1)

    print("\n    Selected functions:")
    print(f"    {'Training':<12} {'Ideal':<10} {'SSE':>12} {'Max|dev|':>12} {'Threshold':>12}")
    print("    " + "-" * 62)
    for s in selection:
        print(
            f"    {s['train_col']:<12} {s['ideal_col']:<10} "
            f"{s['sse']:>12.6f} {s['max_dev']:>12.6f} {s['threshold']:>12.6f}"
        )

    # also print the full SSE summary
    print("\n    SSE summary table (training × ideal):")
    print(selector.summary_dataframe().to_string())

    # ── step 4 : map test points ──────────────────────────────────────
    print("\n[3/5] Mapping test points …")
    try:
        mapper = TestPointMapper(df_test, df_ideal, selection)
        results = mapper.map_all_points()
    except IUBaseError as exc:
        print(f"ERROR during test-point mapping: {exc}")
        sys.exit(1)

    summary = mapper.summary()
    print(f"    Total test points  : {summary['total']}")
    print(f"    Mapped             : {summary['mapped']}")
    print(f"    Unmapped           : {summary['unmapped']}")
    print(f"    Mean |delta_y|     : {summary['mean_delta_y']:.6f}")
    print(f"    Max  |delta_y|     : {summary['max_delta_y']:.6f}")
    print(f"    Min  |delta_y|     : {summary['min_delta_y']:.6f}")
    print(f"    Std  |delta_y|     : {summary['std_delta_y']:.6f}")
    print("\n    Distribution:")
    for ic, cnt in mapper.distribution().items():
        print(f"      {ic}: {cnt} points")

    # ── step 5 : persist mapping results ─────────────────────────────
    print("\n[4/5] Persisting mapping results to database …")
    try:
        loader.persist_test_mapping(results)
    except IUBaseError as exc:
        print(f"ERROR persisting mapping results: {exc}")
        sys.exit(1)
    print(f"    Saved to: {DB_PATH}")

    # ── step 6 : generate figures ─────────────────────────────────────
    print("\n[5/5] Generating visualisations …")
    try:
        visualiser = MatplotlibVisualiser(
            df_train        = df_train,
            df_ideal        = df_ideal,
            df_test         = df_test,
            selection       = selection,
            mapping_results = results,
            output_dir      = OUT_DIR,
            dpi             = 180,
        )
        paths = visualiser.plot_all()
    except IUBaseError as exc:
        print(f"ERROR during visualisation: {exc}")
        sys.exit(1)

    for p in paths:
        print(f"    Saved: {p}")

    print("\n" + "=" * 62)
    print("  Pipeline complete.")
    print("=" * 62)


if __name__ == "__main__":
    main()

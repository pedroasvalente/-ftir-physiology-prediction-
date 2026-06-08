#!/usr/bin/env python
"""
Run Mann-Whitney U + AUC spectral comparison: high-fitness vs low-fitness.

Usage:
    python scripts/run_spectral_comparison.py
    python scripts/run_spectral_comparison.py --matrices SERUM CAPILAR
    python scripts/run_spectral_comparison.py --timepoints 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ftir_pred.analysis.spectral_comparison import run_all_matrices
from ftir_pred.config import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectral group comparison (Mann-Whitney + AUC)")
    parser.add_argument(
        "--matrices", nargs="+",
        default=["CAPILAR", "PLASMA", "SALIVA", "SERUM", "URINE"],
        help="Sample matrices to include",
    )
    parser.add_argument(
        "--timepoints", nargs="+", type=int, default=None,
        help="Restrict to specific timepoints (default: all)",
    )
    parser.add_argument(
        "--group-column", default="vo2max_classes_simplified",
        help="Column with fitness group labels",
    )
    parser.add_argument("--high-label", type=float, default=3.0, help="High-fitness label value")
    parser.add_argument("--low-label",  type=float, default=1.0, help="Low-fitness label value")
    parser.add_argument("--alpha",       type=float, default=0.05, help="FDR significance threshold")
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "spectral_comparison"),
        help="Output directory",
    )
    args = parser.parse_args()

    print("Spectral comparison: high vs low VO2max")
    print(f"  Matrices   : {args.matrices}")
    print(f"  Timepoints : {args.timepoints or 'all'}")
    print(f"  Group col  : {args.group_column} (high={args.high_label}, low={args.low_label})")
    print(f"  FDR alpha  : {args.alpha}")
    print(f"  Output     : {args.out_dir}")
    print()

    run_all_matrices(
        sample_types=args.matrices,
        group_column=args.group_column,
        high_label=args.high_label,
        low_label=args.low_label,
        timepoints=args.timepoints,
        alpha=args.alpha,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()

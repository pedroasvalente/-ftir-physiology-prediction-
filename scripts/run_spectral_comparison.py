#!/usr/bin/env python
"""
Spectral region comparison: high vs low VO2max.

python scripts/run_spectral_comparison.py
python scripts/run_spectral_comparison.py --matrices SERUM CAPILAR
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ftir_pred.analysis.spectral_comparison import run_all_matrices
from ftir_pred.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", nargs="+",
                        default=["CAPILAR", "PLASMA", "SALIVA", "SERUM", "URINE"])
    parser.add_argument("--timepoints", nargs="+", type=int, default=[1])
    parser.add_argument("--top-pct",    type=float, default=20.0)
    parser.add_argument("--min-pts",    type=int,   default=5)
    parser.add_argument("--out-dir",    default=str(RESULTS_DIR / "spectral_comparison"))
    args = parser.parse_args()

    run_all_matrices(
        sample_types=args.matrices,
        timepoints=args.timepoints,
        top_pct=args.top_pct,
        min_consecutive=args.min_pts,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()

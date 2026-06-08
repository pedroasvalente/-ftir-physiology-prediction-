#!/usr/bin/env python
"""
Run PLS-R direct regression for all target × sample_type combinations.

This is the standard quimiometric baseline. Uses GroupKFold cross-validation
(person-aware) to select the optimal number of PLS components.

Usage:
    python scripts/run_plsr.py
    python scripts/run_plsr.py --targets vo2max_absolute vo2max_relative
    python scripts/run_plsr.py --matrices SERUM CAPILAR --max-components 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ftir_pred.analysis.plsr_direct import run_all_plsr
from ftir_pred.config import RESULTS_DIR
from ftir_pred.data.config import REGRESSION_TARGETS


def main() -> None:
    parser = argparse.ArgumentParser(description="PLS-R direct regression")
    parser.add_argument(
        "--targets", nargs="+", default=None,
        help="Targets to predict (default: all 33)",
    )
    parser.add_argument(
        "--matrices", nargs="+",
        default=["CAPILAR", "PLASMA", "SALIVA", "SERUM", "URINE"],
        help="Sample matrices",
    )
    parser.add_argument("--max-components", type=int, default=15, help="Max PLS components")
    parser.add_argument("--n-splits",       type=int, default=5,  help="CV folds")
    parser.add_argument(
        "--timepoints", nargs="+", type=int, default=None,
        help="Restrict to specific timepoints",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "plsr_results"),
        help="Output directory",
    )
    args = parser.parse_args()

    targets = args.targets or REGRESSION_TARGETS
    print("PLS-R direct regression")
    print(f"  Targets        : {len(targets)}")
    print(f"  Matrices       : {args.matrices}")
    print(f"  Max components : {args.max_components}")
    print(f"  CV folds       : {args.n_splits}")
    print(f"  Timepoints     : {args.timepoints or 'all'}")
    print(f"  Output         : {args.out_dir}")
    print()

    run_all_plsr(
        targets=targets,
        sample_types=args.matrices,
        max_components=args.max_components,
        n_splits=args.n_splits,
        timepoints=args.timepoints,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()

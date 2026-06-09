#!/usr/bin/env python
"""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets",        nargs="+", type=str, default=None)
    parser.add_argument("--matrices",       nargs="+", default=["CAPILAR", "PLASMA", "SALIVA", "SERUM", "URINE"])
    parser.add_argument("--max-components", type=int,  default=15)
    parser.add_argument("--n-splits",       type=int,  default=5)
    parser.add_argument("--timepoints",     nargs="+", type=int, default=None)
    parser.add_argument("--out-dir",        default=str(RESULTS_DIR / "plsr_results"))
    args = parser.parse_args()

    run_all_plsr(
        targets=args.targets or REGRESSION_TARGETS,
        sample_types=args.matrices,
        max_components=args.max_components,
        n_splits=args.n_splits,
        timepoints=args.timepoints,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()

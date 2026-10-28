#!/usr/bin/env bash
# Full MIMIC-III pipeline of Section 6 (see README.md). Requires MIMIC_DIR.
set -euo pipefail
cd "$(dirname "$0")"
: "${MIMIC_DIR:?set MIMIC_DIR to the folder with processed-data/ and raw-data/}"
python clean_cohort.py
python build_long_format.py --horizon-days 730 --tag 2y
python scale_features.py --tag 2y
python fit_final_horizon.py 2y
python resample_bands.py 2y 8000 1
python bootstrap_final_lr.py 2y 200 10
python plot_final_horizon.py 2y
python export_results.py 2y

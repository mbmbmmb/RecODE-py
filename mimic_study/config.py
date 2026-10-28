"""Paths shared by the MIMIC-III analysis scripts.

MIMIC_DIR  folder with the input data (set the environment variable MIMIC_DIR):
               processed-data/input.csv, processed-data/valid_aids.csv
                   output of the MIMIC-III benchmark pipeline of Purushotham et al. (2018),
                   https://github.com/USC-Melady/Benchmarking_DL_MIMICIII
               raw-data/ADMISSIONS.csv
                   the ADMISSIONS table of MIMIC-III v1.4
OUT        patient-level intermediate files and fits (never commit; see .gitignore)
RESULTS    aggregate results reported in the paper (committed)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MIMIC_DIR = os.environ.get('MIMIC_DIR', os.path.join(HERE, 'data'))
SRC = os.path.join(MIMIC_DIR, 'processed-data')
RAW = os.path.join(MIMIC_DIR, 'raw-data')
OUT = os.environ.get('MIMIC_OUT', os.path.join(HERE, 'output'))
RESULTS = os.path.join(HERE, 'results')

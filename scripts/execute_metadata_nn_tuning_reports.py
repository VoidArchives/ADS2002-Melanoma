"""Compatibility dispatch for current ROC-selected version-2 saved reports.

The eight familiar NN/report integration targets remain the default. Execution
is owned by execute_metadata_roc_reports, whose receipt pins version 2 and
ROC-AUC and whose kernel guard blocks fitting, inference and reserved-test
access. Historical AP execution is available only in the preserved archive.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=[
    "notebooks/Subset/06_metadata_neural_network.ipynb",
    "notebooks/Subset/07_metadata_tuning.ipynb",
    "notebooks/Subset/08_metadata_robustness.ipynb",
    "notebooks/Subset/14_image_nested_evaluation.ipynb",
    "notebooks/Subset_dedup/06_metadata_neural_network.ipynb",
    "notebooks/Subset_dedup/07_metadata_tuning.ipynb",
    "notebooks/Subset_dedup/08_metadata_robustness.ipynb",
    "notebooks/Subset_dedup/12_combined_evaluation.ipynb",
]


def forwarded_arguments(arguments):
    arguments=list(arguments)
    if any(a=="--notebook" or a.startswith("--notebook=") for a in arguments) or any(a in {"--help","-h"} for a in arguments):
        return arguments
    return arguments+[item for path in NOTEBOOKS for item in ("--notebook",path)]


def main():
    from execute_metadata_roc_reports import main as current_main
    previous=sys.argv
    sys.argv=[str(ROOT/"scripts/execute_metadata_roc_reports.py"),*forwarded_arguments(previous[1:])]
    try:
        current_main()
    finally:
        sys.argv=previous


if __name__=="__main__":
    main()

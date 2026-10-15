"""Melanoma detection from dermoscopic images and patient metadata."""

__version__ = "0.1.0"

# Only re-export torch-free modules here so `import melanoma_detection`
# stays light; import data / image_model / fusion_model explicitly.
from .evaluation import compute_metrics, select_threshold
from .splitting import assert_no_patient_overlap, patient_level_split

__all__ = [
    "__version__",
    "assert_no_patient_overlap",
    "compute_metrics",
    "patient_level_split",
    "select_threshold",
]
"""Freeze the development-only exact-copy variant; never trains a model."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_audit import freeze_development_variant

if __name__ == "__main__":
    print(json.dumps(freeze_development_variant(ROOT), indent=2))

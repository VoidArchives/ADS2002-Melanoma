"""Create/execute the separate clean report notebooks; never start training."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melanoma_detection.dedup_notebooks import build_notebooks, execute_notebooks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--notebooks", nargs="+", help="With --execute-only, refresh selected notebook filenames after a complete execution.")
    args = parser.parse_args()
    if args.notebooks and not args.execute_only:
        parser.error("--notebooks requires --execute-only to preserve other executed reports")
    if not args.execute_only:
        print(f"Built {len(build_notebooks(ROOT))} clean report notebooks.", flush=True)
    if args.execute or args.execute_only:
        execute_notebooks(ROOT, timeout=args.timeout, notebook_names=args.notebooks)


if __name__ == "__main__":
    main()

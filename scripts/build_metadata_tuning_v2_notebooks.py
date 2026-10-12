"""Build ten ROC-AUC-first metadata reports without training or inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from melanoma_detection.metadata_tuning_v2_reporting import NAMES,DEFAULT_RUN_ID,build_notebook,original_snapshot
from build_metadata_nn_tuning_notebooks import render_html


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--condition",choices=("original","dedup_v1","both"),default="both")
    parser.add_argument("--stage",choices=("04","05","06","07","08"),action="append")
    parser.add_argument("--run-id",default=DEFAULT_RUN_ID)
    parser.add_argument("--html",action="store_true")
    parser.add_argument("--html-only",action="store_true",help="Preserve execution outputs while rendering previews")
    args=parser.parse_args()
    root=args.root.resolve()
    conditions=("original","dedup_v1") if args.condition=="both" else (args.condition,)
    for condition in conditions:
        folder="Subset" if condition=="original" else "Subset_dedup"
        for stage in args.stage or list(NAMES[condition]):
            name=NAMES[condition][stage]
            relative=f"notebooks/{folder}/{name}"
            original_snapshot(root,relative)
            path=root/relative
            notebook=nbformat.read(path,as_version=4) if args.html_only else build_notebook(root,condition,stage,args.run_id)
            if not args.html_only:
                temporary=path.with_suffix(".build.tmp")
                nbformat.write(notebook,temporary)
                temporary.replace(path)
            print(relative,flush=True)
            if args.html or args.html_only:
                destination=root/"results/metadata_nn_tuning/v2/report_previews"/folder/path.with_suffix(".html").name
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_text(render_html(notebook,f"ROC-first {condition}: {name}"),encoding="utf-8")
                print(destination.relative_to(root),flush=True)


if __name__=="__main__":
    main()

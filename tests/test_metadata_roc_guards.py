"""Synthetic strict-archive and subprocess report-guard acceptance checks."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from melanoma_detection import metadata_roc_archive as archive


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


class StrictArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.relative="src/synthetic_reporting.py"
        self.current=self.root / self.relative
        self.current.parent.mkdir(parents=True)
        self.current.write_text("changed current implementation",encoding="utf-8")
        self.snapshot=self.root / archive.ARCHIVE / self.relative
        self.snapshot.parent.mkdir(parents=True)
        self.snapshot.write_text("exact historical implementation",encoding="utf-8")
        self.old_hash=digest(self.snapshot)
        self.parent_relative="results/historical_run/receipt.json"
        self.parent=self.root / self.parent_relative
        self.source_record={self.relative:self.old_hash}
        self.snapshot_receipt=self.root / archive.ARCHIVE / "snapshot_receipt.json"
        write_json(self.snapshot_receipt,{"status":"complete","notebook_snapshots":{},"source_snapshots":{self.relative:self.old_hash}})
        self.preservation=self.root / archive.ARCHIVE / "preservation_receipt.json"
        self.preserve_parent()

    def preserve_parent(self):
        write_json(self.parent,{"status":"complete","source_sha256":self.source_record})
        write_json(self.preservation,{"status":"complete","protected_files":{self.parent_relative:digest(self.parent)},"notebook_snapshots":{},"source_snapshots":{self.relative:self.old_hash},"ap_parent_runs":{}})

    def test_archived_source_requires_exact_preserved_parent_and_bytes(self):
        self.assertEqual(archive.original_snapshot(self.root,self.relative),self.snapshot)
        self.assertEqual(archive.verified_source_path(self.root,self.relative,self.old_hash,parent_receipt=self.parent_relative),self.snapshot)
        self.assertEqual(archive.verified_source_path(self.root,self.relative,digest(self.current)),self.current)
        with self.assertRaisesRegex(ValueError,"explicit historical receipt"):
            archive.verified_source_path(self.root,self.relative,self.old_hash)

    def test_snapshot_byte_tamper_rejected_by_both_readers(self):
        self.snapshot.write_text("tampered archive",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"missing or changed"):
            archive.original_snapshot(self.root,self.relative)
        with self.assertRaisesRegex(ValueError,"missing or changed"):
            archive.verified_source_path(self.root,self.relative,self.old_hash,parent_receipt=self.parent)

    def test_rehashed_snapshot_cannot_override_historical_parent_digest(self):
        self.snapshot.write_text("different implementation",encoding="utf-8")
        write_json(self.snapshot_receipt,{"notebook_snapshots":{},"source_snapshots":{self.relative:digest(self.snapshot)}})
        with self.assertRaisesRegex(ValueError,"do not match"):
            archive.verified_source_path(self.root,self.relative,self.old_hash,parent_receipt=self.parent)

    def test_changed_or_unpreserved_parent_and_unknown_source_rejected(self):
        self.parent.write_text(json.dumps({"status":"complete","source_sha256":self.source_record,"changed":True}),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"changed or not preserved"):
            archive.verified_source_path(self.root,self.relative,self.old_hash,parent_receipt=self.parent)
        self.preserve_parent()
        unpreserved=self.root / "results/arbitrary_receipt.json"
        write_json(unpreserved,{"source_sha256":self.source_record})
        with self.assertRaisesRegex(ValueError,"changed or not preserved"):
            archive.verified_source_path(self.root,self.relative,self.old_hash,parent_receipt=unpreserved)
        arbitrary="src/arbitrary_training.py"
        (self.root / arbitrary).write_text("current arbitrary implementation",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"does not bind"):
            archive.verified_source_path(self.root,arbitrary,self.old_hash,parent_receipt=self.parent)
        with self.assertRaisesRegex(ValueError,"missing or changed"):
            archive.original_snapshot(self.root,arbitrary)

    def test_source_receipt_and_archive_paths_cannot_escape(self):
        for source in ("../outside.py",str(self.root.parent / "outside.py")):
            with self.subTest(source=source),self.assertRaisesRegex(ValueError,"escapes"):
                archive.original_snapshot(self.root,source)
            with self.subTest(source=source),self.assertRaisesRegex(ValueError,"escapes"):
                archive.verified_source_path(self.root,source,self.old_hash,parent_receipt=self.parent)
        for parent in ("../outside_receipt.json","configs/fake_receipt.json"):
            with self.subTest(parent=parent),self.assertRaises(ValueError):
                archive.verified_source_path(self.root,self.relative,self.old_hash,parent_receipt=parent)

    def test_windows_source_keys_are_unambiguous_and_exact(self):
        self.source_record={self.relative.replace("/","\\"):self.old_hash}
        self.preserve_parent()
        self.assertEqual(archive.verified_source_path(self.root,self.relative,self.old_hash,parent_receipt=self.parent),self.snapshot)
        self.source_record[self.relative]=self.old_hash
        self.preserve_parent()
        with self.assertRaisesRegex(ValueError,"ambiguous normalized"):
            archive.verified_source_path(self.root,self.relative,self.old_hash,parent_receipt=self.parent)


def guard_source_for(root):
    """Compile the actual guard function without importing an experiment runner."""
    path=ROOT / "scripts/execute_metadata_roc_reports.py"
    tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="kernel_guard")
    namespace={"ROOT":root}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),"exec"),namespace)
    return namespace["kernel_guard"]()


class ReportKernelGuardTests(unittest.TestCase):
    def test_actual_kernel_guard_in_isolated_synthetic_repo(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root=Path(directory)
            package=root / "src/melanoma_detection"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("",encoding="utf-8")
            (package / "dedup_metadata.py").write_text('def fit_network(*args, **kwargs):\n    raise AssertionError("unblocked neural fit")\n',encoding="utf-8")
            (package / "image_uncertainty.py").write_text('\n'.join(f'def {name}(*args, **kwargs):\n    raise AssertionError("unblocked image inference")\n' for name in ("run_smoke","run_inference","load_selected_checkpoint","prepare_tensor_cache","predict_cached")),encoding="utf-8")
            historical=root / "results/historical/predictions.csv"
            historical.parent.mkdir(parents=True)
            historical.write_text("historical evidence\n",encoding="utf-8")
            development=root / "data/subset/processed/train_folds.csv"
            development.parent.mkdir(parents=True)
            development.write_text("development evidence\n",encoding="utf-8")
            reserved=root / "data/subset/test/labels.csv"
            reserved.parent.mkdir(parents=True)
            reserved.write_text("reserved evidence\n",encoding="utf-8")
            reserved_named=root / "data/subset/processed/test_labels.csv"
            reserved_named.write_text("reserved evidence\n",encoding="utf-8")
            archived=root / archive.ARCHIVE / "src/frozen.py"
            archived.parent.mkdir(parents=True)
            archived.write_text("frozen historical source\n",encoding="utf-8")
            temporary=root / "results/metadata_nn_tuning/v2/.temporary"
            temporary.mkdir(parents=True)
            (root / "results/dedup_v1/metadata_nn_tuning/v2").mkdir(parents=True)
            actions='''
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from melanoma_detection import dedup_metadata
from melanoma_detection import image_uncertainty
checks=[]
def expect_block(label, error, action):
    try: action()
    except error: checks.append(label)
    else: raise AssertionError(label + " was permitted")
assert historical.read_text() == "historical evidence\\n"
assert development.read_text() == "development evidence\\n"
assert archived.read_text() == "frozen historical source\\n"
checks.extend(["historical read allowed","development read allowed","archive read allowed"])
expect_block("historical write denied",PermissionError,lambda:historical.write_text("changed"))
expect_block("historical append denied",PermissionError,lambda:open(historical,"a"))
expect_block("historical mkdir denied",PermissionError,lambda:(historical.parent / "new").mkdir())
expect_block("historical delete denied",PermissionError,historical.unlink)
allowed=_report_workspace / "results/metadata_nn_tuning/v2/report_verification/fixture"
allowed.mkdir(parents=True,exist_ok=True)
new=allowed / "new.txt"
new.write_text("v2 evidence")
assert new.read_text() == "v2 evidence"
renamed=allowed / "renamed.txt"
new.rename(renamed)
checks.append("v2 write/mkdir/rename allowed")
expect_block("historical rename source denied",PermissionError,lambda:historical.rename(allowed / "old.txt"))
expect_block("historical rename destination denied",PermissionError,lambda:renamed.rename(historical.parent / "new.txt"))
renamed.unlink()
checks.append("v2 delete allowed")
clean=_report_workspace / "results/dedup_v1/metadata_nn_tuning/v2/fixture"
clean.mkdir(parents=True,exist_ok=True)
(clean / "new.txt").write_text("clean v2 evidence")
checks.append("clean v2 write allowed")
expect_block("reserved directory read denied",PermissionError,reserved.read_bytes)
expect_block("reserved named table read denied",PermissionError,reserved_named.read_text)
expect_block("archive snapshot write denied",PermissionError,lambda:archived.write_text("changed archive"))
expect_block("archive snapshot delete denied",PermissionError,archived.unlink)
expect_block("archive snapshot mkdir denied",PermissionError,lambda:(archived.parent / "new").mkdir())
expect_block("archive snapshot rename source denied",PermissionError,lambda:archived.rename(allowed / "frozen.py"))
rename_candidate=allowed / "rename_candidate.py"
rename_candidate.write_text("new report")
expect_block("archive snapshot rename destination denied",PermissionError,lambda:rename_candidate.rename(archived.parent / "new.py"))
fit_alias=getattr(LogisticRegression(),"fit")
expect_block("indirect sklearn fit denied",RuntimeError,lambda:fit_alias([[0],[1]],[0,1]))
pipeline=Pipeline([("scale",StandardScaler()),("classifier",LogisticRegression())])
expect_block("pipeline fit denied",RuntimeError,lambda:pipeline.fit([[0],[1]],[0,1]))
nn_alias=getattr(dedup_metadata,"fit_network")
expect_block("indirect neural fit denied",RuntimeError,lambda:nn_alias(None,None,42))
for name in ("run_smoke","run_inference","load_selected_checkpoint","prepare_tensor_cache","predict_cached"):
    alias=getattr(image_uncertainty,name)
    expect_block("indirect image " + name + " denied",RuntimeError,lambda:alias(None))
print(json.dumps({"checks":checks}))
'''
            script='import json\nfrom pathlib import Path\n'+"\n".join(f"{name}=Path({str(path)!r})" for name,path in [("historical",historical),("development",development),("reserved",reserved),("reserved_named",reserved_named),("archived",archived)])+'\n'+guard_source_for(root)+'\n'+actions
            environment=os.environ.copy()
            environment.update({"PYTHONDONTWRITEBYTECODE":"1","TMP":str(temporary),"TEMP":str(temporary)})
            completed=subprocess.run([sys.executable,"-B","-c",script],cwd=root,env=environment,text=True,encoding="utf-8",capture_output=True,timeout=60)
            self.assertEqual(completed.returncode,0,completed.stdout+'\n'+completed.stderr)
            result=json.loads(completed.stdout.splitlines()[-1])
            self.assertEqual(len(result["checks"]),27)
            self.assertEqual(historical.read_text(encoding="utf-8"),"historical evidence\n")
            self.assertEqual(archived.read_text(encoding="utf-8"),"frozen historical source\n")


if __name__=="__main__":unittest.main()

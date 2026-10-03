"""Execute the image-results notebook using this Python environment.

The notebook defaults to saved results. This script does not change its explicit
TRAIN_PILOT flag or global kernel registrations.
"""

from pathlib import Path
import argparse
import sys

from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--notebook", choices=["05_image_model_d1.ipynb",
                    "05e_image_architecture_comparison_d1.ipynb"],
                    default="05_image_model_d1.ipynb")
args = parser.parse_args()
path = ROOT / "notebooks" / args.notebook
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv[0] = sys.executable
client = NotebookClient(notebook, km=manager, timeout=600,
                        resources={"metadata": {"path": str(ROOT / "notebooks")}})
client.execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Executed {path} in {sys.executable}")

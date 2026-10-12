"""Execute a selected current image notebook using this Python environment.

Reporting notebooks read saved results; the uncertainty notebook runs inference.
This script preserves notebook flags and does not change global kernel registrations.
"""

from pathlib import Path
import argparse
import sys

from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--notebook", choices=["Subset/09_image_b0_pilot.ipynb",
                    "Subset/10_image_architecture_comparison.ipynb",
                    "Subset/13_image_uncertainty.ipynb"],
                    default="Subset/09_image_b0_pilot.ipynb")
args = parser.parse_args()
path = ROOT / "notebooks" / args.notebook
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv[0] = sys.executable
client = NotebookClient(notebook, km=manager, timeout=600,
                        resources={"metadata": {"path": str(path.parent)}})
client.execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Executed {path} in {sys.executable}")

"""Cache the two declared pretrained weights without constructing or training models."""

from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json

import torch
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
cache = ROOT / "models/image_pretrained/checkpoints"
cache.mkdir(parents=True, exist_ok=True)
records = []
for name, weights in (
    ("densenet121", models.DenseNet121_Weights.IMAGENET1K_V1),
    ("convnext_tiny", models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
):
    filename = Path(urlparse(weights.url).path).name
    prefix = filename.rsplit("-", 1)[1].split(".")[0]
    path = cache / filename
    if not path.is_file():
        torch.hub.download_url_to_file(weights.url, str(path), hash_prefix=prefix, progress=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(prefix):
        raise ValueError(f"Cached weight hash differs for {name}")
    records.append({"architecture": name, "weights": "IMAGENET1K_V1",
                    "url": weights.url, "path": str(path), "sha256": digest,
                    "bytes": path.stat().st_size})
receipt = cache.parent / "weight_receipt.json"
receipt.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"cached": records, "receipt": str(receipt)}, indent=2))

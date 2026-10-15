"""Single source of TorchVision model, preprocessing and short-budget LR presets."""
from dataclasses import dataclass
from typing import Callable

from torch import nn
from torchvision import models

TARGET_EFFECTIVE_BATCH_SIZE = 128
NUMBER_OF_EPOCHS = 3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
MICROBATCH_CANDIDATES = (128, 64, 32, 16, 8)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable
    weights: object
    head_path: str
    classifier_path: str
    fallback_size: int
    backbone_lr: float
    head_lr: float
    dropout: float = 0.2

    @property
    def input_size(self):
        crop = getattr(self.weights.transforms(), "crop_size", None)
        return int(crop[0]) if crop else self.fallback_size

    @property
    def normalization(self):
        transform = self.weights.transforms()
        return transform.mean, transform.std

    @property
    def weight_enum(self):
        return f"{type(self.weights).__name__}.{self.weights.name}"

    def build(self, pretrained=True):
        model = self.builder(weights=self.weights if pretrained else None)

        old = model.get_submodule(self.head_path)
        parent_path, _, attribute = self.head_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model

        new_linear = nn.Linear(old.in_features, 1)

        # EfficientNet already has:
        # classifier = Sequential(Dropout, Linear)
        if self.name.startswith("efficientnet"):
            model.classifier[0] = nn.Dropout(
                p=self.dropout,
                inplace=True,
            )
            model.classifier[1] = new_linear

        # Everything else gets Dropout immediately before final Linear
        else:
            setattr(
                parent,
                attribute,
                nn.Sequential(
                    nn.Dropout(p=self.dropout),
                    new_linear,
                ),
            )

        model._image_model_name = self.name
        model._image_classifier_path = self.classifier_path
        return model


MODEL_REGISTRY = {
    s.name: s for s in (
        ModelSpec("resnet18", models.resnet18, models.ResNet18_Weights.DEFAULT, "fc", "fc", 224, 1e-4, 1e-3, 0.2),
        ModelSpec("resnet34", models.resnet34, models.ResNet34_Weights.DEFAULT, "fc", "fc", 224, 1e-4, 1e-3, 0.2),
        ModelSpec("resnet50", models.resnet50, models.ResNet50_Weights.DEFAULT, "fc", "fc", 224, 5e-5, 1e-3, 0.2),
        ModelSpec("efficientnet_b0", models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT,
                  "classifier.1", "classifier", 224, 1e-4, 1e-3, 0.2),
        ModelSpec("efficientnet_b3", models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT,
                  "classifier.1", "classifier", 300, 5e-5, 1e-3, 0.2),
        ModelSpec("efficientnet_b4", models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT,
                  "classifier.1", "classifier", 380, 3e-5, 5e-4, 0.2),
        ModelSpec("densenet121", models.densenet121, models.DenseNet121_Weights.DEFAULT,
                  "classifier", "classifier", 224, 1e-4, 1e-3, 0.2),
        ModelSpec("convnext_tiny", models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT,
                  "classifier.2", "classifier", 224, 3e-5, 5e-4, 0.2),
    )
}
SUPPORTED_MODELS = set(MODEL_REGISTRY)
# Explicit fallback values and presets are derived from the one registry.
MODEL_INPUT_SIZE = {name: spec.fallback_size for name, spec in MODEL_REGISTRY.items()}
TRAINING_PRESETS = {name: {"backbone_lr": spec.backbone_lr, "head_lr": spec.head_lr}
                    for name, spec in MODEL_REGISTRY.items()}


def get_model_spec(name):
    try:
        return MODEL_REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(f"model_name must be one of {sorted(MODEL_REGISTRY)}") from None

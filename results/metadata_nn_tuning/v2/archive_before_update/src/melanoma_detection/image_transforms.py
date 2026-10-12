"""Versioned preparation profiles for controlled image experiments.

These fixed profiles have no learned state. CLAHE is an optional, lazy dependency
and never changes the unchanged-image profiles when OpenCV is installed later.
"""

from __future__ import annotations

import copy
import importlib

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF


AUGMENTATION_PROFILES = {
    "weak": {
        "version": "weak_v1",
        "parameters": {"horizontal_flip_probability": 0.5,
                       "vertical_flip_probability": 0.5,
                       "rotation_degrees": [-20.0, 20.0],
                       "rotation_interpolation": "nearest", "rotation_fill": 0},
    },
    "stronger": {
        "version": "stronger_v1",
        "parameters": {"horizontal_flip_probability": 0.5,
                       "vertical_flip_probability": 0.5,
                       "rotation_degrees": [-45.0, 45.0],
                       "translation_fraction": [0.05, 0.05],
                       "translation_reference": "original_resized_image",
                       "scale": [0.9, 1.05], "shear_degrees": [0.0, 0.0],
                       "reflection_padding_pixels": 112,
                       "affine_interpolation": "bilinear",
                       "brightness": [0.8, 1.2], "contrast": [0.8, 1.2],
                       "saturation": [0.9, 1.1], "hue": [0.0, 0.0]},
    },
}
PREPROCESSING_PROFILES = {
    "none": {"version": "none_v1", "parameters": {}},
    "clahe": {"version": "clahe_lab_l_v1", "parameters": {
        "colour_space": "CIELAB", "channel": "lightness_only",
        "clip_limit": 2.0, "tile_grid_size": [8, 8],
        "position": "after_resize_before_stochastic_augmentation",
        "learned_state": False}},
}


def validate_profiles(spec):
    """Reject unsupported identities or changed parameters instead of guessing."""
    for name, profiles in (("augmentation", AUGMENTATION_PROFILES),
                           ("preprocessing", PREPROCESSING_PROFILES)):
        candidate = spec[name]
        if candidate not in profiles:
            raise ValueError(f"Unsupported {name}: {candidate!r}; registered profiles are {tuple(profiles)}.")
        profile = profiles[candidate]
        version = spec.get(f"{name}_version", profile["version"])
        if version != profile["version"]:
            raise ValueError(f"Unsupported {name} version: {version!r}.")
        parameters = spec.get(f"{name}_parameters", profile["parameters"])
        if parameters != profile["parameters"]:
            raise ValueError(f"Changed {name} parameters require a new registered profile version.")


def profile_identity(spec):
    validate_profiles(spec)
    return {
        "augmentation": copy.deepcopy(AUGMENTATION_PROFILES[spec["augmentation"]]),
        "preprocessing": copy.deepcopy(PREPROCESSING_PROFILES[spec["preprocessing"]]),
        "order": ["RGB_decode", "whole_image_bilinear_resize", "deterministic_preprocessing",
                  "training_only_stochastic_augmentation", "tensor_conversion", "ImageNet_normalisation"],
        "image_size": spec["image_size"], "resize_interpolation": "bilinear",
        "resize_antialias": True, "centre_crop": False,
        "normalisation_mean": [0.485, 0.456, 0.406],
        "normalisation_std": [0.229, 0.224, 0.225],
    }


def require_opencv():
    try:
        return importlib.import_module("cv2")
    except ImportError as error:
        raise ImportError("CLAHE requires optional opencv-python-headless==4.13.0.92; "
                          "unchanged preprocessing is never substituted.") from error


class LightnessCLAHE:
    """Fixed per-image L-only correction; source arrays are never modified."""
    def __init__(self):
        self.cv2 = require_opencv()
        self.clahe = self.cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, image):
        pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        lab = self.cv2.cvtColor(pixels, self.cv2.COLOR_RGB2LAB)
        corrected = lab.copy()
        corrected[..., 0] = self.clahe.apply(lab[..., 0])
        rgb = self.cv2.cvtColor(corrected, self.cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


class ReflectAffine:
    """Affine in original-image units, with reflection outside its full canvas.

    The central output extraction restores the original field after temporary
    padding. It is not a resized/random image crop. Lesion clipping or reflected
    lesion duplication is nevertheless subject to the experiment's quality gate.
    """
    def __init__(self, padding=112):
        self.padding = padding

    @staticmethod
    def parameters(image):
        width, height = image.size
        angle = float(torch.empty(1).uniform_(-45.0, 45.0).item())
        translate = [int(round(float(torch.empty(1).uniform_(-0.05 * width, 0.05 * width).item()))),
                     int(round(float(torch.empty(1).uniform_(-0.05 * height, 0.05 * height).item())))]
        scale = float(torch.empty(1).uniform_(0.9, 1.05).item())
        return angle, translate, scale

    def apply(self, image, *, angle, translate, scale):
        width, height = image.size
        padded = TF.pad(image, self.padding, padding_mode="reflect")
        transformed = TF.affine(padded, angle=angle, translate=translate,
                                scale=scale, shear=[0.0, 0.0],
                                interpolation=InterpolationMode.BILINEAR,
                                fill=0)
        return TF.crop(transformed, self.padding, self.padding, height, width)

    def __call__(self, image):
        angle, translate, scale = self.parameters(image)
        return self.apply(image, angle=angle, translate=translate, scale=scale)


def deterministic_prepare(spec):
    validate_profiles(spec)
    operations = [transforms.Resize((spec["image_size"], spec["image_size"]),
                                    interpolation=InterpolationMode.BILINEAR,
                                    antialias=True)]
    if spec["preprocessing"] == "clahe":
        operations.append(LightnessCLAHE())
    return transforms.Compose(operations)


def stochastic_augmentation(spec):
    validate_profiles(spec)
    # Keep this exact operation order, including RandomRotation defaults, for
    # historical weak outputs. No colour operation is added to weak.
    operations = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    if spec["augmentation"] == "weak":
        operations.append(transforms.RandomRotation(20))
    else:
        operations += [ReflectAffine(112), transforms.ColorJitter(
            brightness=(0.8, 1.2), contrast=(0.8, 1.2),
            saturation=(0.9, 1.1), hue=0)]
    return transforms.Compose(operations)


def make_preview_transform(spec, training=True):
    operations = [deterministic_prepare(spec)]
    if training:
        operations.append(stochastic_augmentation(spec))
    return transforms.Compose(operations)


def make_transform(spec, training=False):
    return transforms.Compose([make_preview_transform(spec, training),
                               transforms.ToTensor(), transforms.Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])])


__all__ = ["AUGMENTATION_PROFILES", "PREPROCESSING_PROFILES", "profile_identity",
           "validate_profiles", "deterministic_prepare", "stochastic_augmentation",
           "make_preview_transform", "make_transform", "LightnessCLAHE", "ReflectAffine",
           "require_opencv"]

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .utils import RANDOM_STATE

IMAGE_SIZE = 224
# Fixed ImageNet constants for the pretrained backbones; never computed from our data.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

IMAGE_METADATA_FEATURES = ["age_approx", "sex", "anatom_site_general_challenge"]
IMAGE_CATEGORICAL_FEATURES = ["sex", "anatom_site_general_challenge"]


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
class DullRazor:
    """DullRazor-style dark-hair removal: black-hat mask + Telea inpainting.

    This is an approximation, not the original paper's full algorithm. Inspect
    outputs: dark lesion structures can also be masked. Input/output: RGB uint8.
    """

    def __init__(self, kernel_size=17, threshold=10, inpaint_radius=3):
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 3.")
        if not 0 <= threshold <= 255 or inpaint_radius <= 0:
            raise ValueError("Invalid hair threshold or inpaint radius.")
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.inpaint_radius = inpaint_radius

    def __call__(self, image):
        import cv2
        rgb = np.asarray(image)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Hair removal expects RGB uint8 pixels.")
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (self.kernel_size, self.kernel_size))
        hairs = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, kernel)
        mask = np.uint8(hairs > self.threshold) * 255
        return cv2.inpaint(rgb, mask, self.inpaint_radius, cv2.INPAINT_TELEA)


class ImageTransform:
    """Picklable PIL-to-tensor adapter for the existing Dataset and Windows workers."""

    def __init__(self, pipeline, hair_removal=None):
        self.pipeline = pipeline
        self.hair_removal = hair_removal

    def __call__(self, image):
        rgb = np.asarray(image)
        if self.hair_removal is not None:
            rgb = self.hair_removal(rgb)
        return self.pipeline(image=rgb)["image"]


def image_transforms(image_size: int | None = None, augmentation="basic",
                     hair_removal=False, seed=RANDOM_STATE, *, model_name=None,
                     mean=None, std=None, rotation=0):
    """PIL-callable train/eval transforms; basic, strong, or no augmentation.

    Hair removal is deterministic and runs before resize on every split. Strong
    augmentation runs after resizing to image_size to reduce CPU work, using
    Albumentations 2.x names. Pixel-based effects act at the resized resolution.
    """
    if model_name is not None:
        from .model_registry import get_model_spec
        spec = get_model_spec(model_name)
        image_size = spec.input_size if image_size is None else image_size
        model_mean, model_std = spec.normalization
    else:
        image_size = IMAGE_SIZE if image_size is None else image_size
        model_mean, model_std = IMAGE_MEAN, IMAGE_STD
    mean = model_mean if mean is None else mean
    std = model_std if std is None else std
    if rotation < 0 or rotation > 30:
        raise ValueError("rotation must be between 0 and 30 degrees.")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 32:
        raise ValueError("image_size must be an integer >= 32.")
    if augmentation not in {"none", "basic", "strong"}:
        raise ValueError("augmentation must be none, basic, or strong.")
    # Preserve the dependency-light original interface for basic experiments.
    if augmentation != "strong" and not hair_removal:
        from torchvision import transforms
        resize = transforms.Resize((image_size, image_size), antialias=True)
        finish = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        flips = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        if rotation:
            flips.append(transforms.RandomRotation(rotation))
        return (transforms.Compose([resize] + (flips if augmentation == "basic" else []) + finish),
                transforms.Compose([resize] + finish))

    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    augment = []
    if augmentation == "basic":
        augment = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    elif augmentation == "strong":
        augment = [
            A.Transpose(p=0.5), A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=0.75),
            A.OneOf([A.MotionBlur(blur_limit=5), A.MedianBlur(blur_limit=5),
                     A.GaussianBlur(blur_limit=5),
                     A.GaussNoise(std_range=(np.sqrt(5)/255, np.sqrt(30)/255))], p=0.7),
            A.OneOf([A.OpticalDistortion(distort_limit=1.0),
                     A.GridDistortion(num_steps=5, distort_limit=1.0),
                     A.ElasticTransform(alpha=3)], p=0.7),
            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                 val_shift_limit=10, p=0.5),
            A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1),
                     rotate=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=0.85),
        ]
    if rotation and augmentation == "basic":
        augment.append(A.Rotate(limit=rotation, p=0.5))
    finish = [A.Normalize(mean=mean, std=std), ToTensorV2()]
    dropout = [A.CoarseDropout(num_holes_range=(1, 1),
                              hole_height_range=(1, max(1, int(image_size * 0.375))),
                              hole_width_range=(1, max(1, int(image_size * 0.375))),
                              fill=0, p=0.7)] if augmentation == "strong" else []
    train = A.Compose([A.Resize(image_size, image_size)] + augment + dropout + finish, seed=seed)
    evaluation = A.Compose([A.Resize(image_size, image_size)] + finish, seed=seed)
    remover = DullRazor() if hair_removal else None
    return ImageTransform(train, remover), ImageTransform(evaluation, remover)


def image_for_display(tensor):
    """Undo normalisation so a (C, H, W) tensor can be shown with ``imshow``."""
    mean = tensor.new_tensor(IMAGE_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGE_STD)[:, None, None]
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

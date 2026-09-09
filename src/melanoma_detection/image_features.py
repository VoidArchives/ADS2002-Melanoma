import numpy as np
from PIL import Image

from pathlib import Path

import cv2


from .image_utils import resolve_image_path


def extract_image_features(image_path: str | Path, project_root: str | Path, analysis_size=(256,256), blur_sigma: float = 1.0,) -> dict[str,float]:
    """
    Extracts global features: colour, lighting, texture from a single image
    """

    path = resolve_image_path(image_path, project_root)

    with Image.open(path) as image:
        image = image.convert("RGB")    # not ncessary but just standard

        # image.thumbnail(analysis_size)
        image = image.resize(analysis_size, Image.Resampling.LANCZOS)   # fixed size is more comparable

        rgb=(np.asarray(image,dtype=np.float32)/ 255.0)

    #color channels
    red = rgb[:,:,0]
    green = rgb[:,:,1]
    blue = rgb[:,:,2]

    # greyscale brightness
    luminance = ( 0.299 * red + 0.587 * green + 0.114 * blue)

    # Saturation
    maximum_channel = rgb.max(axis=2)
    minimum_channel = rgb.min(axis=2)

    saturation = np.divide(maximum_channel - minimum_channel, maximum_channel, out=np.zeros_like(maximum_channel), where = maximum_channel >0)

    # greyscale entropy
    histogram, _ = np.histogram(luminance, bins=256,range=(0,1))


    probabilities = histogram[histogram >0] / histogram.sum()


    entropy = -np.sum(probabilities * np.log2(probabilities))

    # 1st derivative edge strength
    gradient_y, gradient_x = np.gradient(luminance)


    gradient_strength = np.hypot(gradient_x,gradient_y)
    # np.sqrt(gradient_x**2 + gradient_y**2)

    grayscale = np.clip(luminance * 255, 0,255).astype(np.uint8)


    if blur_sigma > 0:
        grayscale = cv2.GaussianBlur(grayscale,ksize=(0,0), sigmaX=blur_sigma,sigmaY=blur_sigma)


    laplacian = cv2.Laplacian(grayscale, ddepth=cv2.CV_32F,ksize=3,)

    absolute_laplacian = np.abs(laplacian)


    return {
        # colours: mean, standard deviation
        'red_mean': float(red.mean()),
        'green_mean': float(green.mean()),
        'blue_mean': float(blue.mean()),

        'red_std': float(red.std()),
        'green_std': float(green.std()),
        'blue_std': float(blue.std()),

        # lighting, colour intensity

        'brightness_mean': float(luminance.mean()),
        'contrast': float(luminance.std()),
        'saturation_mean': float(saturation.mean()),
        'saturation_std': float(saturation.std()),

        'brightness_q05': float(
            np.percentile(luminance, 5)
        ),
        'brightness_q95': float(
            np.percentile(luminance, 95)
        ),
        # image complexity

        'grayscale_entropy': float(entropy),
        'gradient_mean': float(gradient_strength.mean()),
        'gradient_q95': float(np.percentile(gradient_strength,95)),


        #sharpness, fine detail

        'laplacian_variance': float(laplacian.var()),
        'laplacian_mean_abs': float(absolute_laplacian.mean()),
        'laplacian_q95_abs': float(np.percentile(absolute_laplacian,95)),


        # dark/bright regions

        'dark_prop': float((luminance <0.1).mean()),

        'bright_prop': float((luminance >0.9).mean()),

    }



# def hair_removal()




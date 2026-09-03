import math as m

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from melanoma_detection.image_utils import resolve_image_path

def image_grid(rows, tiles, columns=4):
    """
    Displays indexed JPEG images in a labelled grid
    """

    rows = rows.reset_index(drop=True)


    number_images = len(rows)


    if number_images == 0:
        raise ValueError('no images supplied')



    number_rows = m.ceil(number_images/columns)


    fig,axes = plt.subplots(number_rows, columns, figsize=(4*columns, 4*number_rows))


    for axis, row in zip(axis, rows.itertuples()):
        image_path = resolve_image_path(row.jpeg_path, project_root=project_root) # not sure whats happened here...

        with Image



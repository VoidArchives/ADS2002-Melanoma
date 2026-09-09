import math as m

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from melanoma_detection.image_utils import resolve_image_path

def image_grid(rows, project_root,title, columns=4):
    """
    Displays indexed JPEG images in a labelled grid
    """

    rows = rows.reset_index(drop=True)


    number_images = len(rows)


    if number_images == 0:
        raise ValueError('no images supplied')



    number_rows = m.ceil(number_images/columns)


    fig,axes = plt.subplots(number_rows, columns, figsize=(4*columns, 4*number_rows))
    axes = np.atleast_1d(axes).ravel()


    for axis, row in zip(axes, rows.itertuples()):
        image_path = resolve_image_path(row.jpeg_path, project_root) # not sure whats happened here...

        with Image.open(image_path) as image:
            image_array = np.asarray(image.convert('RGB'))

        diagnosis = 'Melanoma' if row.target == 1 else 'Benign'

        axis.imshow(image_array)
        axis.set_title(
            f'{row.image_name}\n{diagnosis}',
            fontsize=10
        )

        axis.axis('off')


    for axis in axes[number_images:]:
        axis.axis('off')


    fig.suptitle(title,fontsize=16.5)

    plt.tight_layout(rect=[0,0,1,0.96])


    return fig


def feature_extreme_grids(
    rows,
    feature,
    project_root,
    description=None,
    number_images=8,
    columns=4,
):
    """
    Create image grids containing the lowest and highest values
    of a selected image feature.
    """

    if feature not in rows.columns:
        raise KeyError(f"Feature not found: {feature}")

    valid_rows = rows.dropna(subset=[feature])

    if valid_rows.empty:
        raise ValueError(
            f"No valid values available for {feature}"
        )

    number_images = min(
        number_images,
        len(valid_rows),
    )

    label = (
        description
        if description is not None
        else feature.replace("_", " ")
    )

    lowest = valid_rows.nsmallest(
        number_images,
        feature,
    )

    highest = valid_rows.nlargest(
        number_images,
        feature,
    )

    lowest_figure = image_grid(
        rows=lowest,
        project_root=project_root,
        title=f"Lowest {label}",
        columns=columns,
    )

    highest_figure = image_grid(
        rows=highest,
        project_root=project_root,
        title=f"Highest {label}",
        columns=columns,
    )

    return lowest_figure, highest_figure
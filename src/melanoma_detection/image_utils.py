from pathlib import Path


def resolve_image_path(image_path, project_root):
    """
    Converts a stored relative image path into a usable local path
    """
    path = Path(image_path)

    if path.is_absolute():
        return path

    return Path(project_root) / path




from pathlib import Path

import kagglehub
import pandas as pd


COMPETITION = "siim-isic-melanoma-classification"


def get_competition_file(filename):
    """Return the local KaggleHub path for a competition file."""
    
    path = kagglehub.competition_download(
        COMPETITION,
        path=filename
    )

    return Path(path)


def load_train_metadata():
    """Load the training metadata into a pandas DataFrame."""

    path = get_competition_file("train.csv")

    return pd.read_csv(
        path,
        compression="zip"
    )

def load_test_metadata():
    """Load the testing metadata into a pandas DataFrame"""

    path = get_competition_file("test.csv")

    return pd.read_csv(
        path)
    
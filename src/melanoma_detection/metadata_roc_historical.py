"""Archive-bound historical reads and separate v2 table destinations."""
from pathlib import Path


def load_clean_scores(root):
    from .dedup_workflow import load_clean_scores as read
    return read(root)


def latest_evaluation(root):
    from .dedup_workflow import latest_evaluation as read
    return read(root)


def load_matched_metadata(*args, **kwargs):
    from .image_nested_reporting import load_matched_metadata as read
    return read(*args, **kwargs)


def save_metadata_table(root, stage, name, table, *, condition="dedup_v1", run_id="roc_auc_grid36_v1"):
    from .metadata_tuning_v2 import condition_directory
    if Path(stage).name != stage or Path(name).name != name:
        raise ValueError("A report stage and table filename must be simple names.")
    directory = condition_directory(root, condition, run_id, experiment_version=2) / "report" / stage / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    filename = name if name.endswith(".csv") else name + ".csv"
    table.to_csv(directory / filename, index=False)
    return table

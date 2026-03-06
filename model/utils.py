import joblib, sklearn, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import default_pipelines_path, pipeline_prefix, version_prefix, version_width, default_cols, dropped_cols

###### pipeline Saveing I/O ######
def save_pipeline(
    pipeline,
    meta: dict | None = None,
    *,
    target_version: int | None = None,
    compress: int = 3,
    cwd: str | Path | None = None,
) -> tuple[Path, int]:
    """
    Saves a joblib bundle to <default_pipelines_path>_v### and increments/renames the CWD version file.

    Returns (saved_pipeline_path, new_version_int).
    """
    
    vf = _find_version_file()
    target_ver = _parse_version(vf) + 1 if target_version is None else target_version
    out_path = _versioned_pipeline_path(cwd, target_ver)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": pipeline,
        "meta": {
            # TODO: Follow up and make sure we're covering all the important libraries
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            **(meta or {}),
        },
    }

    # Save first; only bump version file if save succeeds
    joblib.dump(bundle, out_path, compress=compress)

    # Preserve the zero-padding style of the version file if it had any
    new_vf_name = f"{version_prefix}{str(target_ver).zfill(version_width)}"
    vf.rename(vf.with_name(new_vf_name))

    return out_path, target_ver

def load_pipeline(pipeline_version: int, cwd: str | Path | None = None) -> Pipeline :
    pipeline_path = _versioned_pipeline_path(cwd, pipeline_version)
    bundle = joblib.load(pipeline_path)
    
    # Confirm that current system matches expected meta versions (sklearn, numpy) and warn if not
    meta = bundle.get("meta", {})
    
    sklearn_version = meta.get("sklearn_version")
    if sklearn_version and sklearn_version != sklearn.__version__:
        warnings.warn(f"WARNING: Loaded pipeline was trained with sklearn version {sklearn_version}, but current version is {sklearn.__version__}. This may cause compatibility issues.")
        
    numpy_version = meta.get("numpy_version")
    if numpy_version and numpy_version != np.__version__:
        warnings.warn(f"WARNING: Loaded pipeline was trained with numpy version {numpy_version}, but current version is {np.__version__}. This may cause compatibility issues.")
    
    return bundle["pipeline"]

def load_latest_pipeline(cwd: str | Path | None = None) -> Pipeline:
    version_int = _get_latest_pipeline_version(cwd)
    return load_pipeline(version_int, cwd)



###### Data Loading and Formatting ######
def read_dataset_from_csv(filePath: str) -> pd.DataFrame:
    data = pd.read_csv(filePath, sep="\t", header=None)
    return format_csv_data(data)

def format_csv_data(data: pd.DataFrame) -> pd.DataFrame:
    pd.set_option('display.max_columns', None)
    data = data.astype(float)
    data = data.rename(columns=default_cols)
    data = data.drop(dropped_cols, axis=1)
    return data

def get_split_data(data: pd.DataFrame, label_col: str = "Marker Channel", test_size: float = 0.2, random_state: int = 42) -> list:
    x = data.drop(columns=[label_col])
    # TODO: Determine how the labels should actually be determined and replace this temp debug function
    y = debug_extend_labels(data)
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def debug_print_dataset_details(dataset: pd.DataFrame) -> None:
    lowest = dataset.iloc[0, 22]
    # TODO: Investigate and fix type complaint about the following line instead of just using a type ignore comment. 
    dataset['Timestamp'] = dataset['Timestamp'] - lowest # type: ignore
    dataset['Label'] = (round(dataset['Timestamp'], 0) % 10) >= 5
    print(dataset)

# Temporary debug function only created to allow me to train a model with more than 3 labels per class
def debug_extend_labels(data: pd.DataFrame) -> pd.Series:
    column_name = "Marker Channel"
    new_column = data[column_name].copy()
    
    current_overwrite = 2
    for i in range(len(data)):
        current_value = data[column_name].iloc[i]
        
        # Check if the current value is non-zero
        if current_value == 0:
            # Overwrite the current row with the new value
            new_column.iloc[i] = current_overwrite
        else:
            current_overwrite = current_value
            
    # Assign the new column back to the DataFrame
    #data[column_name] = new_column
    return new_column


###### File Helper Functions ######
def _find_version_file() -> Path:
    pipeline_folder = Path(__file__).parent
    
    matches = sorted(
        p for p in pipeline_folder.iterdir()
        if p.is_file() and p.name.startswith(version_prefix)
    )

    if len(matches) == 0:
        # If none exists, create version=0 so the first save becomes v001 (or v000→v001 etc).
        vf = pipeline_folder / f"{version_prefix}0"
        vf.touch(exist_ok=True)
        return vf

    if len(matches) > 1:
        raise RuntimeError(
            f"Expected exactly one '{version_prefix}*' file in {pipeline_folder}, found: {[p.name for p in matches]}"
        )

    return matches[0]

def _parse_version(vf: Path) -> int:
    """
    Returns version_int.
    """
    raw = vf.name[len(version_prefix):]
    if not raw.isdigit():
        raise ValueError(f"Version file name must look like '{version_prefix}=<digits>', got: {vf.name}")
    return int(raw)

def _get_latest_pipeline_version(cwd: str | Path | None = None) -> int:
    """
    Returns the latest pipeline version int by parsing the version file in CWD.
    """
    vf = _find_version_file()
    assert vf, f"No version file found in {cwd or Path.cwd()} - expected a file named like '{version_prefix}=<digits>'"
    return _parse_version(vf)

def _versioned_pipeline_path(cwd: str | Path | None = None, version: int = 1) -> Path:
    """
    Returns a new path for a pipeline file with the given version int, in the default pipelines folder.
    """
    cwd_path = Path(cwd) if cwd is not None else Path.cwd()
    return cwd_path / default_pipelines_path / f"{pipeline_prefix}{version:0{version_width}d}.joblib"
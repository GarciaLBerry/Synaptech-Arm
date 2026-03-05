from __future__ import annotations

from pathlib import Path
from typing import List
import joblib, sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import default_cols

from pathlib import Path

###### Model Saveing I/O ######
def save_model(
    pipeline,
    meta: dict | None = None,
    *,
    target_version: int | None = None,
    compress: int = 3,
    cwd: str | Path | None = None,
) -> tuple[Path, int]:
    """
    Saves a joblib bundle to <default_weights_path>_v### and increments/renames the CWD version file.

    Returns (saved_model_path, new_version_int).
    """
    vf = _find_version_file(cwd)
    target_ver = _parse_version(vf) + 1 if target_version is None else target_version
    out_path = _versioned_path(Path(default_weights_path), target_ver)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": pipeline,
        "meta": {
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

def load_model(model_version: int):
    model_path = _versioned_path(Path(default_weights_path), model_version)
    bundle = joblib.load(model_path)
    return bundle["model"], bundle.get("meta", {})

def load_latest_model(cwd: str | Path | None = None):
    version_int = _get_latest_model_version(cwd)
    return load_model(version_int)



###### Data Loading and Formatting ######
def read_dataset_from_csv(filePath: str) -> pd.DataFrame:
    df = pd.read_csv(filePath, names=['Unsorted'])
    return df

def format_data(data: pd.DataFrame) -> pd.DataFrame:
    pd.set_option('display.max_columns', None)
    data = data['Unsorted'].str.split('\t', expand=True)
    data = data.astype(float)
    data = data.rename(columns = default_cols)
    
    return data

def get_split_data(data: pd.DataFrame, label_col: str = "Label", test_size: float = 0.2, random_state: int = 42) -> List[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x = data.drop(columns=[label_col])
    y = data[label_col]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def debug_print_dataset_details(dataset: pd.DataFrame) -> None:
    lowest = dataset.iloc[0, 22]
    # TODO: Investigate and fix type complaint about the following line instead of just using a type ignore comment. 
    dataset['Timestamp'] = dataset['Timestamp'] - lowest # type: ignore
    dataset['Label'] = (round(dataset['Timestamp'], 0) % 10) >= 5
    print(dataset)


###### File Helper Functions ######
default_weights_path: str = "./inference/weights"
version_prefix: str = "version="
version_width: int = 3

def _find_version_file(cwd: str | Path | None = None) -> Path:
    cwd_path = Path(cwd) if cwd is not None else Path.cwd()
    
    matches = sorted(
        p for p in cwd_path.iterdir()
        if p.is_file() and p.name.startswith(version_prefix)
    )

    if len(matches) == 0:
        # If none exists, create version=0 so the first save becomes v001 (or v000→v001 etc).
        vf = cwd_path / f"{version_prefix}0"
        vf.touch(exist_ok=True)
        return vf

    if len(matches) > 1:
        raise RuntimeError(
            f"Expected exactly one '{version_prefix}*' file in {cwd_path}, found: {[p.name for p in matches]}"
        )

    return matches[0]

def _parse_version(vf: Path) -> int:
    """
    Returns version_int.
    digit_count helps preserve zero-padding style when renaming the version file.
    """
    raw = vf.name[len(version_prefix):]
    if not raw.isdigit():
        raise ValueError(f"Version file name must look like '{version_prefix}=<digits>', got: {vf.name}")
    return int(raw)

def _get_latest_model_version(cwd: str | Path | None = None) -> int:
    """
    Returns the latest model version int by parsing the version file in CWD.
    """
    vf = _find_version_file(cwd)
    assert vf, f"No version file found in {cwd or Path.cwd()} - expected a file named like '{version_prefix}=<digits>'"
    return _parse_version(vf)

def _versioned_path(base_path: Path, version: int) -> Path:
    """
    Returns a new path with the same base_path but with new version number inserted before a potential suffix, e.g. "model.joblib" → "model_v001.joblib".
    """
    return base_path.with_name(f"{base_path.stem}_v{version:0{version_width}d}{base_path.suffix}")
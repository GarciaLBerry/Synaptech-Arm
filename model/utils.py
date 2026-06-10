import joblib, sklearn, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import (
    default_pipelines_path,
    pipeline_prefix,
    version_prefix,
    version_width,
    default_cols,
    dropped_cols,
    label_col,
    PACKET_SIZE,
    PACKET_STRIDE,
    SAMPLE_RATE,
    DROP_TRANSITIONS,
    LABEL_TRANSITIONS
)

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
def get_data(
    data_root: str,
    test_size: float = 0.5,
    random_state: int = 42,
    n_groups: int = 12,
) -> list[np.ndarray]:
    """
    Loads recording-level assignments from ``train`` and ``test`` directories.
    CSVs left directly under data_root are split at state transitions or
    10-second boundaries, packetized, and assigned to improve the overall
    train/test class balance.

    ``n_groups`` is retained for compatibility with older callers.
    """
    del n_groups
    root = Path(data_root)
    split_parts = {"train": [], "test": []}
    next_group = 0

    for split_name in ("train", "test"):
        for dataset_path in _csv_paths(root / split_name):
            data = prepare_data_from_file(dataset_path)
            x, y = packetize_prepared_data(
                data,
                drop_transitions=DROP_TRANSITIONS and split_name == "train",
            )
            split_parts[split_name].append((x, y, next_group))
            next_group += 1

    unassigned_parts = []
    max_chunk_rows = SAMPLE_RATE * 10
    for dataset_path in _csv_paths(root):
        data = prepare_data_from_file(dataset_path)
        for chunk in _split_at_transitions_and_max_rows(data, max_chunk_rows):
            if len(chunk) < PACKET_SIZE:
                continue
            x, y = packetize_prepared_data(chunk, drop_transitions=False)
            if len(y) == 0:
                continue
            unassigned_parts.append((x, y, next_group))
            next_group += 1

    assignments = _assign_packetized_chunks(
        split_parts,
        unassigned_parts,
        test_size,
        random_state,
    )
    for part, split_name in zip(unassigned_parts, assignments):
        split_parts[split_name].append(part)

    x_train_parts, y_train_parts, groups_train_parts = _unpack_split_parts(
        split_parts["train"]
    )
    x_test_parts, y_test_parts, groups_test_parts = _unpack_split_parts(
        split_parts["test"]
    )

    return [
        np.concatenate(x_train_parts, axis=0),
        np.concatenate(x_test_parts, axis=0),
        np.concatenate(y_train_parts, axis=0),
        np.concatenate(y_test_parts, axis=0),
        np.concatenate(groups_train_parts, axis=0),
        np.concatenate(groups_test_parts, axis=0),
    ]

def _csv_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )

def _split_at_transitions_and_max_rows(
    data: pd.DataFrame,
    max_rows: int,
) -> list[pd.DataFrame]:
    labels = data[label_col].to_numpy(copy=False)
    transition_indices = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    state_boundaries = np.r_[0, transition_indices, len(data)]
    chunks = []

    for state_start, state_end in zip(state_boundaries[:-1], state_boundaries[1:]):
        for chunk_start in range(state_start, state_end, max_rows):
            chunk_end = min(chunk_start + max_rows, state_end)
            chunks.append(data.iloc[chunk_start:chunk_end].reset_index(drop=True))
    return chunks

def _assign_packetized_chunks(
    split_parts: dict[str, list[tuple[np.ndarray, np.ndarray, int]]],
    chunks: list[tuple[np.ndarray, np.ndarray, int]],
    test_size: float,
    random_state: int,
) -> list[str]:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if not chunks:
        return []

    labels = sorted({
        label
        for parts in [*split_parts.values(), chunks]
        for _, y, _ in parts
        for label in np.unique(y)
    })
    label_to_index = {label: index for index, label in enumerate(labels)}

    def counts(parts):
        result = np.zeros(len(labels), dtype=int)
        for _, y, _ in parts:
            for label, count in zip(*np.unique(y, return_counts=True)):
                result[label_to_index[label]] += count
        return result

    train_counts = counts(split_parts["train"])
    test_counts = counts(split_parts["test"])
    chunk_counts = [counts([chunk]) for chunk in chunks]
    total_counts = train_counts + test_counts + np.sum(chunk_counts, axis=0)
    target_test_counts = total_counts * test_size
    target_test_total = total_counts.sum() * test_size

    def score(candidate_test_counts):
        class_error = np.mean(
            ((candidate_test_counts - target_test_counts) / np.maximum(total_counts, 1))
            ** 2
        )
        total_error = (
            (candidate_test_counts.sum() - target_test_total)
            / max(total_counts.sum(), 1)
        ) ** 2
        return class_error + total_error

    rng = np.random.default_rng(random_state)
    order = list(rng.permutation(len(chunks)))
    order.sort(key=lambda index: len(chunks[index][1]), reverse=True)
    assignments = ["train"] * len(chunks)
    current_test_counts = test_counts.copy()
    for index in order:
        if score(current_test_counts + chunk_counts[index]) < score(current_test_counts):
            assignments[index] = "test"
            current_test_counts += chunk_counts[index]
    return assignments

def _unpack_split_parts(parts):
    return (
        [x for x, _, _ in parts],
        [y for _, y, _ in parts],
        [np.full(len(y), group) for _, y, group in parts],
    )

def prepare_data_from_file(dataset_path: str | Path) -> pd.DataFrame:
    data = read_dataset_from_csv(dataset_path)
    data = format_csv_data(data)
    data = drop_leading_bad_rows(data)
    data = repair_zero_rows(data)
    data[label_col] = extend_labels(data)
    return data

def packetize_prepared_data(
    data: pd.DataFrame,
    drop_transitions: bool = DROP_TRANSITIONS,
) -> tuple[np.ndarray, np.ndarray]:
    x = data.drop(columns=[label_col])
    y = data[label_col]
    return packetize_data(x, y, drop_transitions=drop_transitions)

def read_dataset_from_csv(filePath: str | Path) -> pd.DataFrame:
    return pd.read_csv(filePath, sep="\t", header=None)

def format_csv_data(data: pd.DataFrame) -> pd.DataFrame:
    pd.set_option('display.max_columns', None)
    data = data.astype(float)
    data = data.rename(columns=default_cols)
    
    # TODO: Make this more robust and not fully fail if any of the expected columns are missing - Likely just for loop dropping one column at a time.
    data = data.drop(dropped_cols, axis=1)
    return data

def extend_labels(data: pd.DataFrame) -> pd.Series:
    new_column = data[label_col].copy()
    
    current_overwrite = 2
    for i in range(len(data)):
        current_value = data[label_col].iloc[i]
        
        # Check if the current value is non-zero
        if current_value == 0:
            # Overwrite the current row with the new value
            new_column.iloc[i] = current_overwrite
        else:
            current_overwrite = current_value
            
    return new_column

def packetize_data(
    x: pd.DataFrame,
    y: pd.Series,
    drop_transitions: bool = DROP_TRANSITIONS,
) -> tuple[np.ndarray, np.ndarray]:
    X = x.to_numpy(copy=False)
    Y = y.to_numpy(copy=False)
    
    if len(X) != len(Y):
        raise ValueError(f"x and y length mismatch: len(x)={len(X)} len(y)={len(Y)}")

    n_rows, n_channels = X.shape
    if n_rows < PACKET_SIZE:
        raise ValueError(f"Not enough rows ({n_rows}) for packet_size={PACKET_SIZE}")

    x_packets = []
    y_packets = []

    for start in range(0, n_rows - PACKET_SIZE + 1, PACKET_STRIDE):
        end = start + PACKET_SIZE

        x_seg = X[start:end, :]
        y_seg = Y[start:end]

        # Experiment to see if keeping things more reflective of the movement vs resting helps or hinders.
        packet_label = y_seg[-1] if y_seg[-1] != 2 else y_seg[0]
        
        if not np.all(y_seg == packet_label):
            if drop_transitions:
                continue
            elif LABEL_TRANSITIONS:
                y_packets.append(4)

        if not LABEL_TRANSITIONS:
            y_packets.append(packet_label)

        # Convert from (packet_size, channels) to (channels, packet_size)
        x_packets.append(x_seg.T.astype(np.float32, copy=False))

    if not x_packets:
        return (
            np.empty((0, n_channels, PACKET_SIZE), dtype=np.float32),
            np.empty((0,), dtype=Y.dtype),
        )

    return np.asarray(x_packets), np.asarray(y_packets)


def drop_leading_bad_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows [0..k] where k is the first row index within the first 10 rows
    whose first feature column value is zero.
    If no such row exists, returns data unchanged.
    """
    feature_df = data.drop(columns=[label_col], errors="ignore")
    feature_arr = feature_df.to_numpy()

    zero_idxs = np.where(feature_arr[:10, 0] == 0)[0]
    if zero_idxs.size == 0:
        return data
    
    first_zero_idx = int(zero_idxs[0])
    return data.iloc[first_zero_idx + 1 :].reset_index(drop=True)

def repair_zero_rows(data: pd.DataFrame) -> pd.DataFrame:
    repaired = data.copy()

    zero_check_cols = repaired.columns[:5]

    zero_rows = (repaired[zero_check_cols] == 0).all(axis=1)

    repaired.loc[zero_rows, zero_check_cols] = np.nan
    repaired[zero_check_cols] = repaired[zero_check_cols].interpolate(method="linear")

    return repaired

def _allocate_chunks_per_file(row_counts: list[int], n_groups: int) -> list[int]:
    """
    Allocates exactly n_groups contiguous chunks across source files while
    keeping the resulting chunk row counts as similar as possible.
    """
    if len(row_counts) == 0:
        raise ValueError("At least one dataset file is required.")
    if any(row_count <= 0 for row_count in row_counts):
        raise ValueError(f"Dataset files must contain rows, got row counts: {row_counts}")
    if n_groups < len(row_counts):
        raise ValueError(
            f"n_groups={n_groups} must be at least the number of dataset files "
            f"({len(row_counts)}) so every file is represented."
        )
    if n_groups > sum(row_counts):
        raise ValueError(
            f"n_groups={n_groups} cannot exceed the total number of rows "
            f"({sum(row_counts)})."
        )

    target_rows = sum(row_counts) / n_groups
    chunks_per_file = [1] * len(row_counts)

    for _ in range(n_groups - len(row_counts)):
        best_file_index = min(
            range(len(row_counts)),
            key=lambda file_index: _chunk_allocation_error(
                row_counts[file_index],
                chunks_per_file[file_index] + 1,
                target_rows,
            ) - _chunk_allocation_error(
                row_counts[file_index],
                chunks_per_file[file_index],
                target_rows,
            ),
        )
        chunks_per_file[best_file_index] += 1

    return chunks_per_file

def _chunk_allocation_error(row_count: int, chunk_count: int, target_rows: float) -> float:
    chunk_rows = row_count / chunk_count
    return chunk_count * (chunk_rows - target_rows) ** 2

def _split_data_into_chunks(data: pd.DataFrame, chunk_count: int) -> list[pd.DataFrame]:
    row_indices = np.array_split(np.arange(len(data)), chunk_count)
    return [
        data.iloc[indices].reset_index(drop=True)
        for indices in row_indices
    ]

def _count_possible_packets(row_count: int) -> int:
    if row_count < PACKET_SIZE:
        return 0
    return ((row_count - PACKET_SIZE) // PACKET_STRIDE) + 1

def _choose_test_file_indices(packet_counts: list[int], test_size: float, random_state: int) -> list[int]:
    """
    Chooses whole source files for the test split, minimizing packet-count drift
    from the requested test_size without splitting any source file.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    n_files = len(packet_counts)
    if n_files < 2:
        raise ValueError(
            "At least two dataset files are required to split without leaking packets from the same file."
        )

    total_packets = sum(packet_counts)
    target_packets = total_packets * test_size
    rng = np.random.default_rng(random_state)
    shuffled_indices = list(rng.permutation(n_files))

    reachable: dict[int, int] = {0: 0}
    for file_index in shuffled_indices:
        packet_count = packet_counts[file_index]
        for packet_sum, mask in list(reachable.items()):
            new_sum = packet_sum + packet_count
            if new_sum not in reachable:
                reachable[new_sum] = mask | (1 << file_index)

    candidates = [
        (packet_sum, mask)
        for packet_sum, mask in reachable.items()
        if packet_sum not in (0, total_packets)
    ]
    if len(candidates) == 0:
        raise ValueError("Could not build a non-empty train/test split from the dataset files.")

    _, best_mask = min(
        candidates,
        key=lambda candidate: (
            abs(candidate[0] - target_packets),
            abs(candidate[0] / total_packets - test_size),
        ),
    )

    return [
        file_index
        for file_index in range(n_files)
        if best_mask & (1 << file_index)
    ]

def debug_print_dataset_details(dataset: pd.DataFrame) -> None:
    lowest = dataset.iloc[0, 22]
    # TODO: Investigate and fix type complaint about the following line instead of just using a type ignore comment. 
    dataset['Timestamp'] = dataset['Timestamp'] - lowest # type: ignore
    dataset['Label'] = (round(dataset['Timestamp'], 0) % 10) >= 5
    print(dataset)

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

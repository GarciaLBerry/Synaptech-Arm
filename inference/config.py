from sklearn.linear_model import LogisticRegression
from scipy.stats import  loguniform, uniform, randint

dataset_path = "Evo_Initial_BCI_Data/2026-27-01_Evo_Run04_FiveSets_Gain12.csv"

default_cols = {
    0: "Sample Index",
    1: "EXG Channel 0",
    2: "EXG Channel 1",
    3: "EXG Channel 2",
    4: "EXG Channel 3",
    5: "EXG Channel 4",
    6: "EXG Channel 5",
    7: "EXG Channel 6",
    8: "EXG Channel 7",
    9: "Accel Channel 0",
    10: "Accel Channel 1",
    11: "Accel Channel 2",
    12: "Not Used",
    13: "Digital Channel 0 (D11)",
    14: "Digital Channel 1 (D12)",
    15: "Digital Channel 2 (D13)",
    16: "Digital Channel 3 (D17)",
    17: "Not Used",
    18: "Digital Channel 4 (D18)",
    19: "Analog Channel 0",
    20: "Analog Channel 1",
    21: "Analog Channel 2",
    22: "Timestamp",
    23: "Marker Channel",
    24: "Timestamp (Formatted)"
}

param_dist = {
    'model__C': loguniform(1e-4, 1e2),
    'model__l1_ratio': uniform(0, 1),
    'model__max_iter': randint(400, 500)
}

pipeline = [
    ("model", LogisticRegression(solver="saga", tol=1e-3))
]

root_dir = "saved_pipelines"
prefix = "pipeline"
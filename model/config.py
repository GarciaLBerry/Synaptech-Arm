import math
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from scipy.stats import  loguniform, uniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from preprocessing.custom_transformers import WaveletTransformer, ChannelDropper
from preprocessing.dimensionality_transformers import FactorAnalysisTransformer, SensorTiedFactorAnalysisTransformer


#----------- signal globals -----------#
# OpenBCI recording frequency per second
SAMPLE_RATE: int = 250
# Rows in each packet
PACKET_SIZE: int = 200
# Time in between packet starts
PACKET_STRIDE: int = 50
# The # of prior packets to include for contextual information
N_LAGS: int = 0


#----------- main function globals -----------#
# Seconds in between loops
MAIN_LOOP_DELAY: float = (SAMPLE_RATE / PACKET_STRIDE) / 2
# Seconds without recieving signal to automatically exit.
MAIN_LOOP_TIMEOUT: int = int(5 / MAIN_LOOP_DELAY)


data_root: str = str((Path(__file__).parent.parent / "data").resolve())
dataset_path: str = "Evo_Initial_BCI_Data/2026-27-01_Evo_Run04_FiveSets_Gain12.csv"
default_pipelines_path: str = "./model/pipelines"
version_prefix: str = "version="
version_width: int = 3
pipeline_prefix: str = "pipeline_v"

label_col: str = "Marker Channel"

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
    23: "Marker Channel"
}

dropped_cols = [
    "Sample Index",
    "EXG Channel 5",
    "EXG Channel 6",
    "EXG Channel 7",
    "Accel Channel 0",
    "Accel Channel 1",
    "Accel Channel 2",
    "Not Used",
    "Digital Channel 0 (D11)",
    "Digital Channel 1 (D12)",
    "Digital Channel 2 (D13)",
    "Digital Channel 3 (D17)",
    "Not Used",
    "Digital Channel 4 (D18)",
    "Analog Channel 0",
    "Analog Channel 1",
    "Analog Channel 2",
    "Timestamp"
]

core_cols = [
    "EXG Channel 0",
    "EXG Channel 1",
    "EXG Channel 2",
    "EXG Channel 3",
    "EXG Channel 4"
]

WAVELET_MAX_LEVEL: int = max(2, int(math.floor(math.log2(PACKET_SIZE / 7))))
WAVELET_LEVEL: int = WAVELET_MAX_LEVEL
WAVELET_INCLUDE_COEFFICIENTS: bool = False
WAVELET_INCLUDE_APPROXIMATION: bool = False
param_dist = {
    'wave__wavelet': ['db4'],
    'wave__drop_first_detail': [False],
    'wave__level': [WAVELET_LEVEL],
    #'reduce__n_components': [3],
    'reduce__append_latents': [True, False],
    'model__C': loguniform(1.0, 100.0),
    'model__gamma': loguniform(5e-3, 5e1),
    'model__class_weight': ["balanced", None]
}

pipeline: Pipeline = Pipeline([
    #("drop_channels", ChannelDropper(drop_channel=2)),
    ("wave", WaveletTransformer(
        include_coefficients=WAVELET_INCLUDE_COEFFICIENTS,
        include_approximation=WAVELET_INCLUDE_APPROXIMATION,
    )),
    ("scaler", StandardScaler()),
    #("reduce", SensorTiedFactorAnalysisTransformer(n_sensors=5,
    ("reduce", FactorAnalysisTransformer(
       n_components=5,
       n_energy_details=WAVELET_LEVEL,
       max_iter=5000,
       random_state=42)),
    ("model", SVC(
        kernel='rbf',
        tol=1e-3,
        probability=True,
        cache_size=2000,
        random_state=42
    ))
])

if hasattr(pipeline.named_steps, "reduce"):
    param_dist["reduce__tol"] = loguniform(1e-4, 3e-2)

#----------- training globals -----------#
TRAINING_ITER: int = 75
# Whether the transition packets are dropped or simply labeled differently
DROP_TRANSITIONS: bool = False
LABEL_TRANSITIONS: bool = False

LABEL_MAPPING = {
    1: "DOWN",
    2: "REST",
    3: "UP"
}
if LABEL_TRANSITIONS and not DROP_TRANSITIONS:
    LABEL_MAPPING[4] = "TRANSITION"

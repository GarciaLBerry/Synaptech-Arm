from sklearn.linear_model import LogisticRegression
from scipy.stats import  loguniform, uniform, randint

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
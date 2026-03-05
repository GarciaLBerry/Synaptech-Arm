from typing import Optional
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import  loguniform, uniform, randint
from sklearn.linear_model import LogisticRegression

from . import utils as u

default_dataset_path = "Evo_Initial_BCI_Data/2026-27-01_Evo_Run04_FiveSets_Gain12.csv"

# Placeholder for training logic
def train(path: Optional[str] = None):
    path = path if path else default_dataset_path
    temp_data = u.read_dataset_from_csv(path)
    temp_data = u.format_data(temp_data)
    u.debug_print_dataset_details(temp_data)

    param_dist = {
        'C': loguniform(1e-4, 1e2),
        'l1_ratio': uniform(0, 1),
        'max_iter': randint(400, 500)
    }

    model = LogisticRegression(solver="saga", tol=1e-3)

    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    x = temp_data.drop(columns=['target'])
    y = temp_data['target']
    tuner.fit(x, y)

    print(f"Best Accuracy: {tuner.best_score_: .4f}")
    print(f"Best Parameters: {tuner.best_params_}")

    return tuner.best_estimator_
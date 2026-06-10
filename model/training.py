from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

import numpy as np
from .config import pipeline, param_dist, data_root, TRAINING_ITER
from .inference import predict
from .evaluation import print_train_results, save_cv_training_report
from . import utils

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def train(data_dir: str | None = None):
    if data_dir is None:
        data_dir = data_root
    
    print("Parsing and formatting training data now...")
    x_train, x_test, y_train, y_test, groups_train, groups_test = utils.get_data(data_dir)
    n_train_groups = len(np.unique(groups_train))
    if n_train_groups < 2:
        raise ValueError("Training requires at least two groups for grouped cross-validation.")
    n_cv_splits = min(5, n_train_groups)

    tuner = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "neg_log_loss": "neg_log_loss"
        },
        refit="balanced_accuracy",
        n_iter=TRAINING_ITER,
        cv=StratifiedGroupKFold(
            n_splits=n_cv_splits,
            shuffle=True,
            random_state=42
        ),
        verbose=1,
        random_state=42,
        n_jobs=-1,
        error_score="raise",
        return_train_score=True
    )
    
    print("Starting training data now...")
    tuner.fit(x_train, y_train, groups=groups_train)
    
    print(f"Best Parameters: {tuner.best_params_}")

    predictor = tuner.best_estimator_
    path, version = utils.save_pipeline(predictor, meta={"data_root": data_dir})
    print_train_results(x_test, y_test, tuner, version)
    save_cv_training_report(tuner, version)
    print(f"Model saved at {path} with version {version}")
    
    predict(x_test, y_test)

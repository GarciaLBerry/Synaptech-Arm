from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix
)
import numpy as np
import pandas as pd

from .config import pipeline, param_dist, data_root
from .inference import predict
from . import utils



def train(data_dir: str | None = None):
    if data_dir is None:
        data_dir = data_root
    
    x_train, x_test, y_train, y_test = utils.get_data(data_dir)

    tuner = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "neg_log_loss": "neg_log_loss",
        },
        refit="balanced_accuracy",
        n_iter=50,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    tuner.fit(x_train, y_train)
    
    print_train_results(x_test, y_test, tuner)
    print(f"Best Parameters: {tuner.best_params_}")

    predictor = tuner.best_estimator_
    path, version = utils.save_pipeline(predictor, meta={"data_root": data_dir})
    print(f"Model saved at {path} with version {version}")
    
    predict(x_test, y_test)
    
    # TODO: Develop better printing of training results
    
def print_train_results(x_test, y_test, tuner):
    labels = [1, 2, 3]
    
    y_pred = tuner.best_estimator_.predict(x_test)
    y_proba = tuner.best_estimator_.predict_proba(x_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    test_log_loss = log_loss(y_test, y_proba, labels=labels)
    true_class_prob = np.exp(-test_log_loss)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Log loss: {test_log_loss:.4f}")
    print(f"True-class assigned probability: {true_class_prob:.4f}")
    print("Confusion matrix:")
    print(cm_df)
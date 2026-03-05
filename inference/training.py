from typing import Optional
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import  loguniform, uniform, randint
from sklearn.linear_model import LogisticRegression
from config import pipeline, param_dist

def train(x_train, y_train) -> LogisticRegression:
    tuner = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    tuner.fit(x_train, y_train)

    print(f"Best Accuracy: {tuner.best_score_: .4f}")
    print(f"Best Parameters: {tuner.best_params_}")

    return tuner.best_estimator_
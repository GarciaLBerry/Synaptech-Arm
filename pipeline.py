import joblib
from sklearn.model_selection import RandomizedSearchCV
from config import param_dist, pipeline
from dataset import x_train, y_train
from pipeline_save_helper import next_pipeline_path

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

joblib.dump(tuner.best_estimator_, next_pipeline_path())
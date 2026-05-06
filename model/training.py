from sklearn.model_selection import RandomizedSearchCV

from .config import pipeline, param_dist, dataset_path
from .inference import predict
from . import utils



def train(dataset_path = ""):
    assert dataset_path, "Dataset path must be provided for training."
    
    x_train, x_test, y_train, y_test = utils.get_data(dataset_path)

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

    predictor = tuner.best_estimator_
    path, version = utils.save_pipeline(predictor, meta={"dataset_path": dataset_path})
    print(f"Model saved at {path} with version {version}")
    
    predict(x_test, y_test)
    
    # TODO: Develop better printing of training results
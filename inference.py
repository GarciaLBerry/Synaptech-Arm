import joblib
from dataset import x_test, y_test

from pipeline_save_helper import recent_pipeline_path

loaded_pipeline = joblib.dump(recent_pipeline_path())
predictions = loaded_pipeline.predict(x_test, y_test)
print(predictions)
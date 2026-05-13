from . import config
from . import utils
import pandas as pd
import numpy as np

verbose = False

def predict(x_test, y_test = None, evaluate = False):
    model = utils.load_latest_pipeline()
    predictions = model.predict(x_test)
    
    if evaluate and (y_test is None):
        _, _, _, y_test = utils.get_data(config.dataset_path)
        
    if verbose:
        print(f"Predictions: {predictions[0]}")
    
    # TODO: Make sure this actually return something useful here
    return predictions[0]

from . import config
from . import utils
import pandas as pd
import numpy as np

verbose = False

def predict(x_test, y_test = None, evaluate = False):
    model = utils.load_latest_pipeline()
    predictions = model.predict(x_test)
    
    if evaluate and (y_test is None):
        data = utils.read_dataset_from_csv(config.dataset_path)
        #utils.debug_print_dataset_details(data)
        _, _, _, y_test = utils.get_split_data(data)
        
    if verbose:
        print(f"Predictions: {predictions[0]}")
    
    # TODO: Make sure this actually return something useful here
    return predictions[0]
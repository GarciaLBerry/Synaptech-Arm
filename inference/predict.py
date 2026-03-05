from inference import config, utils

def predict(x_test = None, y_test = None):
    if (x_test is None) or (y_test is None):
        data = utils.read_dataset_from_csv(config.dataset_path)
        data = utils.format_data(data)
        utils.debug_print_dataset_details(data)
        x_train, x_test, y_train, y_test = utils.get_split_data(data)
    model = utils.load_latest_model()
    predictions = model.predict(x_test)
    accuracy = (predictions == y_test).mean()
    print(f"Model Accuracy: {accuracy:.4f}")
    
from inference import train, predict, utils, config

def main():
    data = utils.read_dataset_from_csv(config.dataset_path)
    data = utils.format_data(data)
    utils.debug_print_dataset_details(data)
    x_train, x_test, y_train, y_test = utils.get_split_data(data)
    
    pipeline = train(x_train, y_train)
    path, version = utils.save_model(pipeline, meta={"dataset_path": config.dataset_path})
    print(f"Model saved at {path} with version {version}")
    predict(x_test, y_test)

if __name__ == "__main__":
    print("stub")

from product_classifier.config.core import config
from product_classifier.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)


if __name__ == "__main__":
    run_training()
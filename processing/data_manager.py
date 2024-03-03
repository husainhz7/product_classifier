
from pathlib import Path
import pandas as pd
from product_classifier.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)
    return text

def remove_between_paranthesis(text):
    return re.sub('\([^])*\]', '', text)



def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    # drop unnecessary variables
    data_frame = data_frame[data_frame.columns[1:3]]

    data_frame.name = data_frame.name.apply(remove_special_characters)
    data_frame.name = data_frame.name.apply(remove_between_paranthesis)

    return data_frame


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed
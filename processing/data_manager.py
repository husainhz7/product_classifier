
from pathlib import Path
import pandas as pd
from config.core import DATASET_DIR, TRAINED_MODEL_DIR, PACKAGE_ROOT, config
import re
import joblib
from sklearn.pipeline import Pipeline

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()

def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)
    return text

def remove_between_paranthesis(text):
    pattern = r'\(([^)]*)\)'
    return re.sub(pattern, '', text)

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    # drop unnecessary variables
    data_frame = data_frame.drop(columns=data_frame.columns.difference(['name', 'main_category']))

    data_frame['name'] = data_frame['name'].apply(remove_special_characters)
    data_frame['name'] = data_frame['name'].apply(remove_between_paranthesis)

    return data_frame


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

    return transformed

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.pipeline_save_file}{__version__}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model
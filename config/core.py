
# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from pathlib import Path

from pydantic import BaseModel
from strictyaml import YAML, load

# Project Directories
PACKAGE_ROOT = Path(parent).resolve().parent
#print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent

CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_name: str
    pipeline_save_file: str
    test_size:float
    random_state: int

def create_and_validate_config(parsed_config: YAML = None) -> AppConfig:
    """Run validation on config values."""
    with open(CONFIG_FILE_PATH, "r") as conf_file:
        parsed_config = load(conf_file.read())

    # specify the data attribute from the strictyaml YAML type.
    _config = AppConfig(**parsed_config.data)

    return _config

config = create_and_validate_config()
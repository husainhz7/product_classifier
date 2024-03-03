
# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path

import product_classifier
# Project Directories
PACKAGE_ROOT = Path(product_classifier.__file__).resolve().parent
#print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent

CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
"""
config.py

Central configuration file
for project paths and settings.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

MODELS_DIR = PROJECT_ROOT / "models"

DOCUMENTS_DIR = PROJECT_ROOT / "documents"

SQL_DIR = PROJECT_ROOT / "sql"


TRAIN_DATA_PATH = (
    DATA_DIR / "application_train.csv"
)

TEST_DATA_PATH = (
    DATA_DIR / "application_test.csv"
)


MODEL_PATH = (
    MODELS_DIR / "lgbm_model.pkl"
)

PREPROCESSOR_PATH = (
    MODELS_DIR / "preprocessor.pkl"
)

FEATURE_NAMES_PATH = (
    MODELS_DIR / "feature_names.json"
)

METADATA_PATH = (
    MODELS_DIR / "model_metadata.json"
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"
RESULTS_DIR = PROJ_ROOT / "results"
EXPERIMENTS_DIR = PROJ_ROOT / "experiments"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAINING_DATA_PATH = DATA_DIR / os.environ.get(
    "TRAINING_DATA_FILENAME", "001_3_cleaned_FTIR.csv"
)

RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 52))
R2_THRESHOLD = float(os.environ.get("R2_THRESHOLD", 0.3))


def init_mlflow() -> bool:
    try:
        import dagshub
        dagshub.init(
            repo_owner=os.environ.get("DAGSHUB_REPO_OWNER", "pedroasvalente"),
            repo_name=os.environ.get("DAGSHUB_REPO_NAME", "ftir-physiology-prediction"),
            mlflow=True,
        )
        logger.info("MLflow → DagsHub")
        return True
    except Exception as exc:
        logger.warning(f"DagsHub init failed, using local MLflow: {exc}")
        return False

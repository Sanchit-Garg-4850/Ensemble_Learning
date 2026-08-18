"""
Single source of truth for paths, constants and run params.
Every stage script imports from here instead of hardcoding paths, so the
orchestrator and CI can override things (e.g. DATA_PATH, N_TRIALS) via env vars.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"

RAW_DATA_PATH = Path(os.environ.get("FRAUD_DATA_PATH", DATA_DIR / "creditcard.csv"))
PROCESSED_DATA_PATH = OUTPUT_DIR / "processed.parquet"

TARGET_COL = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Optuna: overridable via env var so CI can run a 2-trial smoke test
# while a local run does the real search (e.g. 50-100 trials).
N_TRIALS = int(os.environ.get("FRAUD_N_TRIALS", 25))
CV_FOLDS = int(os.environ.get("FRAUD_CV_FOLDS", 5))
OPTUNA_STORAGE = f"sqlite:///{OUTPUT_DIR / 'optuna_studies.db'}"

# Shadow-model decision-surface subsample size (Section 10 of the plan)
SURFACE_SUBSAMPLE = 2500

TIERS = {
    "tier1_single": ["decision_tree", "logistic_regression", "knn"],
    "tier2_ensemble": ["random_forest", "bagging", "adaboost", "xgboost", "lightgbm"],
    "tier3_meta": ["voting", "stacking"],
}

RESAMPLERS = [
    "none",  # class_weight="balanced" instead
    "random_over",
    "random_under",
    "smote",
    "adasyn",
    "smotetomek",
    "smoteenn",
]

for d in (DATA_DIR, OUTPUT_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

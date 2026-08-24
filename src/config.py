"""
Single source of truth for paths, constants and run params.
Every stage script imports from here instead of hardcoding paths, so the
orchestrator and CI can override things (e.g. DATA_PATH, N_TRIALS) via env vars.

--- Experiment tracking / versioning ---
RUN_VERSION identifies one tuning experiment (e.g. "v1" = original unregularized
search space, "v2" = regularized search space after the model_configs.py update).
Every experiment-specific artifact name is built through versioned_name(), so
re-running the pipeline under a new RUN_VERSION never overwrites a prior run's
results, models, cached surfaces, or Optuna study. To start a new experiment:
    set FRAUD_RUN_VERSION=v2   (Windows CMD)  /  export FRAUD_RUN_VERSION=v2 (bash)
or just bump the default below and don't set the env var.
Preprocessing artifacts (PROCESSED_DATA_PATH, train.parquet, test.parquet) are
NOT versioned — they're shared across experiments since regularization changes
don't touch the data split/scaling stage.

--- MLflow ---
Optuna's own SQLite DB (OPTUNA_STORAGE below) already tracks every individual
hyperparameter trial. MLflow sits alongside it at a coarser grain: one run per
FINAL model result (best trial refit + test metrics + the model artifact
itself), giving a browsable dashboard instead of grepping output/results/*.json.

MLFLOW_TRACKING_URI defaults to a LOCAL file store (./mlruns, gitignored) --
fine for solo local runs, but invisible to GitHub Actions since mlruns/ is
never pushed. Set the MLFLOW_TRACKING_URI env var to a shared server (e.g. a
DagsHub repo's MLflow endpoint: https://dagshub.com/<user>/<repo>.mlflow) so
local training and CI both log to -- and can query -- the same place. DagsHub
also needs MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD in the
environment; mlflow's client reads those automatically, no code needed here.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Loads variables from a .env file at the project root (gitignored --
# MLFLOW_TRACKING_URI / USERNAME / PASSWORD live there) into the actual
# environment, so `set FRAUD_RUN_VERSION=v2` etc. still work as one-off
# overrides, but the DagsHub credentials don't need to be retyped every
# cmd session. Safe if python-dotenv isn't installed or .env doesn't
# exist yet -- just silently skips loading anything.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

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
]

TRIAL_TIMEOUT_SECONDS = {
    "decision_tree": 180,
    "logistic_regression": 360,
    "knn": 360,
    "random_forest": 600,
    "adaboost": 600,
    "xgboost": 600,
    "lightgbm": 600,
}

# ---------------------------------------------------------------------------
# Experiment / run versioning
# ---------------------------------------------------------------------------
RUN_VERSION = os.environ.get("FRAUD_RUN_VERSION", "v1")


def versioned_name(base: str) -> str:
    """'xgboost' -> 'xgboost_v1' (or whatever RUN_VERSION currently is)."""
    return f"{base}_{RUN_VERSION}"


# Optuna: all runs share one SQLite file, but each version gets its own
# study name, so trials from different search spaces never mix. Old studies
# stay queryable: optuna.load_study(study_name="xgboost_v1", storage=OPTUNA_STORAGE)
OPTUNA_STORAGE = f"sqlite:///{OUTPUT_DIR / 'optuna_studies.db'}"


def optuna_study_name(model_name: str) -> str:
    return versioned_name(model_name)


RESULTS_DIR = OUTPUT_DIR / "results"
SURFACES_DIR = OUTPUT_DIR / "surfaces"
MODELS_VERSIONED_DIR = MODELS_DIR  # models saved directly here with versioned filenames


def results_path(model_name: str, version: str = None) -> Path:
    """output/results/xgboost_v2.json"""
    v = version or RUN_VERSION
    return RESULTS_DIR / f"{model_name}_{v}.json"


def model_path(model_name: str, version: str = None) -> Path:
    """models/xgboost_v2.joblib"""
    v = version or RUN_VERSION
    return MODELS_VERSIONED_DIR / f"{model_name}_{v}.joblib"


def proba_path(model_name: str, version: str = None) -> Path:
    """output/probas/xgboost_v2.npy"""
    v = version or RUN_VERSION
    return OUTPUT_DIR / "probas" / f"{model_name}_{v}.npy"


COMPARISON_CSV_PATH = OUTPUT_DIR / f"model_comparison_{RUN_VERSION}.csv"
APP_MANIFEST_PATH = OUTPUT_DIR / f"app_manifest_{RUN_VERSION}.json"

# ---------------------------------------------------------------------------
# MLflow tracking
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_DIR = ROOT / "mlruns"  # local fallback only (gitignored, single-machine)

# Env var wins (e.g. DagsHub URL) -- falls back to the local file store,
# which works for solo local runs but is invisible to CI.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_DIR.as_uri())
MLFLOW_EXPERIMENT_NAME = os.environ.get("FRAUD_MLFLOW_EXPERIMENT", "fraud-detection-ensemble")

for d in (DATA_DIR, OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, SURFACES_DIR, OUTPUT_DIR / "probas"):
    d.mkdir(parents=True, exist_ok=True)

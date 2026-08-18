"""
Registry of model factories + their Optuna search spaces.
Keeping this in one file means adding a model = one new entry, and both
train.py and the Streamlit app stay in sync automatically.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import RANDOM_STATE


def build_model(name: str, trial):
    """Given an Optuna trial, sample hyperparams and return an unfitted estimator."""
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("dt_max_depth", 2, 20),
            min_samples_split=trial.suggest_int("dt_min_samples_split", 2, 50),
            min_samples_leaf=trial.suggest_int("dt_min_samples_leaf", 1, 20),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if name == "logistic_regression":
        return LogisticRegression(
            C=trial.suggest_float("lr_C", 1e-3, 10, log=True),
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("knn_k", 3, 25),
            weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 500),
            max_depth=trial.suggest_int("rf_max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 10),
            class_weight="balanced",
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "bagging":
        return BaggingClassifier(
            n_estimators=trial.suggest_int("bag_n_estimators", 10, 100),
            max_samples=trial.suggest_float("bag_max_samples", 0.5, 1.0),
            max_features=trial.suggest_float("bag_max_features", 0.5, 1.0),
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "adaboost":
        return AdaBoostClassifier(
            n_estimators=trial.suggest_int("ada_n_estimators", 50, 300),
            learning_rate=trial.suggest_float("ada_lr", 0.01, 1.0, log=True),
            random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 100, 600),
            max_depth=trial.suggest_int("xgb_max_depth", 3, 10),
            learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.5, 1.0),
            eval_metric="aucpr",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("lgbm_n_estimators", 100, 600),
            max_depth=trial.suggest_int("lgbm_max_depth", 3, 12),
            learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("lgbm_subsample", 0.5, 1.0),
            num_leaves=trial.suggest_int("lgbm_num_leaves", 15, 127),
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def build_meta_model(name: str, base_estimators: list):
    """Tier 3: voting / stacking over already-tuned Tier 1/2 estimators."""
    if name == "voting":
        return VotingClassifier(estimators=base_estimators, voting="soft", n_jobs=-1)
    if name == "stacking":
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced",n_jobs=-1),
            n_jobs=-1,
        )
    raise ValueError(f"Unknown meta model: {name}")


OOB_CAPABLE = {"random_forest", "bagging"}

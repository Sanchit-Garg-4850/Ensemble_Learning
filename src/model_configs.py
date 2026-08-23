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
            max_depth=trial.suggest_int("dt_max_depth", 2, 12),
            min_samples_split=trial.suggest_int("dt_min_samples_split", 5, 100),
            min_samples_leaf=trial.suggest_int("dt_min_samples_leaf", 2, 50),
            max_leaf_nodes=trial.suggest_int("dt_max_leaf_nodes", 10, 100),
            ccp_alpha=trial.suggest_float("dt_ccp_alpha", 1e-5, 1e-2, log=True),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if name == "logistic_regression":
        return LogisticRegression(
            C=trial.suggest_float("lr_C", 1e-4, 1.0, log=True),
            penalty="l2",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("knn_k", 5, 50),
            weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 500),
            max_depth=trial.suggest_int("rf_max_depth", 3, 12),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 2, 20),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 0.5]),
            max_samples=trial.suggest_float("rf_max_samples", 0.5, 1.0),
            class_weight="balanced",
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "bagging":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=trial.suggest_int("bag_base_max_depth", 2, 8)),
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
            estimator=DecisionTreeClassifier(max_depth=trial.suggest_int("ada_base_max_depth", 1, 4)),
            n_estimators=trial.suggest_int("ada_n_estimators", 50, 300),
            learning_rate=trial.suggest_float("ada_lr", 0.01, 1.0, log=True),
            algorithm="SAMME",
            random_state=RANDOM_STATE,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 100, 600),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 7),
            learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.5, 1.0),
            min_child_weight=trial.suggest_int("xgb_min_child_weight", 1, 20),
            gamma=trial.suggest_float("xgb_gamma", 1e-4, 5.0, log=True),
            reg_alpha=trial.suggest_float("xgb_reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-4, 10.0, log=True),
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("lgbm_n_estimators", 100, 600),
            max_depth=trial.suggest_int("lgbm_max_depth", 2, 8),
            learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("lgbm_subsample", 0.5, 1.0),
            num_leaves=trial.suggest_int("lgbm_num_leaves", 7, 63),
            min_child_samples=trial.suggest_int("lgbm_min_child_samples", 10, 100),
            reg_alpha=trial.suggest_float("lgbm_reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("lgbm_reg_lambda", 1e-4, 10.0, log=True),
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
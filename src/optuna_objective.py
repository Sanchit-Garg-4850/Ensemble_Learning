"""
Joint Optuna objective: model + resampler + hyperparams, scored by mean
cross-validated PR-AUC (average_precision), computed inside an imblearn
Pipeline so resampling happens fresh per fold (no leakage).
"""
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import RANDOM_STATE, CV_FOLDS, RESAMPLERS
from model_configs import build_model
from resamplers import build_resampler


def make_objective(model_name: str, X, y):
    def objective(trial):
        resampler_name = trial.suggest_categorical("resampler", RESAMPLERS)
        resampler = build_resampler(resampler_name, trial)
        model = build_model(model_name, trial)

        steps = []
        if resampler is not None:
            steps.append(("resample", resampler))
        steps.append(("model", model))
        pipe = Pipeline(steps)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1)
        except ValueError:
            # e.g. SMOTE/ADASYN's k_neighbors > minority-class count in a given
            # fold (can happen with a small dataset, a small CV_FOLDS override,
            # or a rare resampler+fold combo). Treat as a bad trial rather than
            # crashing the whole study.
            return 0.0
        return float(np.mean(scores))

    return objective

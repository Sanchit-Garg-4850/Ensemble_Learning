"""
Joint Optuna objective: model + resampler + hyperparams, scored by mean
cross-validated PR-AUC (average_precision), computed inside an imblearn
Pipeline so resampling happens fresh per fold (no leakage).

Each trial's CV fit runs in a separate subprocess (spawn context, required
on Windows) so a pathologically slow trial -- e.g. a deep tree on an
oversampled fold, or an expensive resampler like SMOTEENN -- can actually be
killed once it exceeds TRIAL_TIMEOUT_SECONDS[model_name], rather than just
abandoned while it keeps burning CPU in the background. A thread-based
timeout can't do this: sklearn/imblearn fits release the GIL during the
compute-heavy C code, so a "timed out" thread keeps running regardless.
Timed-out and errored trials return 0.0, matching the existing
bad-trial convention (see the ValueError handling below).
"""
import multiprocessing as mp

import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import RANDOM_STATE, CV_FOLDS, RESAMPLERS, TRIAL_TIMEOUT_SECONDS
from model_configs import build_model
from resamplers import build_resampler


def _cv_worker(pipe, X, y, cv_folds, random_state, queue):
    """Runs in a child process. Puts ('ok', score) or ('error', msg) on the queue."""
    try:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1)
        queue.put(("ok", float(np.mean(scores))))
    except ValueError:
        # e.g. SMOTE/ADASYN's k_neighbors > minority-class count in a fold
        queue.put(("ok", 0.0))
    except Exception as e:  # noqa: BLE001 -- any other fit failure also becomes a bad trial
        queue.put(("error", repr(e)))


def _cv_score_with_timeout(pipe, X, y, cv_folds, random_state, timeout_seconds):
    if timeout_seconds is None:
        # No timeout configured for this model -- run inline, same as before.
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1)
            return float(np.mean(scores))
        except ValueError:
            return 0.0

    ctx = mp.get_context("spawn")  # spawn is required for a real kill on Windows
    queue = ctx.Queue()
    proc = ctx.Process(target=_cv_worker, args=(pipe, X, y, cv_folds, random_state, queue))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"[optuna] trial exceeded {timeout_seconds}s timeout -- killed, scored 0.0")
        return 0.0

    status, payload = queue.get()
    if status == "error":
        print(f"[optuna] trial failed: {payload} -- scored 0.0")
        return 0.0
    return payload


def make_objective(model_name: str, X, y):
    timeout_seconds = TRIAL_TIMEOUT_SECONDS.get(model_name)  # None = no cap for this model

    def objective(trial):
        resampler_name = trial.suggest_categorical("resampler", RESAMPLERS)
        resampler = build_resampler(resampler_name, trial)
        model = build_model(model_name, trial)

        steps = []
        if resampler is not None:
            steps.append(("resample", resampler))
        steps.append(("model", model))
        pipe = Pipeline(steps)

        return _cv_score_with_timeout(pipe, X, y, CV_FOLDS, RANDOM_STATE, timeout_seconds)

    return objective

"""
Stage: train
For each model in Tier 1 + Tier 2: run an Optuna study (joint resampler +
hyperparam search), refit the best pipeline on the full training set,
evaluate on the held-out test set, cache predict_proba for the Streamlit
threshold slider, and record OOB score where applicable.
Then builds Tier 3 (Voting, Stacking) from the best Tier 1/2 models.

Checkpointed: each model's result is written to output/results/<model>.json
and skipped on re-run unless FRAUD_FORCE_RETRAIN=1 is set — mirrors the
checkpoint-centralization pattern used in the SAP GRC pipeline.
"""
import json
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

from config import (
    OUTPUT_DIR, MODELS_DIR, TARGET_COL, N_TRIALS, TIERS,
    OPTUNA_STORAGE, RANDOM_STATE,
)
from model_configs import build_model, build_meta_model, OOB_CAPABLE
from resamplers import build_resampler
from optuna_objective import make_objective

RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FORCE_RETRAIN = os.environ.get("FRAUD_FORCE_RETRAIN", "0") == "1"
SKIP_OPTUNA_MODELS = {m.strip() for m in os.environ.get("FRAUD_SKIP_OPTUNA", "").split(",") if m.strip()}


def _checkpoint_path(name):
    return RESULTS_DIR / f"{name}.json"


def _already_done(name):
    return _checkpoint_path(name).exists() and not FORCE_RETRAIN


def _eval(y_true, proba, pred) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
    }


def train_single_model(name, X_train, y_train, X_test, y_test):
    if _already_done(name):
        print(f"[train] skip {name} (checkpoint exists)")
        return json.loads(_checkpoint_path(name).read_text())

    print(f"[train] Optuna study for {name} ({N_TRIALS} trials)")
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        study_name=name,
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
    )

    if name in SKIP_OPTUNA_MODELS:
        n_done = len([t for t in study.trials if t.value is not None])
        if n_done == 0:
            raise RuntimeError(f"FRAUD_SKIP_OPTUNA includes '{name}' but no completed trials exist yet.")
        print(f"[train] {name} in FRAUD_SKIP_OPTUNA — accepting best of {n_done} completed trials "
              f"(value={study.best_value:.4f}) instead of continuing to {N_TRIALS}")
    else:
        study.optimize(make_objective(name, X_train, y_train), n_trials=N_TRIALS, show_progress_bar=True)

    best_trial = study.best_trial
    worst_trial = min(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))

    # Refit best pipeline on full training data
    resampler = build_resampler(best_trial.params["resampler"], trial=_FrozenTrialAdapter(best_trial))
    model = build_model(name, _FrozenTrialAdapter(best_trial))
    steps = ([("resample", resampler)] if resampler is not None else []) + [("model", model)]
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = _eval(y_test, proba, pred)

    oob = None
    if name in OOB_CAPABLE:
        try:
            oob = float(pipe.named_steps["model"].oob_score_)
        except Exception:
            oob = None

    joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")
    np.save(RESULTS_DIR / f"{name}_test_proba.npy", proba)

    result = {
        "model": name,
        "best_params": best_trial.params,
        "cv_pr_auc": best_trial.value,
        "worst_trial_params": worst_trial.params,
        "worst_trial_cv_pr_auc": worst_trial.value,
        "test_metrics": metrics,
        "oob_score": oob,
        "train_seconds": round(time.time() - t0, 1),
    }
    _checkpoint_path(name).write_text(json.dumps(result, indent=2))
    print(f"[train] {name}: test PR-AUC={metrics['pr_auc']:.4f} "
          f"(cv={best_trial.value:.4f}) oob={oob}")
    return result


class _FrozenTrialAdapter:
    """Lets us replay a completed trial's fixed params through build_model/
    build_resampler, which expect an Optuna trial object with .suggest_*()."""
    def __init__(self, trial):
        self.params = trial.params

    def suggest_int(self, name, *a, **k):
        return self.params[name]

    def suggest_float(self, name, *a, **k):
        return self.params[name]

    def suggest_categorical(self, name, *a, **k):
        return self.params[name]


def train_meta_models(single_results, X_train, y_train, X_test, y_test, top_k=4):
    """Tier 3: pick the top_k Tier1/2 models by CV PR-AUC as base estimators."""
    ranked = sorted(single_results, key=lambda r: r["cv_pr_auc"], reverse=True)[:top_k]
    base_estimators = [(r["model"], joblib.load(MODELS_DIR / f"{r['model']}.joblib")) for r in ranked]

    meta_results = []
    for meta_name in TIERS["tier3_meta"]:
        if _already_done(meta_name):
            meta_results.append(json.loads(_checkpoint_path(meta_name).read_text()))
            continue
        print(f"[train] fitting meta model: {meta_name} on base={[n for n, _ in base_estimators]}")
        meta = build_meta_model(meta_name, base_estimators)
        meta.fit(X_train, y_train)
        proba = meta.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = _eval(y_test, proba, pred)
        joblib.dump(meta, MODELS_DIR / f"{meta_name}.joblib")
        np.save(RESULTS_DIR / f"{meta_name}_test_proba.npy", proba)
        result = {"model": meta_name, "base_models": [n for n, _ in base_estimators], "test_metrics": metrics}
        _checkpoint_path(meta_name).write_text(json.dumps(result, indent=2))
        meta_results.append(result)
        print(f"[train] {meta_name}: test PR-AUC={metrics['pr_auc']:.4f}")
    return meta_results


def run():
    train_df = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    test_df = pd.read_parquet(OUTPUT_DIR / "test.parquet")
    X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

    single_results = []
    for tier in ("tier1_single", "tier2_ensemble"):
        for name in TIERS[tier]:
            single_results.append(train_single_model(name, X_train, y_train, X_test, y_test))

    meta_results = train_meta_models(single_results, X_train, y_train, X_test, y_test)

    summary = pd.DataFrame(
        [{"model": r["model"], "tier": _tier_of(r["model"]), **r["test_metrics"]} for r in single_results + meta_results]
    ).sort_values("pr_auc", ascending=False)
    summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\n[train] Final comparison:\n", summary.to_string(index=False))


def _tier_of(name):
    for tier, models in TIERS.items():
        if name in models:
            return tier
    return "unknown"


if __name__ == "__main__":
    run()

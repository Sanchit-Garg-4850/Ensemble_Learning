"""
Stage: train
For each model in Tier 1 + Tier 2: run an Optuna study (joint resampler +
hyperparam search), refit the best pipeline on the full training set,
evaluate on the held-out test set, cache predict_proba for the Streamlit
threshold slider, and record OOB score where applicable.
Then builds Tier 3 (Voting, Stacking) from the best Tier 1/2 models.

Checkpointed: each model's result is written to output/results/<model>_<RUN_VERSION>.json
and skipped on re-run unless FRAUD_FORCE_RETRAIN=1 is set — mirrors the
checkpoint-centralization pattern used in the SAP GRC pipeline.

Every artifact (Optuna study, results JSON, saved model, cached proba,
comparison CSV, app manifest) is namespaced by RUN_VERSION (config.py), so
re-running under a new FRAUD_RUN_VERSION never overwrites a prior experiment.

MLflow: Optuna's own SQLite DB already tracks every trial. MLflow logs one
run per FINAL model result (best trial's params + test metrics + the fitted
model artifact) — coarser grain, meant for browsing/comparing final results
across models and across RUN_VERSIONs in the mlflow UI, not for the raw
trial-by-trial search. Checkpoint-skipped models (already trained under this
RUN_VERSION) do NOT get a new mlflow run, so the dashboard reflects actual
work done, not re-runs of the orchestrator.

Tracking URI comes from config.MLFLOW_TRACKING_URI, which reads the
MLFLOW_TRACKING_URI env var (e.g. a DagsHub endpoint) if set, falling back
to a local ./mlruns file store otherwise. Point it at a shared server so
this run's results are visible from CI too (see compare_and_deploy.py).
"""
import json
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

from config import (
    OUTPUT_DIR, TARGET_COL, N_TRIALS, TIERS, RUN_VERSION,
    OPTUNA_STORAGE,
    optuna_study_name, results_path, model_path, proba_path,
    COMPARISON_CSV_PATH,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
)
from model_configs import build_model, build_meta_model, OOB_CAPABLE
from resamplers import build_resampler
from optuna_objective import make_objective

# MLflow's default sklearn artifact format ("skops") runs a security audit
# that only recognizes a small default set of trusted types — it rejected
# imblearn's Pipeline/resampler classes, then xgboost's, then even stdlib
# collections.OrderedDict, one crash at a time as each new model type hit
# it. Confirmed by reading mlflow/sklearn/__init__.py directly: the audit
# only runs when serialization_format == SERIALIZATION_FORMAT_SKOPS (the
# default) — passing "pickle" instead skips that code path entirely, so
# there's no type whitelist to maintain going forward. Trade-off: pickle
# doesn't get skops's extra safety guarantee against loading a malicious
# model file, which is a reasonable trade for a personal project logging
# to your own private DagsHub repo.
MLFLOW_SERIALIZATION_FORMAT = mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE

FORCE_RETRAIN = os.environ.get("FRAUD_FORCE_RETRAIN", "0") == "1"
SKIP_OPTUNA_MODELS = {m.strip() for m in os.environ.get("FRAUD_SKIP_OPTUNA", "").split(",") if m.strip()}


def _init_mlflow():
    """Called once from run(), not at module import time.

    Windows' multiprocessing 'spawn' context (used by the per-trial timeout
    wrapper in optuna_objective.py) re-imports this module in every child
    process it creates. If mlflow.set_experiment(...) ran at module level,
    that live network call to the DagsHub-hosted tracking server would fire
    on EVERY Optuna trial's spawned subprocess, not just once at startup —
    which is slow/flaky enough on a remote call to look like every trial is
    hanging and getting killed by the trial timeout, regardless of resampler
    or model. Keeping it in a function that only run() calls means spawned
    children skip it entirely.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"[train] mlflow tracking uri: {MLFLOW_TRACKING_URI}")


def _already_done(name):
    return results_path(name).exists() and not FORCE_RETRAIN


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
        print(f"[train] skip {name} ({RUN_VERSION}, checkpoint exists)")
        return json.loads(results_path(name).read_text())

    print(f"[train] Optuna study for {name} ({N_TRIALS} trials, run={RUN_VERSION})")
    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        study_name=optuna_study_name(name),
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
        # Top up to N_TRIALS TOTAL, not N_TRIALS more — matters when resuming
        # a study that already has completed trials (e.g. the run crashed
        # after Optuna finished but before the checkpoint JSON was written,
        # or a prior run was interrupted partway through).
        n_done = len([t for t in study.trials if t.value is not None])
        remaining = max(N_TRIALS - n_done, 0)
        if remaining == 0:
            print(f"[train] {name} already has {n_done}/{N_TRIALS} completed trials — "
                  f"skipping further optimization, using existing best (value={study.best_value:.4f})")
        else:
            if n_done > 0:
                print(f"[train] {name} resuming: {n_done}/{N_TRIALS} trials already done, "
                      f"running {remaining} more")
            study.optimize(make_objective(name, X_train, y_train), n_trials=remaining, show_progress_bar=True)

    best_trial = study.best_trial
    worst_trial = min(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))

    # Refit best pipeline on full training data
    resampler = build_resampler(best_trial.params["resampler"], trial=_FrozenTrialAdapter(best_trial))
    model = build_model(name, _FrozenTrialAdapter(best_trial))
    steps = ([("resample", resampler)] if resampler is not None else []) + [("model", model)]
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    proba_train = pipe.predict_proba(X_train)[:, 1]
    pred_train = (proba_train >= 0.5).astype(int)
    train_metrics = _eval(y_train, proba_train, pred_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = _eval(y_test, proba, pred)

    # Positive gap = overfitting (train PR-AUC higher than test). Near-zero or
    # negative is the healthy range; large positive is the regularization
    # search hasn't closed the gap yet.
    pr_auc_gap = round(train_metrics["pr_auc"] - metrics["pr_auc"], 4)

    oob = None
    if name in OOB_CAPABLE:
        try:
            oob = float(pipe.named_steps["model"].oob_score_)
        except Exception:
            oob = None

    train_seconds = round(time.time() - t0, 1)

    joblib.dump(pipe, model_path(name))
    np.save(proba_path(name), proba)

    with mlflow.start_run(run_name=f"{name}_{RUN_VERSION}", nested=True):
        mlflow.set_tags({"run_version": RUN_VERSION, "model": name, "tier": _tier_of(name)})
        mlflow.log_param("resampler", best_trial.params.get("resampler"))
        mlflow.log_params({k: v for k, v in best_trial.params.items() if k != "resampler"})
        mlflow.log_metric("cv_pr_auc", best_trial.value)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metric("pr_auc_gap", pr_auc_gap)
        if oob is not None:
            mlflow.log_metric("oob_score", oob)
        mlflow.log_metric("train_seconds", train_seconds)
        mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format=MLFLOW_SERIALIZATION_FORMAT)

    result = {
        "model": name,
        "run_version": RUN_VERSION,
        "best_params": best_trial.params,
        "cv_pr_auc": best_trial.value,
        "worst_trial_params": worst_trial.params,
        "worst_trial_cv_pr_auc": worst_trial.value,
        "train_metrics": train_metrics,
        "test_metrics": metrics,
        "pr_auc_gap": pr_auc_gap,
        "oob_score": oob,
        "train_seconds": train_seconds,
    }
    results_path(name).write_text(json.dumps(result, indent=2))
    print(f"[train] {name}: test PR-AUC={metrics['pr_auc']:.4f} "
          f"train PR-AUC={train_metrics['pr_auc']:.4f} (gap={pr_auc_gap:+.4f}) "
          f"cv={best_trial.value:.4f} oob={oob}")
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
    base_estimators = [(r["model"], joblib.load(model_path(r["model"]))) for r in ranked]

    meta_results = []
    for meta_name in TIERS["tier3_meta"]:
        if _already_done(meta_name):
            meta_results.append(json.loads(results_path(meta_name).read_text()))
            continue
        print(f"[train] fitting meta model: {meta_name} on base={[n for n, _ in base_estimators]} (run={RUN_VERSION})")
        t0 = time.time()
        meta = build_meta_model(meta_name, base_estimators)
        meta.fit(X_train, y_train)

        proba_train = meta.predict_proba(X_train)[:, 1]
        pred_train = (proba_train >= 0.5).astype(int)
        train_metrics = _eval(y_train, proba_train, pred_train)

        proba = meta.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = _eval(y_test, proba, pred)
        pr_auc_gap = round(train_metrics["pr_auc"] - metrics["pr_auc"], 4)
        train_seconds = round(time.time() - t0, 1)

        joblib.dump(meta, model_path(meta_name))
        np.save(proba_path(meta_name), proba)

        with mlflow.start_run(run_name=f"{meta_name}_{RUN_VERSION}", nested=True):
            mlflow.set_tags({"run_version": RUN_VERSION, "model": meta_name, "tier": "tier3_meta"})
            mlflow.log_param("base_models", ",".join(n for n, _ in base_estimators))
            mlflow.log_metrics(metrics)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metric("pr_auc_gap", pr_auc_gap)
            mlflow.log_metric("train_seconds", train_seconds)
            mlflow.sklearn.log_model(meta, artifact_path="model", serialization_format=MLFLOW_SERIALIZATION_FORMAT)

        result = {
            "model": meta_name,
            "run_version": RUN_VERSION,
            "base_models": [n for n, _ in base_estimators],
            "train_metrics": train_metrics,
            "test_metrics": metrics,
            "pr_auc_gap": pr_auc_gap,
        }
        results_path(meta_name).write_text(json.dumps(result, indent=2))
        meta_results.append(result)
        print(f"[train] {meta_name}: test PR-AUC={metrics['pr_auc']:.4f} "
              f"train PR-AUC={train_metrics['pr_auc']:.4f} (gap={pr_auc_gap:+.4f})")
    return meta_results


def run():
    train_df = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    test_df = pd.read_parquet(OUTPUT_DIR / "test.parquet")
    X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

    with mlflow.start_run(run_name=f"pipeline_{RUN_VERSION}"):
        mlflow.set_tag("run_version", RUN_VERSION)

        single_results = []
        for tier in ("tier1_single", "tier2_ensemble"):
            for name in TIERS[tier]:
                single_results.append(train_single_model(name, X_train, y_train, X_test, y_test))

        meta_results = train_meta_models(single_results, X_train, y_train, X_test, y_test)

        all_results = single_results + meta_results
        summary = pd.DataFrame(
            [{
                "model": r["model"],
                "tier": _tier_of(r["model"]),
                "train_pr_auc": r.get("train_metrics", {}).get("pr_auc"),
                "pr_auc_gap": r.get("pr_auc_gap"),
                **r["test_metrics"],
            } for r in all_results]
        ).sort_values("pr_auc", ascending=False)
        summary.to_csv(COMPARISON_CSV_PATH, index=False)

        mlflow.log_metric("best_pr_auc", float(summary.iloc[0]["pr_auc"]))
        mlflow.set_tag("best_model", summary.iloc[0]["model"])
        mlflow.log_artifact(str(COMPARISON_CSV_PATH))

    print(f"\n[train] Final comparison (run={RUN_VERSION}):\n", summary.to_string(index=False))
    print(f"[train] best model: {summary.iloc[0]['model']} (PR-AUC={summary.iloc[0]['pr_auc']:.4f})")


def _tier_of(name):
    for tier, models in TIERS.items():
        if name in models:
            return tier
    return "unknown"


if __name__ == "__main__":
    run()

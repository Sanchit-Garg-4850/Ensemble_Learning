"""
Shadow-model decision surfaces (plan Section 10). Fits the SAME model class
and best hyperparams as the tuned production model, but restricted to a 2D
feature subset, on a stratified subsample. Explicitly illustrative, not the
production model. Cached to output/surfaces/<model>_<version>__<fx>__<fy>.joblib
so repeated Streamlit toggles don't refit, and so different experiment
versions (e.g. v1 unregularized vs v2 regularized) never share a cache entry.

NOTE: surfaces store a hard "pred" grid (0/1 class output), not "proba".
If you have old cached files from the probability-based version, delete
output/surfaces/*.joblib once before re-running — old payloads only have a
"proba" key and will KeyError against the "pred"-based Streamlit app.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR, TARGET_COL, SURFACE_SUBSAMPLE, RANDOM_STATE,
    RUN_VERSION, SURFACES_DIR, results_path as config_results_path,
)
from model_configs import build_model

SURFACES_DIR.mkdir(parents=True, exist_ok=True)


class _ParamsAdapter:
    """Replays a fixed best_params dict through build_model's trial.suggest_* calls.

    If a param is missing from the stored best_params (e.g. an older tuning run
    predates a param that was later added to model_configs.py's search space),
    falls back to a default drawn from the same range/choices build_model would
    have searched, rather than crashing with a KeyError. A warning is printed
    so the gap is visible instead of silently masked.
    """
    def __init__(self, params, model_name=None):
        self.params = params
        self.model_name = model_name

    def _warn_missing(self, name, fallback):
        print(
            f"[decision_surfaces] '{self.model_name}': param '{name}' missing from "
            f"stored best_params — using fallback {fallback!r} instead. Likely means "
            f"the results JSON predates a search-space change in model_configs.py."
        )

    def suggest_int(self, name, low, high, *a, **k):
        if name in self.params:
            return self.params[name]
        fallback = (low + high) // 2
        self._warn_missing(name, fallback)
        return fallback

    def suggest_float(self, name, low, high, *a, **k):
        if name in self.params:
            return self.params[name]
        fallback = (low + high) / 2
        self._warn_missing(name, fallback)
        return fallback

    def suggest_categorical(self, name, choices, *a, **k):
        if name in self.params:
            return self.params[name]
        fallback = choices[0]
        self._warn_missing(name, fallback)
        return fallback


def _cache_key(model_name, fx, fy, version):
    raw = f"{model_name}_{version}__{fx}__{fy}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def get_or_build_surface(model_name: str, fx: str, fy: str, resolution: int = 200, version: str = None):
    version = version or RUN_VERSION
    key = _cache_key(model_name, fx, fy, version)
    cache_path = SURFACES_DIR / f"{key}.joblib"
    if cache_path.exists():
        return joblib.load(cache_path)

    results_file = config_results_path(model_name, version)
    if not results_file.exists():
        raise ValueError(
            f"No results file found for '{model_name}' ({version}) at {results_file}. "
            f"The model needs to be trained under this run version before a "
            f"shadow decision surface can be built for it."
        )
    stored = json.loads(results_file.read_text())
    if "best_params" not in stored:
        raise ValueError(
            f"'{model_name}' has no tunable hyperparameters to build a shadow model from "
            f"(it's a Tier-3 meta-ensemble — decision surfaces aren't supported for voting/stacking, "
            f"only individual Tier 1/2 models)."
        )
    best_params = {k: v for k, v in stored["best_params"].items() if k != "resampler"}

    train_df = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    sub = train_df.groupby(TARGET_COL, group_keys=False).apply(
        lambda g: g.sample(min(len(g), SURFACE_SUBSAMPLE // 2), random_state=RANDOM_STATE)
    )
    X2 = sub[[fx, fy]].values
    y = sub[TARGET_COL].values

    shadow = build_model(model_name, _ParamsAdapter(best_params, model_name=model_name))
    shadow.fit(X2, y)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid_pred = shadow.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    payload = {"xx": xx, "yy": yy, "pred": grid_pred, "points": X2, "labels": y, "fx": fx, "fy": fy}
    joblib.dump(payload, cache_path)
    return payload


def run():
    """Pipeline-stage entry point: pre-warm the cache for the default
    top-2 fraud-correlated feature pair across every trained model in the
    CURRENT RUN_VERSION, so the Streamlit app's first load is instant
    instead of fitting shadow models on demand."""
    from config import TIERS
    results_dir = SURFACES_DIR.parent / "results"
    if not results_dir.exists():
        print("[decision_surfaces] no trained models yet — skipping cache warm.")
        return
    shadow_capable = set(TIERS["tier1_single"]) | set(TIERS["tier2_ensemble"])
    suffix = f"_{RUN_VERSION}"
    model_names = [
        p.stem[: -len(suffix)]
        for p in results_dir.glob(f"*{suffix}.json")
        if p.stem[: -len(suffix)] in shadow_capable
    ]
    for m in model_names:
        try:
            get_or_build_surface(m, "V14", "V17")
        except Exception as e:
            print(f"[decision_surfaces] skip {m}: {e}")
    print(f"[decision_surfaces] warmed cache for {len(model_names)} models on (V14, V17), run={RUN_VERSION}.")


if __name__ == "__main__":
    run()

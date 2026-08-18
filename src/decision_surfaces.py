"""
Shadow-model decision surfaces (plan Section 10). Fits the SAME model class
and best hyperparams as the tuned production model, but restricted to a 2D
feature subset, on a stratified subsample. Explicitly illustrative, not the
production model. Cached to output/surfaces/<model>__<fx>__<fy>.joblib so
repeated Streamlit toggles don't refit.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TARGET_COL, SURFACE_SUBSAMPLE, RANDOM_STATE, MODELS_DIR
from model_configs import build_model

SURFACES_DIR = OUTPUT_DIR / "surfaces"
SURFACES_DIR.mkdir(parents=True, exist_ok=True)


class _ParamsAdapter:
    """Replays a fixed best_params dict through build_model's trial.suggest_* calls."""
    def __init__(self, params):
        self.params = params

    def suggest_int(self, name, *a, **k):
        return self.params[name]

    def suggest_float(self, name, *a, **k):
        return self.params[name]

    def suggest_categorical(self, name, *a, **k):
        return self.params[name]


def _cache_key(model_name, fx, fy):
    raw = f"{model_name}__{fx}__{fy}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def get_or_build_surface(model_name: str, fx: str, fy: str, resolution: int = 200):
    key = _cache_key(model_name, fx, fy)
    cache_path = SURFACES_DIR / f"{key}.joblib"
    if cache_path.exists():
        return joblib.load(cache_path)

    results_path = OUTPUT_DIR / "results" / f"{model_name}.json"
    best_params = {}
    if results_path.exists():
        stored = json.loads(results_path.read_text())
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

    shadow = build_model(model_name, _ParamsAdapter(best_params)) if best_params else build_model(
        model_name, _ParamsAdapter({})
    )
    shadow.fit(X2, y)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid_proba = shadow.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    payload = {"xx": xx, "yy": yy, "proba": grid_proba, "points": X2, "labels": y, "fx": fx, "fy": fy}
    joblib.dump(payload, cache_path)
    return payload


def run():
    """Pipeline-stage entry point: pre-warm the cache for the default
    top-2 fraud-correlated feature pair across every trained model, so the
    Streamlit app's first load is instant instead of fitting shadow models
    on demand."""
    import json as _json
    results_dir = OUTPUT_DIR / "results"
    if not results_dir.exists():
        print("[decision_surfaces] no trained models yet — skipping cache warm.")
        return
    from config import TIERS
    shadow_capable = set(TIERS["tier1_single"]) | set(TIERS["tier2_ensemble"])
    model_names = [p.stem for p in results_dir.glob("*.json") if p.stem in shadow_capable]
    for m in model_names:
        try:
            get_or_build_surface(m, "V14", "V17")
        except Exception as e:
            print(f"[decision_surfaces] skip {m}: {e}")
    print(f"[decision_surfaces] warmed cache for {len(model_names)} models on (V14, V17).")


if __name__ == "__main__":
    run()

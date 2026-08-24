"""
Stage: shap_explain
Loads the best TREE-BASED model for the CURRENT RUN_VERSION (per
model_comparison_<version>.csv), computes SHAP values on a stratified
subsample of the test set (full 30 features), saves a versioned beeswarm
summary plot + raw shap values for the app/notebook.

Versioned: different experiment versions can promote a different "best
tree-based model" (e.g. v2's regularization might favor LightGBM over an
XGBoost that was overfitting in v1), so every output here is namespaced by
RUN_VERSION rather than overwritten on each run:
  output/shap_summary_<version>.png
  output/shap_values_<version>.npy
  output/shap_X_sample_<version>.parquet
"""
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, TARGET_COL, RANDOM_STATE, RUN_VERSION, COMPARISON_CSV_PATH, model_path

SHAP_SUBSAMPLE = 2000

# TreeExplainer is exact and fast for these; everything else (KNN, Logistic
# Regression, and the Tier-3 meta-ensembles which wrap multiple sub-models)
# would need KernelExplainer, which is a model-agnostic *sampling*
# approximation that calls predict_proba O(background x samples x features)
# times -- too slow to run on every CI/pipeline pass. So we explain the
# best *tree-based* model instead of blindly the top-ranked model in the
# comparison table -- still one of the top performers, and exact rather
# than approximate.
TREE_EXPLAINABLE = {"decision_tree", "random_forest", "bagging", "adaboost", "xgboost", "lightgbm"}


def run():
    if not COMPARISON_CSV_PATH.exists():
        print(f"[shap_explain] {COMPARISON_CSV_PATH.name} not found -- run the train "
              f"stage for run={RUN_VERSION} first. Skipping.")
        return

    comparison = pd.read_csv(COMPARISON_CSV_PATH)
    tree_rows = comparison[comparison["model"].isin(TREE_EXPLAINABLE)]
    if tree_rows.empty:
        print(f"[shap_explain] no tree-based models trained yet under run={RUN_VERSION} -- skipping.")
        return
    best_model_name = tree_rows.iloc[0]["model"]
    print(f"[shap_explain] best tree-based model = {best_model_name} (run={RUN_VERSION}) "
          f"(overall best was {comparison.iloc[0]['model']}; using TreeExplainer for speed/exactness)")

    model = joblib.load(model_path(best_model_name))
    test_df = pd.read_parquet(OUTPUT_DIR / "test.parquet")
    sub = test_df.groupby(TARGET_COL, group_keys=False).apply(
        lambda g: g.sample(min(len(g), SHAP_SUBSAMPLE // 2), random_state=RANDOM_STATE)
    )
    X = sub.drop(columns=[TARGET_COL])

    # Unwrap pipeline down to the raw estimator for tree explainers where possible
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):  # binary clf sometimes returns [class0, class1]
            shap_values = shap_values[1]
    except Exception as e:
        print(f"[shap_explain] TreeExplainer failed ({e}); falling back to a small-sample KernelExplainer")
        X = X.sample(min(50, len(X)), random_state=RANDOM_STATE)
        background = shap.sample(X, min(20, len(X)), random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(estimator.predict_proba, background)
        shap_values = explainer.shap_values(X, nsamples=100)[1]

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    summary_path = OUTPUT_DIR / f"shap_summary_{RUN_VERSION}.png"
    plt.savefig(summary_path, dpi=150)
    plt.close()

    np.save(OUTPUT_DIR / f"shap_values_{RUN_VERSION}.npy", shap_values)
    X.to_parquet(OUTPUT_DIR / f"shap_X_sample_{RUN_VERSION}.parquet", index=False)
    print(f"[shap_explain] saved {summary_path.name} + shap_values_{RUN_VERSION}.npy "
          f"for {best_model_name}")


if __name__ == "__main__":
    run()
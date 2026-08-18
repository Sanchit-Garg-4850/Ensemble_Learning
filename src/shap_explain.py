"""
Stage: shap_explain
Loads the best model per output/model_comparison.csv, computes SHAP values
on a stratified subsample of the test set (full 30 features), saves a
beeswarm summary plot + raw shap values for the app/notebook.
"""
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, MODELS_DIR, TARGET_COL, RANDOM_STATE

SHAP_SUBSAMPLE = 2000

# TreeExplainer is exact and fast for these; everything else (KNN, Logistic
# Regression, and the Tier-3 meta-ensembles which wrap multiple sub-models)
# would need KernelExplainer, which is a model-agnostic *sampling*
# approximation that calls predict_proba O(background x samples x features)
# times — with a VotingClassifier/StackingClassifier wrapping several
# models, this becomes far too slow to run on every CI/pipeline pass. So we
# explain the best *tree-based* model instead of blindly the top-ranked
# model in the comparison table — still one of the top performers, and the
# explanation is exact rather than approximate.
TREE_EXPLAINABLE = {"decision_tree", "random_forest", "bagging", "adaboost", "xgboost", "lightgbm"}


def run():
    comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    tree_rows = comparison[comparison["model"].isin(TREE_EXPLAINABLE)]
    if tree_rows.empty:
        print("[shap_explain] no tree-based models trained yet — skipping.")
        return
    best_model_name = tree_rows.iloc[0]["model"]
    print(f"[shap_explain] best tree-based model = {best_model_name} "
          f"(overall best was {comparison.iloc[0]['model']}; using TreeExplainer for speed/exactness)")

    model = joblib.load(MODELS_DIR / f"{best_model_name}.joblib")
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
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150)
    plt.close()

    np.save(OUTPUT_DIR / "shap_values.npy", shap_values)
    X.to_parquet(OUTPUT_DIR / "shap_X_sample.parquet", index=False)
    print(f"[shap_explain] saved shap_summary.png + shap_values.npy for {best_model_name}")


if __name__ == "__main__":
    run()

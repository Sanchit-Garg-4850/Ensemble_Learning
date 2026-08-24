"""
Streamlit app — reads cached artifacts from output/.
Run: streamlit run app/streamlit_app.py

The comparison and threshold tabs use cached artifacts only.
The decision-surface tab optionally fits a small illustrative shadow model
when output/train.parquet is available.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from decision_surfaces import get_or_build_surface
from config import TIERS, OUTPUT_DIR as _OUTPUT_DIR, RUN_VERSION

OUTPUT_DIR = _OUTPUT_DIR
SHADOW_CAPABLE_MODELS = set(TIERS["tier1_single"]) | set(TIERS["tier2_ensemble"])


def artifact_path(relative_path):
    """Resolve artifact paths regardless of Windows/Linux path separators."""
    if not relative_path:
        return None
    return ROOT / Path(str(relative_path).replace("\\", "/"))


def discover_run_versions():
    """Find every run version that has a manifest, e.g. ['v1', 'v2'], newest last."""
    versions = sorted(
        p.stem.replace("app_manifest_", "")
        for p in OUTPUT_DIR.glob("app_manifest_*.json")
    )
    return versions


st.set_page_config(page_title="Fraud Detection — Ensemble Showcase", layout="wide")

available_versions = discover_run_versions()
if not available_versions:
    st.error(
        "No `output/app_manifest_<version>.json` found for any run. "
        "Run the pipeline first: `make pipeline` (or your training entry point)."
    )
    st.stop()

default_idx = (
    available_versions.index(RUN_VERSION)
    if RUN_VERSION in available_versions
    else len(available_versions) - 1
)
selected_version = st.sidebar.selectbox(
    "Experiment version",
    available_versions,
    index=default_idx,
    help="Each version is a separate tuning run — e.g. v1 = original search space, "
         "v2 = regularized search space. Switch here to compare before/after.",
)


@st.cache_data
def load_manifest(version):
    path = OUTPUT_DIR / f"app_manifest_{version}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_comparison(version):
    return pd.read_csv(OUTPUT_DIR / f"model_comparison_{version}.csv")


@st.cache_data
def load_test_set():
    test_path = OUTPUT_DIR / "test.parquet"
    if not test_path.exists():
        return None
    return pd.read_parquet(test_path)


manifest = load_manifest(selected_version)
if manifest is None:
    st.error(f"No manifest found for version '{selected_version}'.")
    st.stop()

comparison = load_comparison(selected_version)
test_df = load_test_set()
if test_df is None:
    st.error(
        f"`output/test.parquet` is not present in this deployment, so the app "
        f"can't compute confusion matrices or threshold metrics for version "
        f"'{selected_version}'. Every tab depends on this file — make sure it "
        f"was committed alongside the model artifacts."
    )
    st.stop()

st.title("Fraud Detection — Ensemble Learning Showcase")
st.caption(
    f"Viewing experiment **{selected_version}**. Weak learners → bagging/boosting → stacking. "
    f"All metrics come from a held-out test set; V1–V28 are anonymized PCA components "
    f"from the original bank data."
)

tab_compare, tab_surface, tab_threshold = st.tabs(
    ["Model comparison", "Decision surface", "Threshold tuning"]
)

with tab_compare:
    st.subheader("Tuned model comparison (test set)")
    st.dataframe(comparison.style.format({c: "{:.4f}" for c in comparison.columns if c != "model" and c != "tier"}))
    st.bar_chart(comparison.set_index("model")[["pr_auc", "roc_auc", "f1"]])

    if "pr_auc_gap" in comparison.columns:
        st.subheader("Train vs. test PR-AUC (overfitting check)")
        st.caption(
            "Gap = train PR-AUC − test PR-AUC. Near-zero or negative is healthy; "
            "a large positive gap means the model is fitting noise in training "
            "data that doesn't generalize to the held-out test set."
        )
        gap_df = comparison[["model", "train_pr_auc", "pr_auc", "pr_auc_gap"]].rename(
            columns={"pr_auc": "test_pr_auc"}
        )
        st.dataframe(gap_df.style.format({c: "{:.4f}" for c in ["train_pr_auc", "test_pr_auc", "pr_auc_gap"]}))
        st.bar_chart(gap_df.set_index("model")[["train_pr_auc", "test_pr_auc"]])

    oob_rows = []
    for name, info in manifest["models"].items():
        result_path = ROOT / Path(info["result"].replace("\\", "/"))
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if result.get("oob_score") is not None:
            oob_rows.append({"model": name, "oob_score": result["oob_score"]})
    if oob_rows:
        st.subheader("OOB score (bagging-based models)")
        st.dataframe(pd.DataFrame(oob_rows))

with tab_surface:
    st.subheader("Live decision surface")

    train_path = OUTPUT_DIR / "train.parquet"

    if not train_path.exists():
        st.info(
            "The live decision-surface tab is disabled in this deployment because "
            "`output/train.parquet` is not included. The model comparison and "
            "threshold tuning tabs remain fully available from cached test-set artifacts."
        )
    else:
        surface_models = [m for m in manifest["models"] if m in SHADOW_CAPABLE_MODELS]

        if not surface_models:
            st.warning("No shadow-capable models are available for the decision surface.")
        else:
            col1, col2, col3 = st.columns(3)

            model_name = col1.selectbox(
                "Model",
                surface_models,
                help=(
                    "Voting/Stacking aren't shown here — a shadow 2D surface "
                    "needs one tunable model, not an ensemble-of-models."
                ),
            )

            feature_cols = [c for c in test_df.columns if c != "Class"]

            fx = col2.selectbox(
                "Feature X",
                feature_cols,
                index=feature_cols.index("V14") if "V14" in feature_cols else 0,
            )

            fy = col3.selectbox(
                "Feature Y",
                feature_cols,
                index=feature_cols.index("V17") if "V17" in feature_cols else 1,
            )

            st.caption(
                "Shadow model: same model class + tuned hyperparams, refit on just "
                "these 2 features on a stratified subsample — illustrative, not "
                "the production 30-feature model. Boundary shown is the model's "
                "hard predicted class (0/1), not a probability gradient."
            )

            try:
                surface = get_or_build_surface(model_name, fx, fy, version=selected_version)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            fig = go.Figure(
                data=go.Contour(
                    x=surface["xx"][0],
                    y=surface["yy"][:, 0],
                    z=surface["pred"],
                    colorscale=[[0, "steelblue"], [1, "crimson"]],
                    opacity=0.5,
                    showscale=False,
                    contours=dict(start=0, end=1, size=1),
                )
            )

            pts = surface["points"]
            labels = surface["labels"]

            fig.add_trace(
                go.Scatter(
                    x=pts[labels == 0, 0],
                    y=pts[labels == 0, 1],
                    mode="markers",
                    marker=dict(size=4, color="steelblue"),
                    name="normal",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pts[labels == 1, 0],
                    y=pts[labels == 1, 1],
                    mode="markers",
                    marker=dict(size=6, color="crimson", symbol="x"),
                    name="fraud",
                )
            )

            fig.update_layout(
                xaxis_title=fx,
                yaxis_title=fy,
                height=550,
            )

            st.plotly_chart(fig, use_container_width=True)

with tab_threshold:
    st.subheader("Threshold tuning (instant — recomputed from cached probabilities)")
    model_name_t = st.selectbox("Model", list(manifest["models"].keys()), key="thresh_model")
    proba_path = manifest["models"][model_name_t]["test_proba"]
    if proba_path is None:
        st.warning("No cached probabilities for this model yet.")
    else:
        proba = np.load(ROOT / Path(proba_path.replace("\\", "/")))
        y_true = test_df["Class"].values
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
        pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, pred)
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{precision_score(y_true, pred, zero_division=0):.3f}")
        c2.metric("Recall", f"{recall_score(y_true, pred, zero_division=0):.3f}")
        c3.metric("F1", f"{f1_score(y_true, pred, zero_division=0):.3f}")
        st.write("Confusion matrix (rows=actual, cols=predicted)")
        st.dataframe(pd.DataFrame(cm, index=["normal", "fraud"], columns=["pred_normal", "pred_fraud"]))
        
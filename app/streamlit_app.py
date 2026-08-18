"""
Streamlit app — reads only output/app_manifest.json + cached artifacts.
Run: streamlit run app/streamlit_app.py
Everything here is offline-cached; nothing retrains on toggle except the
explicitly-labeled "quick estimate" fallback for novel feature pairs.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from decision_surfaces import get_or_build_surface  # noqa: E402
from config import TIERS  # noqa: E402

OUTPUT_DIR = ROOT / "output"
SHADOW_CAPABLE_MODELS = set(TIERS["tier1_single"]) | set(TIERS["tier2_ensemble"])

st.set_page_config(page_title="Fraud Detection — Ensemble Showcase", layout="wide")


@st.cache_data
def load_manifest():
    path = OUTPUT_DIR / "app_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_comparison():
    return pd.read_csv(OUTPUT_DIR / "model_comparison.csv")


@st.cache_data
def load_test_set():
    return pd.read_parquet(OUTPUT_DIR / "test.parquet")


manifest = load_manifest()
if manifest is None:
    st.error("No `output/app_manifest.json` found. Run the pipeline first: `make pipeline`.")
    st.stop()

comparison = load_comparison()
test_df = load_test_set()

st.title("Fraud Detection — Ensemble Learning Showcase")
st.caption("Weak learners → bagging/boosting → stacking. All metrics come from a held-out test set; "
           "V1–V28 are anonymized PCA components from the original bank data.")

tab_compare, tab_surface, tab_threshold = st.tabs(
    ["Model comparison", "Decision surface", "Threshold tuning"]
)

with tab_compare:
    st.subheader("Tuned model comparison (test set)")
    st.dataframe(comparison.style.format({c: "{:.4f}" for c in comparison.columns if c != "model" and c != "tier"}))
    st.bar_chart(comparison.set_index("model")[["pr_auc", "roc_auc", "f1"]])

    oob_rows = []
    for name, info in manifest["models"].items():
        result_path = ROOT / info["result"]
        result = json.loads(result_path.read_text())
        if result.get("oob_score") is not None:
            oob_rows.append({"model": name, "oob_score": result["oob_score"]})
    if oob_rows:
        st.subheader("OOB score (bagging-based models)")
        st.dataframe(pd.DataFrame(oob_rows))

with tab_surface:
    st.subheader("Live decision surface")
    surface_models = [m for m in manifest["models"] if m in SHADOW_CAPABLE_MODELS]
    col1, col2, col3 = st.columns(3)
    model_name = col1.selectbox("Model", surface_models,
                                 help="Voting/Stacking aren't shown here — a shadow 2D surface "
                                      "needs one tunable model, not an ensemble-of-models.")
    feature_cols = [c for c in test_df.columns if c != "Class"]
    fx = col2.selectbox("Feature X", feature_cols, index=feature_cols.index("V14") if "V14" in feature_cols else 0)
    fy = col3.selectbox("Feature Y", feature_cols, index=feature_cols.index("V17") if "V17" in feature_cols else 1)

    st.caption("Shadow model: same model class + tuned hyperparams, refit on just these 2 features "
               "on a stratified subsample — illustrative, not the production 30-feature model.")

    surface = get_or_build_surface(model_name, fx, fy)
    fig = go.Figure(data=go.Contour(
        x=surface["xx"][0], y=surface["yy"][:, 0], z=surface["proba"],
        colorscale="RdBu_r", opacity=0.7, showscale=True,
    ))
    pts = surface["points"]
    labels = surface["labels"]
    fig.add_trace(go.Scatter(
        x=pts[labels == 0, 0], y=pts[labels == 0, 1], mode="markers",
        marker=dict(size=4, color="steelblue"), name="normal",
    ))
    fig.add_trace(go.Scatter(
        x=pts[labels == 1, 0], y=pts[labels == 1, 1], mode="markers",
        marker=dict(size=6, color="crimson", symbol="x"), name="fraud",
    ))
    fig.update_layout(xaxis_title=fx, yaxis_title=fy, height=550)
    st.plotly_chart(fig, use_container_width=True)

with tab_threshold:
    st.subheader("Threshold tuning (instant — recomputed from cached probabilities)")
    model_name_t = st.selectbox("Model", list(manifest["models"].keys()), key="thresh_model")
    proba_path = manifest["models"][model_name_t]["test_proba"]
    if proba_path is None:
        st.warning("No cached probabilities for this model yet.")
    else:
        proba = np.load(ROOT / proba_path)
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

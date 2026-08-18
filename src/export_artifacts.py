"""
Stage: export_artifacts
Writes output/app_manifest.json — the single file the Streamlit app reads
to discover which models/results/cached probas exist, so the app never
needs to know about pipeline internals or re-derive paths.
"""
import json

from config import OUTPUT_DIR, MODELS_DIR, TIERS


def run():
    manifest = {"models": {}, "comparison_csv": "model_comparison.csv", "shap_summary": "shap_summary.png"}
    for tier, names in TIERS.items():
        for name in names:
            result_path = OUTPUT_DIR / "results" / f"{name}.json"
            model_path = MODELS_DIR / f"{name}.joblib"
            proba_path = OUTPUT_DIR / "results" / f"{name}_test_proba.npy"
            if result_path.exists() and model_path.exists():
                manifest["models"][name] = {
                    "tier": tier,
                    "result": str(result_path.relative_to(OUTPUT_DIR.parent)),
                    "model": str(model_path.relative_to(OUTPUT_DIR.parent)),
                    "test_proba": str(proba_path.relative_to(OUTPUT_DIR.parent)) if proba_path.exists() else None,
                }
    (OUTPUT_DIR / "app_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[export_artifacts] wrote manifest with {len(manifest['models'])} models")


if __name__ == "__main__":
    run()

"""
Stage: export_artifacts
Writes output/app_manifest_<RUN_VERSION>.json -- the file the Streamlit app
reads to discover which models/results/cached probas exist for a given
experiment version, so the app never needs to know about pipeline internals
or re-derive paths.

Versioned: uses config's results_path()/model_path()/proba_path() helpers
(same ones train.py and decision_surfaces.py use) so this stage always
targets the CURRENT RUN_VERSION and never overwrites another version's
manifest. Re-runnable standalone via:
    python src/run_pipeline.py --only export_artifacts
if train.py already ran and you just need to regenerate the manifest
(e.g. after a later shap_explain run added a shap_summary file).
"""
import json

from config import (
    ROOT, OUTPUT_DIR, TIERS, RUN_VERSION,
    results_path, model_path, proba_path,
    APP_MANIFEST_PATH, COMPARISON_CSV_PATH,
)


def _rel(path):
    """Path relative to project ROOT, forward-slash string, for the manifest."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


def run():
    shap_summary_path = OUTPUT_DIR / f"shap_summary_{RUN_VERSION}.png"

    manifest = {
        "run_version": RUN_VERSION,
        "models": {},
        "comparison_csv": _rel(COMPARISON_CSV_PATH) if COMPARISON_CSV_PATH.exists() else None,
        "shap_summary": _rel(shap_summary_path) if shap_summary_path.exists() else None,
    }

    for tier, names in TIERS.items():
        for name in names:
            rp = results_path(name)
            mp = model_path(name)
            pp = proba_path(name)
            if rp.exists() and mp.exists():
                manifest["models"][name] = {
                    "tier": tier,
                    "result": _rel(rp),
                    "model": _rel(mp),
                    "test_proba": _rel(pp) if pp.exists() else None,
                }

    APP_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"[export_artifacts] wrote {APP_MANIFEST_PATH.name} with "
          f"{len(manifest['models'])} models (run={RUN_VERSION})")


if __name__ == "__main__":
    run()
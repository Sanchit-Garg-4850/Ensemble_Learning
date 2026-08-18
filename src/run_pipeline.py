"""
Pipeline orchestrator.

Runs the full stage sequence with checkpointing (skips a stage if its
output marker already exists, unless --force or FRAUD_FORCE=1), logs
timing per stage to output/pipeline_state.json, and exits non-zero on the
first failure so CI can fail fast.

Usage:
    python src/run_pipeline.py                 # run all stages, resume from checkpoints
    python src/run_pipeline.py --force          # ignore checkpoints, rerun everything
    python src/run_pipeline.py --from train      # skip straight to a stage (assumes prior stages already ran)
    python src/run_pipeline.py --only shap_explain
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR  # noqa: E402

STATE_PATH = OUTPUT_DIR / "pipeline_state.json"

# stage_name -> (module, marker_file_relative_to_OUTPUT_DIR)
STAGES = [
    ("data_quality", "data_quality_report.json"),
    ("preprocessing", "train.parquet"),
    ("train", "model_comparison.csv"),
    ("decision_surfaces", None),   # always cheap to (re)build cache on demand
    ("shap_explain", "shap_summary.png"),
    ("export_artifacts", "app_manifest.json"),
]


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run_stage(name: str, force: bool, state: dict):
    marker = dict(STAGES)[name]
    marker_path = OUTPUT_DIR / marker if marker else None

    if not force and marker_path and marker_path.exists():
        print(f"[orchestrator] SKIP {name} (checkpoint found: {marker})")
        return

    print(f"[orchestrator] RUN {name} ...")
    t0 = time.time()
    module = importlib.import_module(name)
    try:
        module.run()
    except Exception as e:
        state[name] = {"status": "failed", "error": str(e)}
        _save_state(state)
        print(f"[orchestrator] FAILED at stage '{name}': {e}")
        raise
    elapsed = round(time.time() - t0, 1)
    state[name] = {"status": "success", "seconds": elapsed}
    _save_state(state)
    print(f"[orchestrator] DONE {name} in {elapsed}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="ignore checkpoints, rerun all stages")
    parser.add_argument("--from", dest="from_stage", default=None, help="start pipeline at this stage")
    parser.add_argument("--only", default=None, help="run a single stage only")
    args = parser.parse_args()

    state = _load_state()
    names = [s for s, _ in STAGES]

    if args.only:
        to_run = [args.only]
    elif args.from_stage:
        idx = names.index(args.from_stage)
        to_run = names[idx:]
    else:
        to_run = names

    print(f"[orchestrator] plan: {to_run}")
    for name in to_run:
        run_stage(name, args.force, state)
    print("[orchestrator] pipeline complete.")


if __name__ == "__main__":
    main()

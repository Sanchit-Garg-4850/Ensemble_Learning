"""
Pipeline orchestrator.

Runs the full stage sequence with checkpointing (skips a stage if its
output marker already exists, unless --force or FRAUD_FORCE=1), logs
timing per stage to output/pipeline_state_<RUN_VERSION>.json, and exits
non-zero on the first failure so CI can fail fast.

Versioning: checkpoint markers for RUN_VERSION-dependent stages (train,
shap_explain, export_artifacts) are namespaced by RUN_VERSION, so re-running
under a new FRAUD_RUN_VERSION won't skip a stage just because an OLDER
version's output already exists. data_quality and preprocessing markers stay
unversioned on purpose -- config.py's PROCESSED_DATA_PATH/train.parquet/
test.parquet are intentionally shared across experiments (regularization
changes don't touch the data split/scaling stage).

Usage:
    set FRAUD_RUN_VERSION=v2
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
from config import OUTPUT_DIR, RUN_VERSION  # noqa: E402

STATE_PATH = OUTPUT_DIR / f"pipeline_state_{RUN_VERSION}.json"

# stage_name -> marker_file_relative_to_OUTPUT_DIR (None = always (re)run)
STAGES = [
    ("data_quality", "data_quality_report.json"),          # unversioned -- shared raw-data check
    ("preprocessing", "train.parquet"),                    # unversioned -- shared across experiments
    ("train", f"model_comparison_{RUN_VERSION}.csv"),
    ("decision_surfaces", None),                            # always cheap to (re)build cache on demand
    ("shap_explain", f"shap_summary_{RUN_VERSION}.png"),
    ("export_artifacts", f"app_manifest_{RUN_VERSION}.json"),
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
        print(f"[orchestrator] SKIP {name} (checkpoint found: {marker}, run={RUN_VERSION})")
        return

    print(f"[orchestrator] RUN {name} (run={RUN_VERSION}) ...")
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

    print(f"[orchestrator] run_version={RUN_VERSION} plan: {to_run}")
    for name in to_run:
        run_stage(name, args.force, state)
    print(f"[orchestrator] pipeline complete (run={RUN_VERSION}).")


if __name__ == "__main__":
    main()

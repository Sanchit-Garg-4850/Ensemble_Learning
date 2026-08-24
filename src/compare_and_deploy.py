"""
CD gate: compares the best PR-AUC produced by this branch's training run
against the best PR-AUC currently deployed on main (tracked in best_score.json
at the repo root). If the candidate is better, updates best_score.json and
merges this branch into main -- which Streamlit Community Cloud is watching,
so pushing to main triggers an automatic redeploy. If not better, does
nothing and exits cleanly (no deploy happens).

Expects to run from a full checkout (fetch-depth: 0) with git user configured
and a token with push access to origin.

MLflow model registry:
After a successful deploy, promotes the winning model to the "Production"
stage in the MLflow Model Registry. Tracking URI/experiment come from
config.py (config.MLFLOW_TRACKING_URI / config.MLFLOW_EXPERIMENT_NAME), same
single source of truth train.py uses -- so as long as MLFLOW_TRACKING_URI
points at a SHARED server (e.g. a DagsHub repo's MLflow endpoint, set as a
GitHub Actions secret) rather than the local-only default, this step works
identically whether run locally or from CI. If config.py can't be imported,
or the tracking store isn't reachable, or no matching run is found, this
step prints why and skips -- it never blocks the actual deploy above.
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BEST_SCORE_FILE = ROOT / "best_score.json"
REGISTERED_MODEL_NAME = "fraud-detection-best-model"


def run_git(*args, check=True):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"[cd] git {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def find_candidate_best():
    files = sorted(glob.glob(str(ROOT / "output" / "model_comparison_v*.csv")))
    if not files:
        print("[cd] no output/model_comparison_v*.csv found on this branch -- nothing to evaluate.")
        sys.exit(1)
    latest = files[-1]
    run_version = Path(latest).stem.replace("model_comparison_", "")
    df = pd.read_csv(latest)
    if df.empty or "pr_auc" not in df.columns:
        print(f"[cd] {latest} is empty or missing pr_auc column.")
        sys.exit(1)
    best_row = df.sort_values("pr_auc", ascending=False).iloc[0]
    print(f"[cd] candidate best: {best_row['model']} pr_auc={best_row['pr_auc']:.4f} "
          f"(run_version={run_version}, from {latest})")
    return {
        "model": str(best_row["model"]),
        "pr_auc": float(best_row["pr_auc"]),
        "run_version": run_version,
        "source_file": latest,
    }


def fetch_main_best():
    run_git("fetch", "origin", "main", check=False)
    show = subprocess.run(
        ["git", "show", "origin/main:best_score.json"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if show.returncode != 0:
        print("[cd] no best_score.json on main yet -- treating current best as 0.0.")
        return {"model": None, "pr_auc": 0.0}
    data = json.loads(show.stdout)
    print(f"[cd] main's current best: {data.get('model')} pr_auc={data.get('pr_auc', 0.0):.4f}")
    return data


def deploy(candidate_best):
    current_branch = run_git("rev-parse", "--abbrev-ref", "HEAD")
    BEST_SCORE_FILE.write_text(json.dumps(candidate_best, indent=2))
    run_git("add", "best_score.json")
    commit = subprocess.run(
        ["git", "commit", "-m",
         f"New best model: {candidate_best['model']} (PR-AUC {candidate_best['pr_auc']:.4f})"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(commit.stdout or commit.stderr)

    run_git("checkout", "main")
    run_git("pull", "origin", "main")
    merge = subprocess.run(
        ["git", "merge", "--no-ff", current_branch, "-m",
         f"Deploy: {candidate_best['model']} PR-AUC {candidate_best['pr_auc']:.4f}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if merge.returncode != 0:
        print(f"[cd] merge into main failed:\n{merge.stdout}\n{merge.stderr}")
        sys.exit(1)
    run_git("push", "origin", "main")
    print("[cd] deployed -- pushed to main, Streamlit Cloud will auto-redeploy.")


def register_best_model(candidate_best):
    """Best-effort promotion of the winning model to the MLflow Model
    Registry's 'Production' stage. Never raises -- any failure just means
    this step is skipped; the actual deploy above already happened."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("[cd] mlflow not installed in this environment -- skipping model registry step.")
        return

    try:
        from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
    except ImportError:
        # Fallback if src/ isn't importable from wherever this is invoked --
        # still respects a shared MLFLOW_TRACKING_URI if the env var is set.
        MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
        MLFLOW_EXPERIMENT_NAME = os.environ.get("FRAUD_MLFLOW_EXPERIMENT", "fraud-detection-ensemble")
        if not MLFLOW_TRACKING_URI:
            print("[cd] config.py not importable and no MLFLOW_TRACKING_URI set -- skipping registry step.")
            return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    local_only = MLFLOW_TRACKING_URI.startswith("file:")
    if local_only:
        print(f"[cd] MLFLOW_TRACKING_URI ({MLFLOW_TRACKING_URI}) is a local file store -- "
              f"only reachable if this script is running on the same machine that trained "
              f"the model. On CI this will fail to find the run. Set MLFLOW_TRACKING_URI to "
              f"a shared server (e.g. DagsHub) to make this work from GitHub Actions too.")

    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    except Exception as e:
        print(f"[cd] could not reach mlflow tracking store ({MLFLOW_TRACKING_URI}): {e} "
              f"-- skipping registry step.")
        return

    if experiment is None:
        print(f"[cd] mlflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found at "
              f"{MLFLOW_TRACKING_URI} -- skipping registry step.")
        return

    run_name = f"{candidate_best['model']}_{candidate_best['run_version']}"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print(f"[cd] no mlflow run named '{run_name}' found in experiment "
              f"'{MLFLOW_EXPERIMENT_NAME}' -- skipping registry step.")
        return

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[cd] registered '{REGISTERED_MODEL_NAME}' v{mv.version} "
          f"({candidate_best['model']}, run={candidate_best['run_version']}) -> Production")


def main():
    candidate_best = find_candidate_best()
    main_best = fetch_main_best()

    if candidate_best["pr_auc"] > main_best.get("pr_auc", 0.0):
        print(f"[cd] {candidate_best['pr_auc']:.4f} > {main_best.get('pr_auc', 0.0):.4f} -- deploying.")
        deploy(candidate_best)
        register_best_model(candidate_best)
    else:
        print(
            f"[cd] {candidate_best['pr_auc']:.4f} did not beat current best "
            f"{main_best.get('pr_auc', 0.0):.4f} -- skipping deploy."
        )


if __name__ == "__main__":
    main()

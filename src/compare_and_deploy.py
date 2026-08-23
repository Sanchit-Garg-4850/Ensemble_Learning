"""
CD gate: compares the best PR-AUC produced by this branch's training run
against the best PR-AUC currently deployed on main (tracked in best_score.json
at the repo root). If the candidate is better, updates best_score.json and
merges this branch into main -- which Streamlit Community Cloud is watching,
so pushing to main triggers an automatic redeploy. If not better, does
nothing and exits cleanly (no deploy happens).

Expects to run from a full checkout (fetch-depth: 0) with git user configured
and a token with push access to origin.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BEST_SCORE_FILE = ROOT / "best_score.json"


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
    df = pd.read_csv(latest)
    if df.empty or "pr_auc" not in df.columns:
        print(f"[cd] {latest} is empty or missing pr_auc column.")
        sys.exit(1)
    best_row = df.sort_values("pr_auc", ascending=False).iloc[0]
    print(f"[cd] candidate best: {best_row['model']} pr_auc={best_row['pr_auc']:.4f} (from {latest})")
    return {"model": str(best_row["model"]), "pr_auc": float(best_row["pr_auc"]), "source_file": latest}


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
    print(f"[cd] deployed -- pushed to main, Streamlit Cloud will auto-redeploy.")


def main():
    candidate_best = find_candidate_best()
    main_best = fetch_main_best()

    if candidate_best["pr_auc"] > main_best.get("pr_auc", 0.0):
        print(f"[cd] {candidate_best['pr_auc']:.4f} > {main_best.get('pr_auc', 0.0):.4f} -- deploying.")
        deploy(candidate_best)
    else:
        print(
            f"[cd] {candidate_best['pr_auc']:.4f} did not beat current best "
            f"{main_best.get('pr_auc', 0.0):.4f} -- skipping deploy."
        )


if __name__ == "__main__":
    main()
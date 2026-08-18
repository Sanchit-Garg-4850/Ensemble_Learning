# Fraud Detection — Ensemble Learning Showcase

Validated end-to-end (all 6 pipeline stages green, Makefile + orchestrator +
CI all tested) against the real `creditcard.csv` you uploaded, plus a synthetic
420-row fixture used by CI. Full run instructions below.

## 0. Requirements — tools & API keys

**No API keys are needed anywhere in this project.** Every library
(scikit-learn, imbalanced-learn, XGBoost, LightGBM, Optuna, SHAP, Streamlit)
is open-source and runs 100% locally/offline. The only "account" involved is
optional: a free **Streamlit Community Cloud** account if you want to deploy
the app publicly later (Section 6), and that's just GitHub OAuth, not an API key.

**Tools to install locally:**
- Python 3.11 (3.12 also confirmed working)
- `make` (ships with macOS/Linux; on Windows use WSL or run the commands
  inside the Makefile manually)
- Git + a GitHub account (for the CI/CD part, Section 7)

That's it. `pip install -r requirements.txt` pulls everything else.

## 1. Repo layout

```
fraud-detection-project/
├── data/creditcard.csv        # put the Kaggle CSV here (gitignored — 150MB)
├── src/
│   ├── config.py               # single source of truth: paths, constants, run params
│   ├── data_quality.py         # Stage 1 — schema/missing/dup/outlier/leakage checks
│   ├── preprocessing.py        # Stage 2 — log1p, scaling, stratified split
│   ├── model_configs.py        # model factories + Optuna search spaces (Tier 1/2/3)
│   ├── resamplers.py           # SMOTE/ADASYN/etc. factory, used inside CV folds only
│   ├── optuna_objective.py     # joint resampler+model+hyperparam objective (PR-AUC)
│   ├── train.py                # Stage 3 — runs Optuna per model, builds Tier 3 meta models
│   ├── decision_surfaces.py    # Stage 4 — shadow 2D models, cached by (model, fx, fy)
│   ├── shap_explain.py         # Stage 5 — SHAP on best tree-based model, 30 features
│   ├── export_artifacts.py     # Stage 6 — writes output/app_manifest.json for the app
│   └── run_pipeline.py         # ORCHESTRATOR — runs all stages, checkpointed, resumable
├── app/streamlit_app.py        # reads app_manifest.json only — no pipeline internals
├── notebooks/                  # EDA/PCA/tree-viz notebook (exploratory — plan Sections 5-6)
├── tests/
│   ├── fixtures/sample.csv     # 420-row synthetic fixture (same schema, fake values) — no real data needed for CI
│   └── test_pipeline_smoke.py  # runs all stages against the fixture — this is what CI runs
├── .github/workflows/ci.yml    # lint + smoke test on every push/PR
├── Makefile                    # local entry point (install/pipeline/test/app)
└── requirements.txt
```

## 2. How the pieces map to your plan doc

| Plan section | Where it lives |
|---|---|
| §4 Data quality | `src/data_quality.py` |
| §5-6 EDA / PCA / t-SNE | `notebooks/01_eda.ipynb` (exploratory, not part of the automated pipeline — see note below) |
| §7 Preprocessing | `src/preprocessing.py` |
| §8 Modeling tiers | `src/model_configs.py` (factories) + `src/train.py` (runner) |
| §9 Optuna joint search | `src/optuna_objective.py` + `src/train.py` |
| §10 Decision surfaces | `src/decision_surfaces.py` |
| §11 SHAP | `src/shap_explain.py` |
| §13 Streamlit app | `app/streamlit_app.py` |
| §15 orchestration | `src/run_pipeline.py` + `Makefile` |

**Why EDA is a notebook, not a pipeline stage:** §5-6 is exploratory/narrative
(countplots, clustermaps, PCA scatter, skew commentary) meant for you to read
and write interview talking points around — it doesn't produce an artifact
another stage consumes. Keeping it as a notebook that you run once and keep
for the README/resume story is the right shape; forcing it into `run.py`
would just make the orchestrator slower for no downstream benefit. I've left
`notebooks/01_eda.ipynb` as a stub — reuses `src.data_quality` and
`src.preprocessing` so it never duplicates logic.

## 3. Run it locally — step by step

```bash
# 1. Get the code + dataset in place
cd fraud-detection-project
cp /path/to/creditcard.csv data/          # the file you uploaded — already placed if you're using the delivered zip

# 2. Set up the environment
make install                # creates .venv, installs requirements.txt

# 3. Run the full pipeline (resumable — reruns only skip completed stages)
make pipeline                # ≈ several minutes for Tier 1, longer for Tier 2/3 depending on N_TRIALS

# Tune search depth via env var (default 50 trials/model, 5-fold CV):
FRAUD_N_TRIALS=100 FRAUD_CV_FOLDS=5 make pipeline

# Force a full rerun ignoring checkpoints (e.g. after editing model_configs.py):
make pipeline-force

# Run just one stage (useful while iterating):
cd src && python run_pipeline.py --only shap_explain
cd src && python run_pipeline.py --from decision_surfaces   # resume from a stage onward

# 4. Launch the app
make app                     # streamlit run app/streamlit_app.py

# 5. Run tests / lint (same as CI)
make test
make lint
```

**Checkpointing:** every stage writes a marker file to `output/` (e.g.
`model_comparison.csv` marks `train` as done) and `src/train.py` additionally
checkpoints *per model* to `output/results/<model>.json`, so if trial N of
`xgboost` crashes your laptop, rerunning `make pipeline` picks up at
`xgboost` — the already-finished `decision_tree`, `logistic_regression`, etc.
are not retrained. This mirrors the checkpoint-centralization pattern from
the SAP GRC pipeline. Optuna itself also persists trials to
`output/optuna_studies.db` (SQLite, `load_if_exists=True`), so even a
half-finished study for one model resumes from its last completed trial
rather than restarting that model's search from trial 0.

## 4. Validated results (subsample run, 4,492 rows: all 492 real fraud rows + 4,000 normal, 3 Optuna trials/model)

This was run against your actual uploaded data to confirm the pipeline
produces the story the plan wants — not synthetic numbers:

| model | tier | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|---|
| voting | tier3_meta | 0.9457 | 0.9849 | 0.905 |
| stacking | tier3_meta | 0.9455 | 0.9851 | 0.884 |
| random_forest | tier2_ensemble | 0.9424 | 0.9829 | 0.867 |
| lightgbm | tier2_ensemble | 0.9388 | 0.9819 | 0.901 |
| xgboost | tier2_ensemble | 0.9282 | 0.9736 | 0.905 |
| logistic_regression | tier1_single | 0.9241 | 0.9742 | 0.857 |
| decision_tree | tier1_single | 0.8381 | 0.9041 | 0.759 |

Exactly the narrative arc the plan calls for: single weak learner (decision
tree) at the bottom, boosted/bagged ensembles in the middle, meta-ensembles
(voting/stacking) on top. Run the full pipeline with real `N_TRIALS`
(50-100) on the full 285K rows for your actual resume numbers — this table
is a 3-trial smoke run to prove the pipeline works, not your final result.

## 5. Notes on things I deviated from / tightened in the plan

- **SMOTE/ADASYN `k_neighbors` vs small CV folds**: caught by the CI smoke
  test — with small folds, `k_neighbors` can exceed the minority class count
  in a fold and crashes `fit`. Fixed by treating that as a bad trial
  (`return 0.0`) inside the Optuna objective rather than crashing the whole
  study. Only matters at small scale, but it's the right general defense.
- **SHAP on Tier 3 (voting/stacking)**: `TreeExplainer` doesn't apply to a
  `VotingClassifier`/`StackingClassifier` wrapping multiple sub-models, and
  falling back to `KernelExplainer` (a sampling approximation) on an
  ensemble-of-ensembles is slow enough to be impractical as a routine
  pipeline stage. `shap_explain.py` explains the best **tree-based** model
  instead (exact, fast) — still one of your top performers per the table
  above, and you can call out in interviews that TreeExplainer's exactness
  was the reason, not a workaround.
- **`use_label_encoder` XGBoost warning**: harmless (deprecated param from an
  older API generation the model still accepts) — safe to ignore or strip if
  it bothers you in the notebook output.

## 6. Deploying the Streamlit app (optional, no API key)

1. Push this repo to GitHub (dataset stays out via `.gitignore` — see
   caveat below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, point it at `app/streamlit_app.py`.
3. **Caveat**: the app reads `output/` and `models/` artifacts, which are
   also gitignored (they're generated, not source). For a public deploy
   you'll need to either (a) commit a slimmed-down `output/`+`models/` for
   the app only (a few MB of joblib files + JSON — totally fine to commit,
   unlike the 150MB raw CSV), or (b) add a `Dockerfile`/CI step that runs
   `make pipeline` at deploy time. Given free-tier compute limits, (a) is
   simpler — run the pipeline locally once, then `git add -f output/
   models/` for the app-serving subset only.

## 7. CI/CD (`.github/workflows/ci.yml`)

**What runs on every push/PR (GitHub-hosted, free, no self-hosted runner
needed):**
1. `flake8` lint
2. `pytest tests/` — runs **all 4 pipeline stages end-to-end** against the
   420-row synthetic fixture (`tests/fixtures/sample.csv`, 2 Optuna trials,
   2 CV folds) in a few seconds. This isn't testing model *quality* — it's
   testing that the pipeline doesn't break: import errors, API drift across
   sklearn/imblearn/optuna versions, broken paths, checkpoint logic. This
   is exactly what caught the SMOTE `k_neighbors` bug above before it ever
   reached a real run.

**What does NOT run in the free CI job — full training on the real 285K-row
dataset with 50-100 Optuna trials per model.** That's multiple hours of
compute; GitHub's free runners have a 6-hour job cap and no data volume to
mount your Kaggle CSV into anyway. The workflow has a second job,
`full-training`, gated behind `workflow_dispatch` + `runs-on: self-hosted` —
wire it to a self-hosted runner (a spare machine, or a cloud VM you control)
with `data/creditcard.csv` present, and trigger it manually from the
Actions tab when you want a full retrain archived as a build artifact. This
is the standard shape for ML pipelines with heavy training: cheap
correctness checks on every commit, expensive real training as a deliberate,
manually-triggered job — not something you want auto-firing on every push.

## 8. Suggested `N_TRIALS` / timing guide (rough, single laptop, full 285K rows)

| N_TRIALS | Tier 1 (3 models) | Tier 2 (5 models) | Tier 3 (2 models) | Total |
|---|---|---|---|---|
| 10 (quick check) | ~2 min | ~10 min | ~2 min | ~15 min |
| 50 (plan default) | ~8 min | ~45 min | ~5 min | ~1 hr |
| 100 (resume-quality) | ~15 min | ~90 min | ~8 min | ~2 hrs |

KNN and XGBoost/LightGBM on the full 285K rows dominate the time; if you're
iterating on code rather than final numbers, keep `FRAUD_N_TRIALS` low and
only crank it up for your final resume-quality run.

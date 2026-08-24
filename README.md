# Fraud Detection — Ensemble Learning Showcase

A 3-tier ensemble learning pipeline for credit card fraud detection, built on
the [ULB Kaggle Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
(285K transactions, 0.17% fraud incidence — a 1:580 class imbalance).

The project benchmarks 10 models across three tiers — single learners,
bagging/boosting ensembles, and Voting/Stacking meta-ensembles — with a
joint Optuna search over both hyperparameters and imbalance-handling
strategy, full MLflow experiment tracking, a CI/CD pipeline that
auto-promotes the best model, and an interactive Streamlit app for
exploring the results.

**Live demo:** [Streamlit app](https://ensemblelearning-7twystgmwcga9x5xl9zhkb.streamlit.app/)
**Experiment tracking:** [DagsHub MLflow dashboard](https://dagshub.com/Sanchit-Garg-4850/fraud-detection-experiment-tracking.mlflow)

---

## Results

| Model | Tier | PR-AUC | ROC-AUC |
|---|---|---|---|
| **XGBoost (tuned)** | Bagging/Boosting | **0.8336** | **0.9684** |
| LightGBM | Bagging/Boosting | 0.8207 | 0.9661 |
| Voting (soft) | Meta-ensemble | 0.8116 | 0.9776 |
| Stacking | Meta-ensemble | 0.8069 | 0.9774 |
| Bagging | Bagging/Boosting | 0.7824 | 0.9672 |
| Random Forest | Bagging/Boosting | 0.7788 | 0.9773 |
| KNN | Single learner | 0.7776 | 0.9103 |
| Logistic Regression (baseline) | Single learner | 0.6688 | 0.9630 |
| AdaBoost | Bagging/Boosting | 0.6639 | 0.9677 |
| Decision Tree | Single learner | 0.6418 | 0.8942 |

XGBoost was the top performer, outperforming the weakest single learner
(Decision Tree, 0.6418 PR-AUC) by a wide margin — confirming the value of
ensembling on severely imbalanced data. Every ensemble tier beats every
single learner on PR-AUC, and the two boosting models (XGBoost, LightGBM)
edge out the Voting/Stacking meta-ensembles built on top of them. See the
[Streamlit app](https://ensemblelearning-7twystgmwcga9x5xl9zhkb.streamlit.app/)
for the full model comparison, decision surfaces, and SHAP explainability.

---

## Architecture

The pipeline is a linear, checkpointed sequence of six stages, each reading
the previous stage's output and writing its own:

```
data_quality → preprocessing → train → decision_surfaces → shap_explain → export_artifacts
```

| Stage | What it does |
|---|---|
| **1. Data quality** | Schema, missing-value, duplicate, outlier, and leakage checks |
| **2. Preprocessing** | Log1p transform, scaling, stratified train/test split |
| **3. Train** | Runs the Optuna joint search per model, builds Tier 3 meta-models |
| **4. Decision surfaces** | Fits cached 2D shadow models for visualization on arbitrary feature pairs |
| **5. SHAP explain** | Global feature-importance explanation on the best tree-based model |
| **6. Export artifacts** | Writes the manifest the Streamlit app reads |

An **orchestrator** (`run_pipeline.py`) runs all six stages end-to-end,
checkpointed per stage and per model — a crash mid-run resumes from the
last completed model rather than restarting. Optuna itself persists every
trial to a local SQLite DB, so a half-finished hyperparameter search also
resumes rather than starting over.

Every artifact path is versioned by a `RUN_VERSION` string, so re-running
the pipeline for a new experiment never overwrites a previous one.

**Tracking & deployment layer**, running alongside the pipeline:
- **MLflow** (hosted on DagsHub) logs one run per final model result,
  nested under a parent run for the whole `RUN_VERSION`.
- **CI** (GitHub Actions) runs lint + a full pipeline smoke test against a
  synthetic fixture on every push.
- **CD** (GitHub Actions) compares a candidate run's PR-AUC against the
  currently deployed best score, merges to `main` and promotes the model
  in the MLflow Model Registry if it wins, and does nothing otherwise.
  Streamlit Cloud auto-redeploys from `main`.

```
fraud-detection-project/
├── src/
│   ├── config.py             # paths, constants, RUN_VERSION plumbing
│   ├── data_quality.py       # Stage 1
│   ├── preprocessing.py      # Stage 2
│   ├── model_configs.py      # model factories + Optuna search spaces
│   ├── resamplers.py         # SMOTE/ADASYN/etc. factory (CV-fold scoped)
│   ├── optuna_objective.py   # joint resampler + model + hyperparam objective
│   ├── train.py              # Stage 3
│   ├── decision_surfaces.py  # Stage 4
│   ├── shap_explain.py       # Stage 5
│   ├── export_artifacts.py   # Stage 6
│   ├── compare_and_deploy.py # CD gate + MLflow model-registry promotion
│   └── run_pipeline.py       # orchestrator
├── app/streamlit_app.py      # reads the exported manifest only
├── notebooks/                # exploratory EDA / PCA / t-SNE
├── tests/                    # pipeline smoke tests + synthetic fixture
├── .github/workflows/        # ci.yml, cd.yml
└── Makefile
```

---

## Tools & stack

| Category | Tools |
|---|---|
| Modeling | scikit-learn, XGBoost, LightGBM |
| Imbalance handling | imbalanced-learn (SMOTE, ADASYN, SMOTETomek) |
| Hyperparameter search | Optuna (joint resampler + model + hyperparameter objective) |
| Explainability | SHAP |
| Experiment tracking | MLflow (hosted on DagsHub) |
| App | Streamlit |
| CI/CD | GitHub Actions |
| Data | pandas |

Everything is open-source and runs locally — **no API keys required**.

---

## Getting started

```bash
git clone https://github.com/Sanchit-Garg-4850/Ensemble_Learning
cd fraud-detection-project

# Place the dataset (not included — see Dataset section below)
cp /path/to/creditcard.csv data/

make install          # sets up a virtualenv, installs requirements.txt
make pipeline         # runs the full pipeline (resumable)
make app              # launches the Streamlit app
```

Tune search depth via environment variables:

```bash
FRAUD_N_TRIALS=100 FRAUD_CV_FOLDS=5 make pipeline
```

Run a single stage while iterating:

```bash
cd src && python run_pipeline.py --only shap_explain
cd src && python run_pipeline.py --from decision_surfaces
```

**Requirements:** Python 3.11+, `make`, Git. No GPU required.

---

## Dataset

The raw dataset (~150MB) isn't committed — it exceeds GitHub's 100MB file
limit and is licensed via Kaggle. Download it from
[Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it at
`data/creditcard.csv`. CI runs against a 420-row synthetic fixture instead,
so the test suite needs no real data.

## CI/CD

- **`ci.yml`** runs on every push/PR: lint (flake8) + a full pipeline
  smoke test against the synthetic fixture — catches import errors, API
  drift across library versions, and broken checkpoint logic in seconds.
- **`cd.yml`** gates promotion, not training: pushing to a `candidate`
  branch compares the run's PR-AUC against the current best and, if it
  wins, merges to `main` and promotes the model in the MLflow Model
  Registry. Full training on the real 285K-row dataset (50–100 Optuna
  trials/model, up to a couple of hours) is a separate, manually-triggered
  `workflow_dispatch` job intended for a self-hosted runner — GitHub's free
  hosted runners can't hold the dataset or the runtime.

## Design notes

- **Resampling is fold-scoped**, not applied before the CV split — this
  prevents synthetic (SMOTE/ADASYN) samples from leaking between train and
  validation folds.
- **SHAP explains the best tree-based model**, not the Voting/Stacking
  meta-ensembles — `TreeExplainer` doesn't apply to a classifier wrapping
  multiple sub-models, and the `KernelExplainer` fallback is too slow to
  run routinely.
- **EDA lives in a notebook, not a pipeline stage** — it's exploratory and
  narrative, not something a downstream stage consumes as an artifact.
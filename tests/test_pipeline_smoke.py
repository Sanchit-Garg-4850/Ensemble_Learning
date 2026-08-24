"""
Smoke tests: run each pipeline stage against the tiny synthetic fixture
(tests/fixtures/sample.csv, 420 rows) to catch import errors, API
mismatches, and broken paths fast in CI. These do NOT validate model
quality -- that's the job of the real local run against the full dataset.

Versioning: forced to FRAUD_RUN_VERSION="smoketest" so CI's throwaway
artifacts never collide with your real local v1/v2 (etc.) experiment
files, and never accidentally get picked up as a CD deploy candidate --
compare_and_deploy.py globs "model_comparison_v*.csv" on purpose, which
"model_comparison_smoketest.csv" doesn't match.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ["FRAUD_DATA_PATH"] = str(ROOT / "tests" / "fixtures" / "sample.csv")
os.environ["FRAUD_N_TRIALS"] = "2"
os.environ["FRAUD_CV_FOLDS"] = "2"
os.environ["FRAUD_RUN_VERSION"] = "smoketest"

import importlib # noqa: E402
import config  # noqa: E402


def _reload_config():
    importlib.reload(config)


def test_data_quality_stage(tmp_path):
    import data_quality
    report = data_quality.run()
    assert report["shape"][0] == 420
    assert report["missing_values_total"] == 0


def test_preprocessing_stage():
    import preprocessing
    preprocessing.run()
    assert (config.OUTPUT_DIR / "train.parquet").exists()
    assert (config.OUTPUT_DIR / "test.parquet").exists()


def test_train_stage_tier1_only(monkeypatch):
    # Restrict to one cheap model so the smoke test stays fast
    import config as cfg
    monkeypatch.setitem(cfg.TIERS, "tier1_single", ["decision_tree"])
    monkeypatch.setitem(cfg.TIERS, "tier2_ensemble", ["random_forest"])
    monkeypatch.setitem(cfg.TIERS, "tier3_meta", ["voting"])
    import train
    train.run()
    assert config.COMPARISON_CSV_PATH.exists()


def test_export_artifacts_stage():
    import export_artifacts
    export_artifacts.run()
    assert config.APP_MANIFEST_PATH.exists()

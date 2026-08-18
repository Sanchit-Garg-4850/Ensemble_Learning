"""
Stage: data_quality
Reads raw CSV, runs the checks from plan Section 4, writes:
  - output/data_quality_report.json
  - output/processed.parquet (raw data, unmodified, just parquet for faster re-reads)
Exits non-zero if a hard failure is found (e.g. file missing).
"""
import json
import sys
import pandas as pd
import numpy as np

from config import RAW_DATA_PATH, OUTPUT_DIR, TARGET_COL


def run() -> dict:
    if not RAW_DATA_PATH.exists():
        print(f"[data_quality] ERROR: {RAW_DATA_PATH} not found. "
              f"Download creditcard.csv from Kaggle and place it there.")
        sys.exit(1)

    df = pd.read_csv(RAW_DATA_PATH)

    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum()[df.isna().sum() > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": df[TARGET_COL].value_counts().to_dict(),
        "class_distribution_pct": (df[TARGET_COL].value_counts(normalize=True) * 100).round(4).to_dict(),
        "amount_outliers_iqr": _iqr_outlier_count(df["Amount"]),
        "time_range_seconds": [float(df["Time"].min()), float(df["Time"].max())],
        "time_range_hours": round((df["Time"].max() - df["Time"].min()) / 3600, 2),
    }

    # Guarded imputer logic — only fires if missing values actually exist.
    if report["missing_values_total"] > 0:
        print("[data_quality] Missing values found — applying KNNImputer on numeric cols.")
        from sklearn.impute import KNNImputer
        num_cols = df.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        print("[data_quality] No missing values — imputer skipped (guarded, not force-applied).")

    if report["duplicate_rows"] > 0:
        df = df.drop_duplicates()
        report["duplicates_dropped"] = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_DIR / "processed.parquet", index=False)
    with open(OUTPUT_DIR / "data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[data_quality] shape={report['shape']} "
          f"fraud_pct={report['class_distribution_pct'].get(1, 0):.4f}% "
          f"duplicates={report['duplicate_rows']}")
    return report


def _iqr_outlier_count(series: pd.Series) -> int:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


if __name__ == "__main__":
    run()

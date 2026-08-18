"""
Stage: preprocessing
log1p on Amount, StandardScaler on Amount/Time, stratified train/test split.
Resampling is intentionally NOT applied here — it happens inside the imblearn
Pipeline per CV fold during Optuna search (see optuna_objective.py) to avoid
leaking synthetic minority points into the validation/test folds.
Writes output/train.parquet, output/test.parquet, output/scaler.joblib
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_DIR, TARGET_COL, RANDOM_STATE, TEST_SIZE


def run():
    df = pd.read_parquet(OUTPUT_DIR / "processed.parquet")

    skew_before = df["Amount"].skew()
    df["Amount"] = np.log1p(df["Amount"])
    skew_after = df["Amount"].skew()
    print(f"[preprocessing] Amount skew: {skew_before:.3f} -> {skew_after:.3f} (log1p)")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    train_df.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    test_df.to_parquet(OUTPUT_DIR / "test.parquet", index=False)
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")

    print(f"[preprocessing] train={train_df.shape} test={test_df.shape} "
          f"train_fraud_pct={y_train.mean()*100:.4f}% test_fraud_pct={y_test.mean()*100:.4f}%")


if __name__ == "__main__":
    run()

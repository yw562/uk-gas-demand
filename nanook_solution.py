#!/usr/bin/env python3
"""
Nanook Quantitative Modelling Task — Reference Solution (Python)

Author: (Your Name)
Description:
    Train a time-aware regression model to predict UK domestic gas demand
    using provided weather features. The script:
      1) Loads train/test CSVs.
      2) Engineers calendar + simple aggregate features.
      3) Evaluates multiple baselines on an out-of-time (2017) validation.
      4) Selects the best model by RMSE.
      5) Retrains on full train period and generates submission.csv.

How to run:
    python nanook_solution.py --train train.csv --test test.csv --out submission.csv

Optional flags:
    --no-permutation      # Skip permutation importance (off by default)
    --seed 42             # Random seed (default: 42)

Outputs:
    - Prints validation metrics (MAE/RMSE/MAPE) for seasonal_naive, ridge, hgbr
    - Writes submission.csv with columns: id, demand
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


# ----------------------------
# Metrics utilities
# ----------------------------
@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float

    def as_dict(self) -> Dict[str, float]:
        return {"MAE": self.mae, "RMSE": self.rmse, "MAPE": self.mape}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Return MAE, RMSE, MAPE (safe for near-zero demand)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # <- version-agnostic
    eps = 1e-6
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(eps, np.abs(y_true)))) * 100.0)
    return Metrics(mae=mae, rmse=rmse, mape=mape)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nanook Quant Modelling Task")
    p.add_argument("--train", type=str, default="train.csv")
    p.add_argument("--test", type=str, default="test.csv")
    p.add_argument("--out", type=str, default="submission.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-permutation", action="store_true", help="Skip permutation importance calculation")
    return p.parse_args()


# ----------------------------
# Feature engineering
# ----------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and compact aggregate features. Does NOT touch target."""
    out = df.copy()

    # Parse date
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year.astype(np.int16)
    out["month"] = out["date"].dt.month.astype(np.int8)
    out["dow"] = out["date"].dt.dayofweek.astype(np.int8)
    out["doy"] = out["date"].dt.dayofyear.astype(np.int16)
    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)
    # Seasonal cycles
    out["sin_doy"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * out["doy"] / 365.25)

    # Column groups (1..13 lags)
    day_temps = [f"temp_{i}" for i in range(1, 14) if f"temp_{i}" in out.columns]
    night_temps = [f"temp_night_{i}" for i in range(1, 14) if f"temp_night_{i}" in out.columns]
    day_wind = [f"wind_{i}" for i in range(1, 14) if f"wind_{i}" in out.columns]
    night_wind = [f"wind_night_{i}" for i in range(1, 14) if f"wind_night_{i}" in out.columns]
    ssrd = [f"ssrd_ratio_{i}" for i in range(1, 14) if f"ssrd_ratio_{i}" in out.columns]

    def add_basic_stats(prefix: str, cols: List[str]) -> None:
        if not cols:
            return
        mat = out[cols]
        out[f"{prefix}_mean"] = mat.mean(axis=1)
        out[f"{prefix}_min"] = mat.min(axis=1)
        out[f"{prefix}_max"] = mat.max(axis=1)
        out[f"{prefix}_std"] = mat.std(axis=1)

    add_basic_stats("temp_day", day_temps)
    add_basic_stats("temp_night", night_temps)
    add_basic_stats("wind_day", day_wind)
    add_basic_stats("wind_night", night_wind)
    add_basic_stats("ssrd", ssrd)

    # Thermal gradient proxy: day - night
    if set(day_temps) and set(night_temps):
        out["temp_daynight_mean_diff"] = out["temp_day_mean"] - out["temp_night_mean"]

    return out


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return model input columns (drop id/date/target)."""
    drop_cols = {"id", "date", "demand"}
    return [c for c in df.columns if c not in drop_cols]


# ----------------------------
# Baselines
# ----------------------------
def seasonal_naive_predict(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """
    Simple climatology baseline:
        Predict the mean demand for each (day-of-year, is_weekend) learned from train_df.
        If a (doy, is_weekend) pair is unseen, fall back to the overall mean.
    """
    tmp = train_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["doy"] = tmp["date"].dt.dayofyear
    tmp["is_weekend"] = (tmp["date"].dt.dayofweek >= 5).astype(int)
    table = tmp.groupby(["doy", "is_weekend"])["demand"].mean()

    v = val_df.copy()
    v["date"] = pd.to_datetime(v["date"])
    v["doy"] = v["date"].dt.dayofyear
    v["is_weekend"] = (v["date"].dt.dayofweek >= 5).astype(int)
    preds = []
    overall = float(tmp["demand"].mean())
    for _, row in v[["doy", "is_weekend"]].iterrows():
        preds.append(float(table.get((row["doy"], row["is_weekend"]), overall)))
    return np.asarray(preds)


# ----------------------------
# Modelling
# ----------------------------
def build_ridge_pipeline(feature_names: List[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=3.0)),
        ]
    )


def build_hgbr_pipeline(feature_names: List[str], seed: int) -> Pipeline:
    # HistGradientBoosting handles NaNs natively; no scaling required.
    hgbr = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=350,
        max_bins=255,
        early_stopping=True,
        random_state=seed,
    )
    return Pipeline(steps=[("model", hgbr)])


def evaluate_models(train_df: pd.DataFrame, seed: int, run_perm: bool = False) -> Tuple[str, Dict[str, Metrics], Dict[str, np.ndarray]]:
    """
    Split by time: 2009-2016 train, 2017 validation.
    Train baselines (seasonal), Ridge, and HGBR; return metrics and predictions.
    """
    eng = engineer_features(train_df)

    # Time split
    eng["date"] = pd.to_datetime(eng["date"])
    train_mask = eng["date"] < pd.Timestamp("2017-01-01")
    tr = eng[train_mask].copy()
    va = eng[~train_mask].copy()

    y_tr = tr["demand"].to_numpy()
    y_va = va["demand"].to_numpy()

    feats = get_feature_columns(eng)
    X_tr = tr[feats]
    X_va = va[feats]

    results: Dict[str, Metrics] = {}
    preds: Dict[str, np.ndarray] = {}

    # 1) Seasonal naive
    preds["seasonal_naive"] = seasonal_naive_predict(tr, va)
    results["seasonal_naive"] = compute_metrics(y_va, preds["seasonal_naive"])

    # 2) Ridge (linear baseline)
    ridge = build_ridge_pipeline(feats)
    ridge.fit(X_tr, y_tr)
    preds["ridge"] = ridge.predict(X_va)
    results["ridge"] = compute_metrics(y_va, preds["ridge"])

    # 3) HistGradientBoosting (nonlinear)
    hgbr = build_hgbr_pipeline(feats, seed=seed)
    hgbr.fit(X_tr, y_tr)
    preds["hgbr"] = hgbr.predict(X_va)
    results["hgbr"] = compute_metrics(y_va, preds["hgbr"])

    # Report
    print("\nValidation metrics (2017 hold-out):")
    for name, m in results.items():
        print(f"  - {name:16s} | MAE={m.mae:7.3f}  RMSE={m.rmse:7.3f}  MAPE={m.mape:6.2f}%")

    # Optional: permutation importance for HGBR
    if run_perm:
        try:
            print("\nTop permutation importances (HGBR):")
            perm = permutation_importance(hgbr, X_va, y_va, n_repeats=5, random_state=seed)
            importances = pd.Series(perm.importances_mean, index=feats).sort_values(ascending=False)
            print(importances.head(15).to_string())
        except Exception as e:
            print(f"[WARN] Permutation importance failed: {e}")

    # Select best by RMSE
    best_name = min(results.keys(), key=lambda k: results[k].rmse)
    print(f"\nSelected model by RMSE: {best_name}")
    return best_name, results, preds


def fit_full_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, best_name: str, seed: int) -> np.ndarray:
    """Retrain the winning model on full 2009-2017 and predict for the test set."""
    eng_train = engineer_features(train_df)
    eng_test = engineer_features(test_df)

    feats = get_feature_columns(eng_train)
    X_full = eng_train[feats]
    y_full = eng_train["demand"].to_numpy()
    X_test = eng_test[feats]

    if best_name == "ridge":
        model = build_ridge_pipeline(feats)
    elif best_name == "hgbr":
        model = build_hgbr_pipeline(feats, seed=seed)
    else:
        # If seasonal naive happens to win (unlikely), use it.
        tmp_tr = train_df.copy()
        tmp_tr["date"] = pd.to_datetime(tmp_tr["date"])
        tmp_tr["doy"] = tmp_tr["date"].dt.dayofyear
        tmp_tr["is_weekend"] = (tmp_tr["date"].dt.dayofweek >= 5).astype(int)
        table = tmp_tr.groupby(["doy", "is_weekend"])["demand"].mean()
        tt = test_df.copy()
        tt["date"] = pd.to_datetime(tt["date"])
        tt["doy"] = tt["date"].dt.dayofyear
        tt["is_weekend"] = (tt["date"].dt.dayofweek >= 5).astype(int)
        overall = float(tmp_tr["demand"].mean())
        preds = []
        for _, row in tt[["doy", "is_weekend"]].iterrows():
            preds.append(float(table.get((row["doy"], row["is_weekend"]), overall)))
        return np.asarray(preds)

    model.fit(X_full, y_full)
    pred = model.predict(X_test)
    # Safety: clip to positive
    pred = np.clip(pred, 0.0, None)
    return pred


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # Load
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    # Quick sanity
    assert "demand" in train.columns, "Train must contain 'demand' column"
    assert set(["id", "date"]).issubset(train.columns), "Train must contain 'id' and 'date'"
    assert set(["id", "date"]).issubset(test.columns), "Test must contain 'id' and 'date'"

    # Evaluate on 2017 hold-out and select model
    best_name, metrics, _ = evaluate_models(train, seed=args.seed, run_perm=not args.no_permutation)

    # Fit best model on full data and predict test
    preds = fit_full_and_predict(train, test, best_name=best_name, seed=args.seed)

    # Save submission
    sub = pd.DataFrame({"id": test["id"].astype(int), "demand": preds})
    sub.to_csv(args.out, index=False)
    print(f"\nWrote predictions to: {args.out}")
    print("Done.")


if __name__ == "__main__":
    main()

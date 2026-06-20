from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "calisol23.csv"
MODEL_PATH = ROOT / "models" / "conductivity_model.joblib"
REPORT_PATH = ROOT / "models" / "training_report.json"


def clean_salt(value: str) -> str:
    value = str(value).strip()
    aliases = {"LiN(CF3SO2)2": "LiTFSI"}
    return aliases.get(value, value)


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("Missing data/raw/calisol23.csv. Run scripts/sync_public_data.py first.")

    df = pd.read_csv(DATA_PATH)
    solvent_columns = list(df.columns[7:])
    df["salt"] = df["salt"].map(clean_salt)
    df = df.dropna(subset=["k", "T", "c", "salt", "c units", "solvent ratio type"]).copy()
    df = df[(df["k"] >= 0) & (df["k"] <= df["k"].quantile(0.997))]

    base = df[["T", "c", *solvent_columns]].astype(float)
    categorical = pd.get_dummies(
        df[["salt", "c units", "solvent ratio type"]].rename(
            columns={"c units": "unit", "solvent ratio type": "ratio"}
        ),
        prefix=["salt", "unit", "ratio"],
        dtype=float,
    )
    x = pd.concat([base.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)
    y = df["k"].to_numpy(dtype=float)

    fallback_groups = pd.Series(df.index.astype(str), index=df.index)
    groups = df["doi"].fillna(fallback_groups).astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))
    model = ExtraTreesRegressor(
        n_estimators=220,
        min_samples_leaf=2,
        max_features=0.82,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x.iloc[train_idx], y[train_idx])
    pred = model.predict(x.iloc[test_idx])
    metrics = {
        "mae": round(float(mean_absolute_error(y[test_idx], pred)), 4),
        "r2": round(float(r2_score(y[test_idx], pred)), 4),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "group_split": "DOI",
        "target": "conductivity k as reported by CALiSol-23",
    }
    positive_solvents = df[solvent_columns].gt(0)
    salt_counts = df["salt"].value_counts().to_dict()
    solvent_counts = {
        salt: positive_solvents.loc[index].sum(axis=0).astype(int).to_dict()
        for salt, index in df.groupby("salt").groups.items()
    }
    pair_counts: dict[str, dict[str, int]] = {}
    for salt, group in df.groupby("salt"):
        counts: dict[str, int] = {}
        for _, row in group[solvent_columns].iterrows():
            active = sorted(row.index[row.gt(0)].tolist())
            if len(active) == 2:
                key = "|".join(active)
                counts[key] = counts.get(key, 0) + 1
        pair_counts[salt] = counts
    salt_ranges = {
        salt: {
            "temperature_min": float(group["T"].min()),
            "temperature_max": float(group["T"].max()),
            "concentration_min": float(group["c"].min()),
            "concentration_max": float(group["c"].max()),
            "target_std": float(max(group["k"].std(), 0.5)),
        }
        for salt, group in df.groupby("salt")
    }
    bundle = {
        "model": model,
        "feature_columns": list(x.columns),
        "solvent_columns": solvent_columns,
        "supported_salts": sorted(df["salt"].unique().tolist()),
        "metrics": metrics,
        "salt_counts": salt_counts,
        "solvent_counts": solvent_counts,
        "pair_counts": pair_counts,
        "salt_ranges": salt_ranges,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

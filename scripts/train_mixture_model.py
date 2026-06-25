from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from app.mixture_features import feature_row_from_record


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mixture_experiments.csv"
MODEL_PATH = ROOT / "models" / "mixture_model.joblib"
REPORT_PATH = ROOT / "models" / "mixture_model_report.json"


def make_matrix(df: pd.DataFrame, known_salts: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows = [
        feature_row_from_record(record, known_salts)
        for record in df.to_dict(orient="records")
    ]
    x = pd.DataFrame(rows).fillna(0.0)
    x = x.reindex(sorted(x.columns), axis=1)
    return x, list(x.columns)


def grouped_metrics(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    model: HistGradientBoostingRegressor,
    inverse=None,
) -> dict[str, float | int | str]:
    unique_groups = groups.astype(str).nunique()
    if len(y) < 12 or unique_groups < 2:
        pred = model.predict(x)
        actual = y
        split_name = "training_only"
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(splitter.split(x, y, groups=groups.astype(str)))
        fold_model = HistGradientBoostingRegressor(
            max_iter=model.max_iter,
            learning_rate=model.learning_rate,
            max_leaf_nodes=model.max_leaf_nodes,
            min_samples_leaf=model.min_samples_leaf,
            l2_regularization=model.l2_regularization,
            random_state=42,
        )
        fold_model.fit(x.iloc[train_idx], y[train_idx])
        pred = fold_model.predict(x.iloc[test_idx])
        actual = y[test_idx]
        split_name = "group_shuffle_by_source"
    if inverse:
        pred_eval = inverse(pred)
        actual_eval = inverse(actual)
    else:
        pred_eval = pred
        actual_eval = actual
    metrics = {
        "mae": round(float(mean_absolute_error(actual_eval, pred_eval)), 5),
        "r2": round(float(r2_score(actual_eval, pred_eval)), 4) if len(actual_eval) > 1 else 0.0,
        "validation_rows": int(len(actual_eval)),
        "validation_split": split_name,
    }
    return metrics


def train_conductivity(df: pd.DataFrame, known_salts: list[str]) -> dict:
    train = df.dropna(subset=["conductivity", "temperature_c"]).copy()
    train = train[(train["conductivity"].astype(float) >= 0) & (train["component_count"].between(1, 3))]
    if len(train) < 50:
        return {"available": False, "reason": "not_enough_conductivity_rows", "rows": int(len(train))}
    x, feature_columns = make_matrix(train, known_salts)
    y = np.log1p(train["conductivity"].to_numpy(dtype=float))
    model = HistGradientBoostingRegressor(
        max_iter=260,
        learning_rate=0.055,
        max_leaf_nodes=31,
        min_samples_leaf=12,
        l2_regularization=0.08,
        random_state=42,
    )
    model.fit(x, y)
    metrics = grouped_metrics(
        x,
        y,
        train["doi"].fillna(train["source_id"]),
        model,
        inverse=np.expm1,
    )
    metrics.update(
        {
            "available": True,
            "rows": int(len(train)),
            "target": "conductivity",
            "target_transform": "log1p",
            "unique_salts": int(train["salt"].nunique()),
            "unique_formulas": int(
                train[["salt", "solvent_a", "solvent_b", "solvent_c"]]
                .astype(str)
                .apply(lambda row: "|".join(map(str, row)), axis=1)
                .nunique()
            ),
        }
    )
    return {"model": model, "feature_columns": feature_columns, "metrics": metrics}


def train_solubility(df: pd.DataFrame, known_salts: list[str]) -> dict:
    train = df.dropna(subset=["solubility_mole_fraction", "temperature_c"]).copy()
    train = train[(train["solubility_mole_fraction"].astype(float) > 0) & (train["component_count"].between(1, 3))]
    if len(train) < 20:
        return {"available": False, "reason": "not_enough_solubility_rows", "rows": int(len(train))}
    x, feature_columns = make_matrix(train, known_salts)
    y = np.log10(train["solubility_mole_fraction"].to_numpy(dtype=float))
    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.055,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        l2_regularization=0.20,
        random_state=42,
    )
    model.fit(x, y)
    formula_groups = (
        train[["salt", "solvent_a", "solvent_b", "solvent_c"]]
        .astype(str)
        .apply(lambda row: "|".join(map(str, row)), axis=1)
    )
    metrics = grouped_metrics(
        x,
        y,
        formula_groups,
        model,
        inverse=lambda values: np.power(10.0, values),
    )
    metrics.update(
        {
            "available": True,
            "rows": int(len(train)),
            "target": "solubility_mole_fraction",
            "target_transform": "log10",
            "unique_salts": int(train["salt"].nunique()),
            "unique_formulas": int(formula_groups.nunique()),
            "lino3_binary_rows": int(
                len(train[(train["salt"] == "LiNO3") & (train["component_count"] == 2)])
            ),
        }
    )
    return {"model": model, "feature_columns": feature_columns, "metrics": metrics}


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("Missing data/mixture_experiments.csv. Run scripts/build_mixture_experiments.py first.")
    df = pd.read_csv(DATA_PATH)
    known_salts = sorted(str(salt) for salt in df["salt"].dropna().unique())
    conductivity = train_conductivity(df, known_salts)
    solubility = train_solubility(df, known_salts)

    all_x, all_features = make_matrix(df, known_salts)
    feature_mean = all_x.mean(axis=0).to_dict()
    feature_std = all_x.std(axis=0).replace(0, 1.0).fillna(1.0).to_dict()
    bundle = {
        "known_salts": known_salts,
        "feature_columns": all_features,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "conductivity": {
            "model": conductivity.get("model"),
            "feature_columns": conductivity.get("feature_columns", all_features),
            "metrics": conductivity["metrics"] if conductivity.get("model") else conductivity,
            "supported_salts": sorted(df.dropna(subset=["conductivity"])["salt"].dropna().astype(str).unique().tolist()),
        },
        "solubility": {
            "model": solubility.get("model"),
            "feature_columns": solubility.get("feature_columns", all_features),
            "metrics": solubility["metrics"] if solubility.get("model") else solubility,
            "supported_salts": sorted(df.dropna(subset=["solubility_mole_fraction"])["salt"].dropna().astype(str).unique().tolist()),
        },
        "training_summary": {
            "rows": int(len(df)),
            "source_counts": df["source_name"].value_counts().to_dict(),
            "target_counts": df["source_type"].value_counts().to_dict(),
            "component_counts": {
                str(key): int(value)
                for key, value in df["component_count"].value_counts().sort_index().items()
            },
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    report = {
        "conductivity": bundle["conductivity"]["metrics"],
        "solubility": bundle["solubility"]["metrics"],
        "training_summary": bundle["training_summary"],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from app.mixture_features import solvent_properties


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "oedb_computational_properties.csv"
MODEL_PATH = ROOT / "models" / "oedb_auxiliary_model.joblib"
REPORT_PATH = ROOT / "models" / "oedb_auxiliary_model_report.json"

TARGETS = {
    "viscosity_mpas": {
        "column": "Viscosity (mPa·s)",
        "transform": "log1p",
    },
    "density_g_cm3": {
        "column": "Density (g/cm³)",
        "transform": "identity",
    },
    "cation_diffusivity_m2_s": {
        "column": "Cation's Diffusivity (m²/s)",
        "transform": "log10",
    },
    "anion_diffusivity_m2_s": {
        "column": "Anion's Diffusivity (m²/s)",
        "transform": "log10",
    },
    "solvent_diffusivity_m2_s": {
        "column": "Solvent's Diffusivity (m²/s)",
        "transform": "log10",
    },
    "cation_anion_coordination": {
        "column": "Coordination Number (Cation ← Anion)",
        "transform": "identity",
    },
    "cation_solvent_coordination": {
        "column": "Coordination Number (Cation ← Solvent)",
        "transform": "identity",
    },
}


def transformed(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    if transform == "log10":
        return np.log10(np.maximum(values, 1e-14))
    return values


def inverse(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "log1p":
        return np.expm1(values)
    if transform == "log10":
        return np.power(10.0, values)
    return values


def feature_row(record: dict, cations: list[str], anions: list[str], solvents: list[str]) -> dict[str, float]:
    solvent = str(record["solvent"])
    props = solvent_properties(solvent)
    row = {
        "concentration_mol_kg": float(record["Concentration (mol/kg)"]),
        "molecular_weight": float(props.get("molecular_weight", 0.0)),
        "tpsa": float(props.get("tpsa", 0.0)),
        "logp": float(props.get("logp", 0.0)),
        "hbond_acceptors": float(props.get("hbond_acceptors", 0.0)),
        "hbond_donors": float(props.get("hbond_donors", 0.0)),
        "rotatable_bonds": float(props.get("rotatable_bonds", 0.0)),
        "fraction_csp3": float(props.get("fraction_csp3", 0.0)),
        "ring_count": float(props.get("ring_count", 0.0)),
        "dielectric_constant": float(props.get("dielectric_constant", 18.0)),
        "viscosity_mpas_solvent": float(props.get("viscosity_mpas", 1.2)),
        "boiling_point_c": float(props.get("boiling_point_c", 130.0)),
        "flash_point_c": float(props.get("flash_point_c", 35.0)),
        "donor_number": float(props.get("donor_number", 16.0)),
    }
    for cation in cations:
        row[f"cation_{cation}"] = 1.0 if record["cation"] == cation else 0.0
    for anion in anions:
        row[f"anion_{anion}"] = 1.0 if record["anion"] == anion else 0.0
    for known_solvent in solvents:
        row[f"solvent_{known_solvent}"] = 1.0 if solvent == known_solvent else 0.0
    return row


def make_matrix(df: pd.DataFrame, cations: list[str], anions: list[str], solvents: list[str]) -> pd.DataFrame:
    rows = [feature_row(record, cations, anions, solvents) for record in df.to_dict(orient="records")]
    return pd.DataFrame(rows).fillna(0.0).reindex(sorted(rows[0].keys()), axis=1)


def train_target(df: pd.DataFrame, x: pd.DataFrame, target_name: str, spec: dict) -> dict:
    column = spec["column"]
    transform = spec["transform"]
    valid = df[column].notna() & (df[column].astype(float) >= 0)
    train_x = x.loc[valid].copy()
    train_df = df.loc[valid].copy()
    y_raw = train_df[column].to_numpy(dtype=float)
    y = transformed(y_raw, transform)
    model = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(train_x, y)

    groups = (
        train_df[["cation", "anion", "solvent"]]
        .astype(str)
        .apply(lambda row: "|".join(row), axis=1)
    )
    if len(train_df) >= 50 and groups.nunique() > 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(splitter.split(train_x, y, groups=groups))
        fold_model = HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=0.05,
            random_state=42,
        )
        fold_model.fit(train_x.iloc[train_idx], y[train_idx])
        pred = inverse(fold_model.predict(train_x.iloc[test_idx]), transform)
        actual = y_raw[test_idx]
        validation_rows = int(len(test_idx))
        validation = "group_shuffle_by_cation_anion_solvent"
    else:
        pred = inverse(model.predict(train_x), transform)
        actual = y_raw
        validation_rows = int(len(train_df))
        validation = "training_only"
    metrics = {
        "rows": int(len(train_df)),
        "mae": float(f"{float(mean_absolute_error(actual, pred)):.6g}"),
        "r2": round(float(r2_score(actual, pred)), 4) if len(actual) > 1 else 0.0,
        "validation_rows": validation_rows,
        "validation": validation,
        "target_column": column,
        "transform": transform,
    }
    return {"model": model, "metrics": metrics}


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("Missing data/oedb_computational_properties.csv. Run scripts/sync_oedb_data.py first.")
    df = pd.read_csv(DATA_PATH)
    cations = sorted(df["cation"].dropna().astype(str).unique().tolist())
    anions = sorted(df["anion"].dropna().astype(str).unique().tolist())
    solvents = sorted(df["solvent"].dropna().astype(str).unique().tolist())
    x = make_matrix(df, cations, anions, solvents)
    targets = {
        name: train_target(df, x, name, spec)
        for name, spec in TARGETS.items()
    }
    bundle = {
        "models": {name: value["model"] for name, value in targets.items()},
        "target_specs": TARGETS,
        "target_metrics": {name: value["metrics"] for name, value in targets.items()},
        "feature_columns": list(x.columns),
        "cations": cations,
        "anions": anions,
        "solvents": solvents,
        "source": {
            "name": "OEDB computational",
            "url": "https://oedb.jp/",
            "doi": "10.1038/s41524-026-02093-y",
            "version": "2026-05-11",
            "data_type": "high-throughput molecular dynamics simulation",
        },
        "training_summary": {
            "rows": int(len(df)),
            "unique_cations": int(df["cation"].nunique()),
            "unique_anions": int(df["anion"].nunique()),
            "unique_solvents": int(df["solvent"].nunique()),
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    report = {
        "source": bundle["source"],
        "training_summary": bundle["training_summary"],
        "target_metrics": bundle["target_metrics"],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

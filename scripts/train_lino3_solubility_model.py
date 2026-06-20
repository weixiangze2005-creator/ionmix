from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from app.chemistry import molecular_descriptors


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "lino3_solubility.csv"
CATALOG_PATH = ROOT / "data" / "processed" / "solvent_catalog.csv"
MODEL_PATH = ROOT / "models" / "lino3_solubility_model.joblib"
REPORT_PATH = ROOT / "models" / "lino3_solubility_report.json"

FEATURES = [
    "temperature_c",
    "molecular_weight",
    "tpsa",
    "logp",
    "hbond_acceptors",
    "rotatable_bonds",
    "fraction_csp3",
    "ring_count",
    "dielectric_constant",
    "viscosity_mpas",
    "donor_number",
]


def build_rows() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    catalog = pd.read_csv(CATALOG_PATH).set_index("code")
    rows = []
    for record in data.to_dict(orient="records"):
        descriptors = molecular_descriptors(record["smiles"])
        properties = catalog.loc[record["solvent"]] if record["solvent"] in catalog.index else None
        rows.append(
            {
                **record,
                **descriptors,
                "dielectric_constant": float(properties["dielectric_constant"]) if properties is not None else 24.5,
                "viscosity_mpas": float(properties["viscosity_mpas"]) if properties is not None else 1.07,
                "donor_number": float(properties["donor_number"]) if properties is not None else 19.2,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = build_rows()
    x = df[FEATURES].astype(float)
    y = np.log10(df["solubility_mole_fraction"].to_numpy(dtype=float))
    weights = 1.0 / np.square(df["relative_uncertainty"].to_numpy(dtype=float))
    weights = weights / np.mean(weights)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=1,
        max_features=0.85,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_scaled, y, sample_weight=weights)
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(x_scaled, y, sample_weight=weights)

    # Leave one solvent family out: this intentionally measures the hard
    # structure-extrapolation case rather than an easy random-row split.
    logo = LeaveOneGroupOut()
    cv_pred = np.empty_like(y)
    for train_index, test_index in logo.split(x, y, groups=df["solvent"]):
        fold_scaler = StandardScaler()
        fold_train = fold_scaler.fit_transform(x.iloc[train_index])
        fold_test = fold_scaler.transform(x.iloc[test_index])
        fold_model = Ridge(alpha=10.0)
        fold_model.fit(fold_train, y[train_index], sample_weight=weights[train_index])
        cv_pred[test_index] = fold_model.predict(fold_test)
    metrics = {
        "rows": int(len(df)),
        "unique_solvents": int(df["solvent"].nunique()),
        "temperature_min_c": float(df["temperature_c"].min()),
        "temperature_max_c": float(df["temperature_c"].max()),
        "leave_one_solvent_out_mae_log10": round(float(mean_absolute_error(y, cv_pred)), 4),
        "leave_one_solvent_out_r2_log10": round(float(r2_score(y, cv_pred)), 4),
        "prediction_model": "Ridge(alpha=10) with ExtraTrees uncertainty ensemble",
        "primary_source_doi": "10.1016/j.fluid.2017.12.034",
        "primary_measurement_uncertainty": "5%",
    }
    bundle = {
        "model": model,
        "ridge_model": ridge_model,
        "scaler": scaler,
        "features": FEATURES,
        "training_x_scaled": x_scaled,
        "training_log_y": y,
        "training_solvents": df["solvent"].tolist(),
        "relative_uncertainties": df["relative_uncertainty"].tolist(),
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    REPORT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

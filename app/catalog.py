from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.chemistry import molecular_descriptors


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_CATALOG = ROOT / "data" / "processed" / "solvent_catalog.csv"
SEED_CATALOG = ROOT / "data" / "solvent_seed.csv"

TRAINING_CODE_MAP = {
    "GBL": "g-Butyrolactone",
}

# 0 is poor and 1 is strong. These are coarse screening priors, not measured
# electrochemical-window labels.
OXIDATION_PRIOR = {
    "EC": 0.78, "PC": 0.75, "DMC": 0.70, "EMC": 0.72, "DEC": 0.73,
    "FEC": 0.84, "Sulfolane": 0.92, "TEP": 0.88, "TMP": 0.85,
    "DME": 0.42, "DOL": 0.38, "THF": 0.30, "2-MeTHF": 0.34,
    "2-Glyme": 0.42, "3-Glyme": 0.44, "4-Glyme": 0.46,
    "DMSO": 0.56, "AN": 0.68, "DMF": 0.45, "NMP": 0.48, "GBL": 0.76,
    "BTFE": 0.86, "TTE": 0.90,
}
REDUCTION_PRIOR = {
    "EC": 0.74, "FEC": 0.90, "DME": 0.74, "DOL": 0.72, "THF": 0.63,
    "2-MeTHF": 0.72, "2-Glyme": 0.73, "3-Glyme": 0.70, "4-Glyme": 0.68,
    "DMSO": 0.30, "AN": 0.42, "DMF": 0.30, "NMP": 0.32,
    "PC": 0.45, "DMC": 0.56, "EMC": 0.58, "DEC": 0.60,
    "Sulfolane": 0.45, "GBL": 0.48, "BTFE": 0.83, "TTE": 0.88,
}
HAZARD_PRIOR = {
    "DMF": 0.85, "NMP": 0.82, "AN": 0.72, "DMSO": 0.35,
    "DEE": 0.75, "THF": 0.65, "DME": 0.62, "DOL": 0.58,
    "2-MeTHF": 0.52, "MA": 0.60, "EA": 0.50, "FEC": 0.42,
    "Sulfolane": 0.48, "TEP": 0.38, "TMP": 0.42,
}
MELTING_POINT_C = {
    "EC": 36.4, "PC": -49.0, "DMC": 4.6, "EMC": -53.0, "DEC": -43.0,
    "DME": -58.0, "DMSO": 18.5, "AN": -45.7, "EA": -84.0, "MA": -98.0,
    "FEC": 18.0, "DOL": -95.0, "2-MeTHF": -137.0, "THF": -109.0,
    "Sulfolane": 27.5, "2-Glyme": -68.0, "3-Glyme": -45.0, "4-Glyme": -30.0,
    "DMF": -61.0, "GBL": -43.5, "NMP": -24.0, "DEE": -116.0,
    "TEP": -56.0, "TMP": -46.0, "BTFE": -60.0, "TTE": -60.0,
}


def load_catalog() -> pd.DataFrame:
    path = PROCESSED_CATALOG if PROCESSED_CATALOG.exists() else SEED_CATALOG
    df = pd.read_csv(path)
    descriptor_rows = [molecular_descriptors(s) for s in df["smiles"]]
    desc = pd.DataFrame(descriptor_rows)
    for col in desc:
        if col not in df.columns:
            df[col] = desc[col]
        else:
            df[col] = df[col].fillna(desc[col])
    df["training_code"] = df["code"].replace(TRAINING_CODE_MAP)
    df["oxidation_prior"] = df["code"].map(OXIDATION_PRIOR).fillna(0.60)
    df["reduction_prior"] = df["code"].map(REDUCTION_PRIOR).fillna(0.55)
    df["hazard_prior"] = df["code"].map(HAZARD_PRIOR).fillna(0.45)
    df["melting_point_c"] = df["code"].map(MELTING_POINT_C).fillna(-20.0)
    df["pubchem_url"] = df.get("pubchem_url", pd.Series(dtype=str)).fillna("")
    return df

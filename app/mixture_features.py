from __future__ import annotations

from functools import lru_cache
from math import log
from typing import Any, Iterable

import numpy as np
import pandas as pd

from app.catalog import load_catalog
from app.chemistry import molecular_descriptors


SOLVENT_FALLBACKS = {
    "Ethanol": {
        "name": "Ethanol",
        "smiles": "CCO",
        "dielectric_constant": 24.55,
        "viscosity_mpas": 1.074,
        "boiling_point_c": 78.37,
        "flash_point_c": 13.0,
        "donor_number": 19.2,
        "oxidation_prior": 0.28,
        "reduction_prior": 0.22,
        "hazard_prior": 0.62,
        "melting_point_c": -114.1,
    },
    "Methanol": {
        "name": "Methanol",
        "smiles": "CO",
        "dielectric_constant": 32.63,
        "viscosity_mpas": 0.544,
        "boiling_point_c": 64.7,
        "flash_point_c": 11.0,
        "donor_number": 19.0,
        "oxidation_prior": 0.24,
        "reduction_prior": 0.20,
        "hazard_prior": 0.74,
        "melting_point_c": -97.6,
    },
}

DEFAULT_SOLVENT = {
    "name": "Unknown solvent",
    "smiles": "C",
    "dielectric_constant": 18.0,
    "viscosity_mpas": 1.2,
    "boiling_point_c": 130.0,
    "flash_point_c": 35.0,
    "donor_number": 16.0,
    "oxidation_prior": 0.55,
    "reduction_prior": 0.52,
    "hazard_prior": 0.50,
    "melting_point_c": -40.0,
}

SOLVENT_PROPERTIES = [
    "molecular_weight",
    "tpsa",
    "logp",
    "hbond_acceptors",
    "hbond_donors",
    "rotatable_bonds",
    "fraction_csp3",
    "ring_count",
    "dielectric_constant",
    "viscosity_mpas",
    "boiling_point_c",
    "flash_point_c",
    "donor_number",
    "oxidation_prior",
    "reduction_prior",
    "hazard_prior",
    "melting_point_c",
]

RATIO_BASES = [
    "mole_fraction_of_solvents",
    "v",
    "w",
    "m",
]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def solvent_property_table() -> pd.DataFrame:
    catalog = load_catalog().copy()
    rows = []
    for code, fallback in SOLVENT_FALLBACKS.items():
        if code in set(catalog["code"]):
            continue
        descriptors = molecular_descriptors(fallback["smiles"])
        rows.append({"code": code, **fallback, **descriptors})
    if rows:
        catalog = pd.concat([catalog, pd.DataFrame(rows)], ignore_index=True)
    catalog = catalog.set_index("code", drop=False)
    return catalog


def solvent_properties(code: str | None) -> pd.Series:
    table = solvent_property_table()
    if code and code in table.index:
        return table.loc[code]
    fallback = dict(DEFAULT_SOLVENT)
    fallback["code"] = code or "UNKNOWN"
    fallback.update(molecular_descriptors(fallback["smiles"]))
    return pd.Series(fallback)


def normalise_components(components: Iterable[tuple[str | None, float | None]]) -> list[tuple[str, float]]:
    clean = [
        (str(code), max(_as_float(ratio), 0.0))
        for code, ratio in components
        if code is not None and str(code) and _as_float(ratio) > 0
    ]
    total = sum(value for _, value in clean)
    if total <= 0:
        return []
    return [(code, value / total) for code, value in clean]


def feature_row(
    *,
    salt: str,
    temperature_c: float,
    concentration: float | None,
    concentration_unit: str | None,
    solvent_ratio_basis: str | None,
    components: list[tuple[str, float]],
    known_salts: Iterable[str],
) -> dict[str, float]:
    components = normalise_components(components)
    if not components:
        raise ValueError("At least one solvent component is required.")

    row: dict[str, float] = {
        "temperature_c": _as_float(temperature_c, 25.0),
        "concentration": _as_float(concentration, 0.0),
        "has_concentration": 1.0 if concentration is not None and not pd.isna(concentration) else 0.0,
        "component_count": float(len(components)),
        "ratio_entropy": float(-sum(frac * log(max(frac, 1e-12)) for _, frac in components)),
        "max_component_fraction": float(max(frac for _, frac in components)),
    }
    props = [(solvent_properties(code), frac) for code, frac in components]
    for prop in SOLVENT_PROPERTIES:
        values = np.array([_as_float(series.get(prop), DEFAULT_SOLVENT.get(prop, 0.0)) for series, _ in props])
        fractions = np.array([frac for _, frac in props])
        if prop == "viscosity_mpas":
            weighted = float(np.exp(np.sum(fractions * np.log(np.maximum(values, 0.05)))))
        else:
            weighted = float(np.sum(fractions * values))
        row[f"{prop}_mean"] = weighted
        row[f"{prop}_min"] = float(np.min(values))
        row[f"{prop}_max"] = float(np.max(values))
        row[f"{prop}_range"] = float(np.max(values) - np.min(values))

    for known_salt in sorted(set(known_salts)):
        row[f"salt_{known_salt}"] = 1.0 if salt == known_salt else 0.0
    basis = str(solvent_ratio_basis or "").strip()
    for ratio_basis in RATIO_BASES:
        row[f"ratio_basis_{ratio_basis}"] = 1.0 if basis == ratio_basis else 0.0
    row["ratio_basis_other"] = 1.0 if basis and basis not in RATIO_BASES else 0.0
    for unit in ["mol/kg", "mol/l", "mol/L", "m"]:
        row[f"concentration_unit_{unit}"] = 1.0 if concentration_unit == unit else 0.0
    row["concentration_unit_other"] = 1.0 if concentration_unit and concentration_unit not in {"mol/kg", "mol/l", "mol/L", "m"} else 0.0
    return row


def feature_row_from_record(record: dict[str, Any], known_salts: Iterable[str]) -> dict[str, float]:
    components = normalise_components(
        [
            (record.get("solvent_a"), record.get("ratio_a")),
            (record.get("solvent_b"), record.get("ratio_b")),
            (record.get("solvent_c"), record.get("ratio_c")),
        ]
    )
    return feature_row(
        salt=str(record.get("salt")),
        temperature_c=_as_float(record.get("temperature_c"), 25.0),
        concentration=record.get("concentration"),
        concentration_unit=record.get("concentration_unit"),
        solvent_ratio_basis=record.get("solvent_ratio_basis"),
        components=components,
        known_salts=known_salts,
    )

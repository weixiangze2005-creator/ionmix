from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
CALISOL_PATH = ROOT / "data" / "raw" / "calisol23.csv"
OUTPUT_PATH = ROOT / "data" / "mixture_experiments.csv"
REPORT_PATH = ROOT / "data" / "mixture_experiments_report.json"

THERMOML_URL = "https://trc.nist.gov/ThermoML/10.1016/j.fluid.2017.12.034.json"
THERMOML_DOI = "10.1016/j.fluid.2017.12.034"

THERMOML_SALTS = {
    2: "LiF",
    3: "LiCl",
    4: "LiBr",
    5: "LiNO3",
    6: "LiTFSI",
    7: "LiPF6",
}

THERMOML_SOLVENTS = {
    8: "AN",
    9: "DMC",
    10: "DMSO",
    11: "Ethanol",
    12: "PC",
}

SALT_ALIASES = {
    "LiN(CF3SO2)2": "LiTFSI",
}

SCHEMA_COLUMNS = [
    "source_id",
    "source_type",
    "source_name",
    "doi",
    "salt",
    "temperature_c",
    "concentration",
    "concentration_unit",
    "solvent_a",
    "ratio_a",
    "solvent_b",
    "ratio_b",
    "solvent_c",
    "ratio_c",
    "solvent_ratio_basis",
    "component_count",
    "solubility_mole_fraction",
    "fully_dissolved",
    "conductivity",
    "conductivity_unit",
    "viscosity_mpas",
    "phase_stable_24h",
    "phase_stable_7d",
    "relative_uncertainty",
    "notes",
]


def blank_row(**values: Any) -> dict[str, Any]:
    row = {column: None for column in SCHEMA_COLUMNS}
    row.update(values)
    return row


def clean_salt(value: Any) -> str:
    value = str(value).strip()
    return SALT_ALIASES.get(value, value)


def normalise_ratios(active: list[tuple[str, float]]) -> list[tuple[str, float]]:
    total = sum(max(float(value), 0.0) for _, value in active)
    if total <= 0:
        return []
    return [(solvent, 100.0 * float(value) / total) for solvent, value in active]


def calisol_rows() -> list[dict[str, Any]]:
    if not CALISOL_PATH.exists():
        return []
    df = pd.read_csv(CALISOL_PATH)
    solvent_columns = list(df.columns[7:])
    rows = []
    for index, record in df.iterrows():
        try:
            conductivity = float(record["k"])
            temperature_k = float(record["T"])
            concentration = float(record["c"])
        except (TypeError, ValueError):
            continue
        active = [
            (column, float(record[column]))
            for column in solvent_columns
            if pd.notna(record[column]) and float(record[column]) > 0
        ]
        if not (1 <= len(active) <= 3):
            continue
        ratios = normalise_ratios(active)
        if not ratios:
            continue
        padded = ratios + [(None, None)] * (3 - len(ratios))
        rows.append(
            blank_row(
                source_id=f"CALiSol-23:{index}",
                source_type="conductivity",
                source_name="CALiSol-23",
                doi=str(record.get("doi", "")).strip() or None,
                salt=clean_salt(record["salt"]),
                temperature_c=round(temperature_k - 273.15, 4),
                concentration=concentration,
                concentration_unit=str(record.get("c units", "")).strip() or None,
                solvent_a=padded[0][0],
                ratio_a=padded[0][1],
                solvent_b=padded[1][0],
                ratio_b=padded[1][1],
                solvent_c=padded[2][0],
                ratio_c=padded[2][1],
                solvent_ratio_basis=str(record.get("solvent ratio type", "")).strip() or None,
                component_count=len(ratios),
                conductivity=conductivity,
                conductivity_unit="as reported by CALiSol-23",
                fully_dissolved=True,
                notes="Parsed from public CALiSol-23 conductivity table; ratios normalised to percent.",
            )
        )
    return rows


def fetch_thermoml() -> dict[str, Any]:
    response = requests.get(
        THERMOML_URL,
        timeout=30,
        headers={"User-Agent": "ionmix-data-builder/0.1 (+https://ionmix.cn)"},
    )
    response.raise_for_status()
    return response.json()


def variable_map(dataset: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(variable["nVarNumber"]): variable for variable in dataset.get("Variable", [])}


def value_by_variable(num_values: dict[str, Any]) -> dict[int, float]:
    return {
        int(item["nVarNumber"]): float(item["nVarValue"])
        for item in num_values.get("VariableValue", [])
    }


def prop_value(num_values: dict[str, Any]) -> tuple[float | None, float | None]:
    values = num_values.get("PropertyValue", [])
    if not values:
        return None, None
    value = float(values[0]["nPropValue"])
    uncertainty = (
        values[0]
        .get("CombinedUncertainty", {})
        .get("nCombExpandUncertValue")
    )
    relative = None
    if uncertainty is not None and value > 0:
        relative = max(0.0, float(uncertainty) / value)
    return value, relative


def thermoml_rows() -> list[dict[str, Any]]:
    data = fetch_thermoml()
    rows = []
    for dataset_index, dataset in enumerate(data.get("PureOrMixtureData", []), start=1):
        components = [
            int(component["RegNum"]["nOrgNum"])
            for component in dataset.get("Component", [])
        ]
        salt_ids = [component for component in components if component in THERMOML_SALTS]
        solvent_ids = [component for component in components if component in THERMOML_SOLVENTS]
        if len(salt_ids) != 1 or not solvent_ids:
            continue
        salt = THERMOML_SALTS[salt_ids[0]]
        variables = variable_map(dataset)
        solvent_fraction_var = None
        temperature_var = None
        solvent_fraction_id = None
        for number, variable in variables.items():
            variable_id = variable.get("VariableID", {})
            variable_type = variable_id.get("VariableType", {})
            if "eTemperature" in variable_type:
                temperature_var = number
            if "eSolventComposition" in variable_type:
                solvent_fraction_var = number
                solvent_fraction_id = int(variable_id["RegNum"]["nOrgNum"])
        for value_index, num_values in enumerate(dataset.get("NumValues", []), start=1):
            values = value_by_variable(num_values)
            if temperature_var not in values:
                continue
            solubility, relative_uncertainty = prop_value(num_values)
            if solubility is None:
                continue
            if len(solvent_ids) == 1:
                ratios = [(THERMOML_SOLVENTS[solvent_ids[0]], 100.0)]
            elif len(solvent_ids) == 2 and solvent_fraction_var in values:
                first_fraction = float(values[solvent_fraction_var])
                first = THERMOML_SOLVENTS[solvent_fraction_id]
                second_id = next(item for item in solvent_ids if item != solvent_fraction_id)
                second = THERMOML_SOLVENTS[second_id]
                ratios = [(first, 100.0 * first_fraction), (second, 100.0 * (1.0 - first_fraction))]
            else:
                continue
            padded = ratios + [(None, None)] * (3 - len(ratios))
            rows.append(
                blank_row(
                    source_id=f"ThermoML:{THERMOML_DOI}:{dataset_index}:{value_index}",
                    source_type="solubility",
                    source_name="NIST ThermoML",
                    doi=THERMOML_DOI,
                    salt=salt,
                    temperature_c=round(float(values[temperature_var]) - 273.15, 4),
                    solvent_a=padded[0][0],
                    ratio_a=padded[0][1],
                    solvent_b=padded[1][0],
                    ratio_b=padded[1][1],
                    solvent_c=padded[2][0],
                    ratio_c=padded[2][1],
                    solvent_ratio_basis="mole_fraction_of_solvents",
                    component_count=len(ratios),
                    solubility_mole_fraction=solubility,
                    fully_dissolved=True,
                    relative_uncertainty=relative_uncertainty,
                    notes="Parsed from NIST ThermoML JSON for Xin et al. lithium-salt solubility study.",
                )
            )
    return rows


def main() -> None:
    rows = [*calisol_rows(), *thermoml_rows()]
    if not rows:
        raise SystemExit("No public experiment rows were parsed.")
    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    df = df.drop_duplicates(subset=["source_id"]).sort_values(
        ["source_type", "salt", "component_count", "temperature_c", "source_id"],
        kind="stable",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    report = {
        "rows": int(len(df)),
        "source_counts": df["source_name"].value_counts().to_dict(),
        "target_counts": df["source_type"].value_counts().to_dict(),
        "salt_counts": df["salt"].value_counts().to_dict(),
        "component_counts": {
            str(key): int(value)
            for key, value in df["component_count"].value_counts().sort_index().items()
        },
        "thermoml_url": THERMOML_URL,
        "thermoml_doi": THERMOML_DOI,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

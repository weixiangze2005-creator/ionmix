from __future__ import annotations

import json
import os
from math import exp, log
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.mixture_features import solvent_properties


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "oedb_auxiliary_model.joblib"
REPORT_PATH = ROOT / "models" / "oedb_auxiliary_model_report.json"

SALT_TO_IONS = {
    "LiPF6": ("Li", "PF6"),
    "LiTFSI": ("Li", "TFSI"),
    "LiFSI": ("Li", "FSI"),
    "LiBF4": ("Li", "BF4"),
    "LiClO4": ("Li", "ClO4"),
}

OEDB_PROPERTY_CODE_MAP = {
    # OEDB solvent code MA is methyl acetoacetate. The app keeps MA as
    # methyl acetate and uses MAA for methyl acetoacetate.
    "MA": "MAA",
}

LOG_AVERAGE_TARGETS = {
    "viscosity_mpas",
    "cation_diffusivity_m2_s",
    "anion_diffusivity_m2_s",
    "solvent_diffusivity_m2_s",
}


def _inverse(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(values)
    if transform == "log10":
        return np.power(10.0, values)
    return values


class OEDBAuxiliaryModel:
    def __init__(self, path: Path = MODEL_PATH):
        self.bundle: dict[str, Any] | None = None
        setting = os.getenv("IONMIX_OEDB_MODEL", "").lower()
        self.disabled = setting in {"0", "false", "off", "disabled"}
        if path.exists() and not self.disabled:
            self.bundle = joblib.load(path)

    @property
    def available(self) -> bool:
        return self.bundle is not None

    @property
    def metrics(self) -> dict:
        if self.bundle:
            return {
                "source": self.bundle["source"],
                "training_summary": self.bundle["training_summary"],
                "target_metrics": self.bundle["target_metrics"],
            }
        if REPORT_PATH.exists():
            data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            data["runtime_status"] = "disabled" if self.disabled else "model_not_loaded"
            return data
        return {"runtime_status": "disabled" if self.disabled else "model_not_trained"}

    @property
    def supported_salts(self) -> list[str]:
        if not self.bundle:
            return []
        supported = []
        for salt, (cation, anion) in SALT_TO_IONS.items():
            if cation in self.bundle["cations"] and anion in self.bundle["anions"]:
                supported.append(salt)
        return supported

    @property
    def supported_solvents(self) -> list[str]:
        return [] if not self.bundle else list(self.bundle["solvents"])

    def _feature_row(self, cation: str, anion: str, solvent: str, concentration: float) -> dict[str, float]:
        props = solvent_properties(OEDB_PROPERTY_CODE_MAP.get(solvent, solvent))
        row = {
            "concentration_mol_kg": float(concentration),
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
        for known in self.bundle["cations"]:
            row[f"cation_{known}"] = 1.0 if cation == known else 0.0
        for known in self.bundle["anions"]:
            row[f"anion_{known}"] = 1.0 if anion == known else 0.0
        for known in self.bundle["solvents"]:
            row[f"solvent_{known}"] = 1.0 if solvent == known else 0.0
        return row

    def predict_many(self, requests: list[dict]) -> list[dict]:
        outputs = [{"available": False, "coverage": 0.0} for _ in requests]
        if not self.bundle:
            return outputs

        rows = []
        component_refs: list[tuple[int, float]] = []
        coverage_by_request: dict[int, float] = {}
        normalised_by_request: dict[int, list[tuple[str, float]]] = {}
        for request_index, request in enumerate(requests):
            salt = request["salt"]
            if salt not in SALT_TO_IONS:
                continue
            cation, anion = SALT_TO_IONS[salt]
            if cation not in self.bundle["cations"] or anion not in self.bundle["anions"]:
                continue
            components = request["components"]
            total = sum(max(float(fraction), 0.0) for _, fraction in components)
            normalised = [
                (solvent, max(float(fraction), 0.0) / total)
                for solvent, fraction in components
                if total > 0 and fraction > 0
            ]
            normalised_by_request[request_index] = normalised
            coverage = 0.0
            for solvent, fraction in normalised:
                if solvent not in self.bundle["solvents"]:
                    continue
                coverage += float(fraction)
                rows.append(
                    self._feature_row(
                        cation,
                        anion,
                        solvent,
                        float(request.get("concentration") or 1.0),
                    )
                )
                component_refs.append((request_index, float(fraction)))
            if coverage > 0:
                coverage_by_request[request_index] = min(1.0, coverage)

        if not rows:
            return outputs

        x = pd.DataFrame(rows).reindex(columns=self.bundle["feature_columns"], fill_value=0.0).fillna(0.0)
        target_predictions = {}
        for target, model in self.bundle["models"].items():
            spec = self.bundle["target_specs"][target]
            target_predictions[target] = np.maximum(
                0.0,
                _inverse(model.predict(x), spec["transform"]),
            )

        grouped: dict[int, list[tuple[float, dict[str, float]]]] = {}
        for row_index, (request_index, weight) in enumerate(component_refs):
            values = {
                target: float(predictions[row_index])
                for target, predictions in target_predictions.items()
            }
            grouped.setdefault(request_index, []).append((weight, values))

        for request_index, per_component in grouped.items():
            coverage = coverage_by_request.get(request_index, 0.0)
            if not per_component:
                continue
            prediction: dict[str, float | bool] = {"available": True, "coverage": coverage}
            target_names = self.bundle["models"].keys()
            for target in target_names:
                values = [(weight, component[target]) for weight, component in per_component]
                weight_total = sum(weight for weight, _ in values)
                if weight_total <= 0:
                    continue
                if target in LOG_AVERAGE_TARGETS:
                    mixed = exp(
                        sum(weight * log(max(value, 1e-14)) for weight, value in values)
                        / weight_total
                    )
                else:
                    mixed = sum(weight * value for weight, value in values) / weight_total
                prediction[target] = float(mixed)
            confidence = 100.0 * (0.18 + 0.40 * coverage)
            if requests[request_index]["salt"] in self.supported_salts:
                confidence += 12.0
            if len(normalised_by_request.get(request_index, [])) > 1:
                confidence -= 10.0  # single-solvent MD model mixed by rule, not direct mixture labels
            prediction["confidence"] = float(np.clip(confidence, 20.0, 70.0))
            prediction["source"] = "OEDB-MD"
            outputs[request_index] = prediction
        return outputs

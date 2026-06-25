from __future__ import annotations

import json
import os
from math import exp
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.mixture_features import feature_row


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mixture_model.joblib"
REPORT_PATH = ROOT / "models" / "mixture_model_report.json"


class MixturePropertyModel:
    def __init__(self, path: Path = MODEL_PATH):
        self.bundle: dict[str, Any] | None = None
        setting = os.getenv("IONMIX_MIXTURE_MODEL", "").lower()
        disabled = setting in {"0", "false", "off", "disabled"}
        if path.exists() and not disabled:
            self.bundle = joblib.load(path)
        self.disabled = disabled

    @property
    def available(self) -> bool:
        return self.bundle is not None

    @property
    def metrics(self) -> dict:
        if self.bundle:
            return {
                "conductivity": self.bundle["conductivity"]["metrics"],
                "solubility": self.bundle["solubility"]["metrics"],
                "training_summary": self.bundle["training_summary"],
            }
        if REPORT_PATH.exists():
            data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            data["runtime_status"] = "disabled" if self.disabled else "model_not_loaded"
            return data
        return {"runtime_status": "disabled" if self.disabled else "model_not_trained"}

    @property
    def known_salts(self) -> list[str]:
        return [] if not self.bundle else list(self.bundle["known_salts"])

    @property
    def conductivity_supported_salts(self) -> list[str]:
        if not self.bundle:
            return []
        return list(self.bundle["conductivity"].get("supported_salts", []))

    @property
    def solubility_supported_salts(self) -> list[str]:
        if not self.bundle:
            return []
        return list(self.bundle["solubility"].get("supported_salts", []))

    def _matrix(self, requests: list[dict], feature_columns: list[str]) -> pd.DataFrame:
        rows = [
            feature_row(
                salt=request["salt"],
                temperature_c=request["temperature_c"],
                concentration=request.get("concentration"),
                concentration_unit=request.get("concentration_unit"),
                solvent_ratio_basis=request.get("solvent_ratio_basis"),
                components=request["components"],
                known_salts=self.known_salts,
            )
            for request in requests
        ]
        return pd.DataFrame(rows).reindex(columns=feature_columns, fill_value=0.0).fillna(0.0)

    def _domain_scores(self, x: pd.DataFrame) -> np.ndarray:
        if not self.bundle:
            return np.zeros(len(x))
        mean = pd.Series(self.bundle["feature_mean"]).reindex(x.columns).fillna(0.0)
        std = pd.Series(self.bundle["feature_std"]).reindex(x.columns).replace(0, 1.0).fillna(1.0)
        distances = np.sqrt(np.mean(np.square((x - mean) / std), axis=1))
        return np.exp(-np.asarray(distances, dtype=float) / 3.0)

    def predict_many(self, requests: list[dict]) -> list[dict]:
        if not self.bundle or not requests:
            return []
        outputs = [
            {
                "conductivity": None,
                "conductivity_score": None,
                "solubility_mole_fraction": None,
                "solubility_score": None,
                "confidence": 0.0,
                "used_targets": [],
                "confidence_factors": {},
            }
            for _ in requests
        ]

        conductivity_requests = [
            (index, request)
            for index, request in enumerate(requests)
            if request["salt"] in self.conductivity_supported_salts
            and self.bundle["conductivity"]["model"] is not None
        ]
        if conductivity_requests:
            indices, subset = zip(*conductivity_requests)
            feature_columns = self.bundle["conductivity"]["feature_columns"]
            x = self._matrix(list(subset), feature_columns)
            predictions = np.expm1(self.bundle["conductivity"]["model"].predict(x))
            domain_scores = self._domain_scores(x)
            for index, prediction, domain_score in zip(indices, predictions, domain_scores):
                conductivity = max(0.0, float(prediction))
                outputs[index]["conductivity"] = conductivity
                outputs[index]["conductivity_score"] = float(np.clip(conductivity / 15.0, 0.0, 1.0))
                outputs[index]["used_targets"].append("conductivity")
                outputs[index]["confidence_factors"]["mixture_conductivity_domain"] = round(float(domain_score), 4)

        solubility_requests = [
            (index, request)
            for index, request in enumerate(requests)
            if request["salt"] in self.solubility_supported_salts
            and self.bundle["solubility"]["model"] is not None
        ]
        if solubility_requests:
            indices, subset = zip(*solubility_requests)
            feature_columns = self.bundle["solubility"]["feature_columns"]
            x = self._matrix(list(subset), feature_columns)
            log_predictions = self.bundle["solubility"]["model"].predict(x)
            mole_fractions = np.clip(np.power(10.0, log_predictions), 1e-8, 0.75)
            domain_scores = self._domain_scores(x)
            for index, mole_fraction, domain_score in zip(indices, mole_fractions, domain_scores):
                mole_fraction = float(mole_fraction)
                outputs[index]["solubility_mole_fraction"] = mole_fraction
                outputs[index]["solubility_score"] = float(np.clip((np.log10(mole_fraction) + 4.0) / 3.5, 0.0, 1.0))
                outputs[index]["used_targets"].append("solubility")
                outputs[index]["confidence_factors"]["mixture_solubility_domain"] = round(float(domain_score), 4)

        for output in outputs:
            used = output["used_targets"]
            if not used:
                continue
            factors = list(output["confidence_factors"].values())
            domain = float(np.mean(factors)) if factors else 0.3
            target_bonus = 0.18 if len(used) > 1 else 0.0
            confidence = 100.0 * (0.28 + 0.42 * domain + target_bonus)
            output["confidence"] = float(np.clip(confidence, 25.0, 82.0))
            output["confidence_factors"]["mixture_model_target_count"] = len(used)
            output["confidence_factors"]["mixture_model_domain_mean"] = round(domain, 4)
        return outputs

    def confidence_for_unlabelled(self, salt: str, components: list[tuple[str, float]]) -> float:
        if not self.bundle:
            return 0.0
        request = {
            "salt": salt,
            "temperature_c": 25.0,
            "concentration": None,
            "concentration_unit": None,
            "solvent_ratio_basis": "v",
            "components": components,
        }
        x = self._matrix([request], self.bundle["feature_columns"])
        return float(100.0 * (0.20 + 0.35 * exp(-np.sqrt(np.mean(np.square(x.to_numpy()))) / 25.0)))

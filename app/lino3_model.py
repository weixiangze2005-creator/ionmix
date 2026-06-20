from __future__ import annotations

from math import exp
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "lino3_solubility_model.joblib"


class LiNO3SolubilityModel:
    def __init__(self, path: Path = MODEL_PATH):
        self.bundle = joblib.load(path) if path.exists() else None

    @property
    def available(self) -> bool:
        return self.bundle is not None

    @property
    def metrics(self) -> dict:
        return {} if not self.bundle else dict(self.bundle["metrics"])

    def _mixture_row(self, a: pd.Series, b: pd.Series, fraction_a: float, temperature_c: float) -> dict:
        fraction_b = 1.0 - fraction_a
        row = {"temperature_c": temperature_c}
        for feature in self.bundle["features"]:
            if feature == "temperature_c":
                continue
            if feature == "viscosity_mpas":
                row[feature] = float(
                    np.exp(
                        fraction_a * np.log(max(float(a[feature]), 0.05))
                        + fraction_b * np.log(max(float(b[feature]), 0.05))
                    )
                )
            else:
                row[feature] = float(fraction_a * a[feature] + fraction_b * b[feature])
        return row

    def predict_many(self, requests: list[dict]) -> list[dict]:
        if not self.bundle or not requests:
            return []
        rows = [
            self._mixture_row(
                request["a"], request["b"], request["fraction_a"], request["temperature_c"]
            )
            for request in requests
        ]
        x = pd.DataFrame(rows, columns=self.bundle["features"])
        x_scaled = self.bundle["scaler"].transform(x)
        model = self.bundle["model"]
        tree_predictions = np.vstack([tree.predict(x_scaled) for tree in model.estimators_])
        tree_means = np.mean(tree_predictions, axis=0)
        log_predictions = self.bundle["ridge_model"].predict(x_scaled)
        training_log_y = np.asarray(self.bundle["training_log_y"], dtype=float)
        log_predictions = np.clip(
            log_predictions,
            float(training_log_y.min() - 0.35),
            float(training_log_y.max() + 0.20),
        )
        mole_fractions = np.clip(np.power(10.0, log_predictions), 1e-7, 0.65)
        ensemble_stds = np.std(tree_predictions, axis=0) + 0.5 * np.abs(
            log_predictions - tree_means
        )
        training_x = np.asarray(self.bundle["training_x_scaled"], dtype=float)
        known_solvents = set(self.bundle["training_solvents"])
        mean_quality = 1.0 - float(np.mean(self.bundle["relative_uncertainties"]))
        outputs = []
        for index, request in enumerate(requests):
            distances = np.sqrt(np.mean(np.square(training_x - x_scaled[index]), axis=1))
            nearest_distance = float(np.min(distances))
            domain_score = exp(-nearest_distance / 1.35)
            local_density = float(np.mean(distances < 1.5))
            density_score = min(1.0, local_density / 0.28)
            ensemble_std = float(ensemble_stds[index])
            ensemble_score = exp(-ensemble_std / 0.85)

            temperature_c = request["temperature_c"]
            t_min = self.metrics["temperature_min_c"]
            t_max = self.metrics["temperature_max_c"]
            if t_min <= temperature_c <= t_max:
                temperature_score = 1.0
            else:
                temperature_score = exp(
                    -min(abs(temperature_c - t_min), abs(temperature_c - t_max)) / 25.0
                )
            fraction_a = request["fraction_a"]
            a, b = request["a"], request["b"]
            component_coverage = (
                fraction_a * (1.0 if a["code"] in known_solvents else 0.35)
                + (1.0 - fraction_a) * (1.0 if b["code"] in known_solvents else 0.35)
            )
            mole_fraction = float(mole_fractions[index])
            ml_score = float(np.clip((np.log10(mole_fraction) + 4.0) / 3.5, 0.0, 1.0))
            agreement_score = exp(-abs(ml_score - request["heuristic_score"]) / 0.35)
            confidence = 100.0 * (
                0.08
                + 0.22 * domain_score
                + 0.15 * ensemble_score
                + 0.13 * density_score
                + 0.12 * temperature_score
                + 0.14 * component_coverage
                + 0.10 * agreement_score
                + 0.06 * mean_quality
            )
            raw_confidence = max(confidence, 18.0)
            # Smooth saturation preserves ranking differences instead of
            # forcing many candidates onto one identical hard cap.
            confidence = float(18.0 + 60.0 * np.tanh((raw_confidence - 18.0) / 60.0))
            outputs.append(
                {
                    "mole_fraction": mole_fraction,
                    "score": ml_score,
                    "confidence": confidence,
                    "ensemble_std_log10": ensemble_std,
                    "domain_distance": nearest_distance,
                    "local_density": local_density,
                    "agreement_score": agreement_score,
                    "confidence_factors": {
                        "domain_similarity": round(domain_score, 4),
                        "ensemble_agreement": round(ensemble_score, 4),
                        "local_data_density": round(density_score, 4),
                        "temperature_coverage": round(temperature_score, 4),
                        "component_coverage": round(component_coverage, 4),
                        "physics_model_agreement": round(agreement_score, 4),
                    },
                }
            )
        return outputs

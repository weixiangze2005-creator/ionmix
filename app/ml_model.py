from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from math import exp, log1p

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "conductivity_model.joblib"
REPORT_PATH = ROOT / "models" / "training_report.json"


class ConductivityModel:
    def __init__(self, path: Path = MODEL_PATH):
        self.bundle: dict[str, Any] | None = None
        runtime_setting = os.getenv("IONMIX_CONDUCTIVITY_MODEL", "").lower()
        explicitly_enabled = runtime_setting in {"1", "true", "on", "enabled"}
        explicitly_disabled = runtime_setting in {
            "0",
            "false",
            "off",
            "disabled",
        }
        # Loading the full ExtraTrees conductivity model can exceed Render's
        # free 512 MB memory limit. Keep production lightweight by default;
        # local scripts/tests explicitly opt into the full model.
        self.disabled = explicitly_disabled or not explicitly_enabled
        if path.exists() and not self.disabled:
            self.bundle = joblib.load(path)

    @property
    def available(self) -> bool:
        return self.bundle is not None

    @property
    def supported_salts(self) -> list[str]:
        return [] if not self.bundle else list(self.bundle["supported_salts"])

    @property
    def supported_solvents(self) -> list[str]:
        return [] if not self.bundle else list(self.bundle["solvent_columns"])

    @property
    def metrics(self) -> dict[str, float]:
        if self.bundle:
            return dict(self.bundle["metrics"])
        if REPORT_PATH.exists():
            import json

            metrics = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
            metrics["runtime_status"] = (
                "disabled_for_memory_limit" if self.disabled else "model_not_loaded"
            )
            return metrics
        return {"runtime_status": "disabled_for_memory_limit"} if self.disabled else {}

    def predict(
        self,
        salt: str,
        solvent_a: str,
        solvent_b: str,
        fraction_a: float,
        temperature_k: float,
        concentration: float,
        concentration_unit: str = "mol/kg",
        ratio_type: str = "v",
    ) -> tuple[float | None, float | None]:
        if not self.bundle:
            return None, None
        if salt not in self.supported_salts:
            return None, None
        if solvent_a not in self.supported_solvents or solvent_b not in self.supported_solvents:
            return None, None

        row = {col: 0.0 for col in self.bundle["feature_columns"]}
        row["T"] = float(temperature_k)
        row["c"] = float(concentration)
        row[solvent_a] = float(fraction_a)
        row[solvent_b] = float(1.0 - fraction_a)
        for prefix, value in (
            ("salt_", salt),
            ("unit_", concentration_unit),
            ("ratio_", ratio_type),
        ):
            key = prefix + value
            if key in row:
                row[key] = 1.0

        x = pd.DataFrame([row], columns=self.bundle["feature_columns"])
        model = self.bundle["model"]
        pred = max(0.0, float(model.predict(x)[0]))
        tree_x = x.to_numpy()
        tree_preds = np.array([tree.predict(tree_x)[0] for tree in model.estimators_], dtype=float)
        uncertainty = float(np.std(tree_preds))
        return pred, uncertainty

    def predict_many(self, requests: list[dict]) -> list[tuple[float, float]]:
        if not self.bundle or not requests:
            return []
        rows = []
        for request in requests:
            row = {col: 0.0 for col in self.bundle["feature_columns"]}
            row["T"] = float(request["temperature_k"])
            row["c"] = float(request["concentration"])
            row[request["solvent_a"]] = float(request["fraction_a"])
            row[request["solvent_b"]] = float(1.0 - request["fraction_a"])
            for prefix, value in (
                ("salt_", request["salt"]),
                ("unit_", request["concentration_unit"]),
                ("ratio_", request["ratio_type"]),
            ):
                key = prefix + value
                if key in row:
                    row[key] = 1.0
            rows.append(row)
        x = pd.DataFrame(rows, columns=self.bundle["feature_columns"])
        model = self.bundle["model"]
        predictions = np.maximum(0.0, model.predict(x))
        tree_x = x.to_numpy()
        per_tree = np.vstack([tree.predict(tree_x) for tree in model.estimators_])
        uncertainties = np.std(per_tree, axis=0)
        return [
            (float(pred), float(unc), self.confidence(request, float(unc)))
            for request, pred, unc in zip(requests, predictions, uncertainties)
        ]

    def confidence(self, request: dict, uncertainty: float) -> float:
        salt = request["salt"]
        stats = self.bundle["salt_ranges"][salt]
        target_std = max(float(stats["target_std"]), 0.5)
        ensemble_score = exp(-uncertainty / target_std)

        salt_counts = self.bundle["salt_counts"]
        max_salt_count = max(salt_counts.values())
        salt_density = log1p(salt_counts.get(salt, 0)) / log1p(max_salt_count)

        solvent_counts = self.bundle["solvent_counts"].get(salt, {})
        count_a = solvent_counts.get(request["solvent_a"], 0)
        count_b = solvent_counts.get(request["solvent_b"], 0)
        solvent_density = min(1.0, (min(count_a, count_b) / 120.0) ** 0.5)

        pair_key = "|".join(sorted((request["solvent_a"], request["solvent_b"])))
        pair_count = self.bundle["pair_counts"].get(salt, {}).get(pair_key, 0)
        pair_evidence = min(1.0, log1p(pair_count) / log1p(80))

        def range_score(value: float, lower: float, upper: float) -> float:
            if lower <= value <= upper:
                return 1.0
            scale = max(upper - lower, 1.0)
            distance = lower - value if value < lower else value - upper
            return exp(-distance / scale)

        domain_score = 0.5 * range_score(
            request["temperature_k"], stats["temperature_min"], stats["temperature_max"]
        ) + 0.5 * range_score(
            request["concentration"], stats["concentration_min"], stats["concentration_max"]
        )
        confidence = 100.0 * (
            0.08
            + 0.27 * ensemble_score
            + 0.18 * salt_density
            + 0.16 * solvent_density
            + 0.16 * pair_evidence
            + 0.15 * domain_score
        )
        return float(np.clip(confidence, 22.0, 95.0))

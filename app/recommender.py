from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, log, sqrt

import numpy as np
import pandas as pd

from app.catalog import load_catalog
from app.lino3_model import LiNO3SolubilityModel
from app.ml_model import ConductivityModel


SALT_ALIASES = {
    "lino3": "LiNO3",
    "硝酸锂": "LiNO3",
    "lipf6": "LiPF6",
    "六氟磷酸锂": "LiPF6",
    "litfsi": "LiTFSI",
    "双三氟甲磺酰亚胺锂": "LiTFSI",
    "lifsi": "LiFSI",
    "libf4": "LiBF4",
    "liclo4": "LiClO4",
}

SALT_SMILES = {
    "LiNO3": "[Li+].[O-][N+](=O)[O-]",
    "LiPF6": "[Li+].F[P-](F)(F)(F)(F)F",
    "LiTFSI": "[Li+].[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "LiFSI": "[Li+].[N-](S(=O)(=O)F)S(=O)(=O)F",
    "LiBF4": "[Li+].F[B-](F)(F)F",
    "LiClO4": "[Li+].[O-]Cl(=O)(=O)=O",
}

DEFAULT_WEIGHTS = {
    "solubility": 0.35,
    "conductivity": 0.30,
    "safety": 0.12,
    "stability": 0.15,
    "low_temperature": 0.08,
}

LINO3_AFFINITY = {
    "DMSO": 1.00,
    "DMF": 0.94,
    "NMP": 0.92,
    "4-Glyme": 0.88,
    "3-Glyme": 0.86,
    "2-Glyme": 0.83,
    "DME": 0.78,
    "THF": 0.70,
    "DOL": 0.68,
    "AN": 0.66,
    "Sulfolane": 0.64,
    "GBL": 0.62,
    "PC": 0.46,
    "EC": 0.32,
    "FEC": 0.34,
    "TEP": 0.48,
    "TMP": 0.48,
    "DMC": 0.34,
    "EMC": 0.32,
    "DEC": 0.30,
    "EA": 0.30,
    "MA": 0.28,
    "BTFE": 0.10,
    "TTE": 0.08,
}


def canonical_salt(value: str) -> str:
    cleaned = value.strip()
    return SALT_ALIASES.get(cleaned.lower(), cleaned)


def _minmax(series: pd.Series, value: float, inverse: bool = False) -> float:
    lo, hi = float(series.quantile(0.05)), float(series.quantile(0.95))
    score = (float(value) - lo) / max(hi - lo, 1e-9)
    score = float(np.clip(score, 0.0, 1.0))
    return 1.0 - score if inverse else score


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


@dataclass
class RecommendationOptions:
    salt: str
    temperature_c: float = 25.0
    concentration: float = 1.0
    concentration_unit: str = "mol/kg"
    min_flash_point_c: float = -20.0
    max_mixture_viscosity: float = 6.0
    exclude_high_hazard: bool = True
    application: str = "lithium_metal"
    top_k: int = 10
    weights: dict[str, float] | None = None
    allow_relaxed_fallback: bool = True


class FormulationRecommender:
    def __init__(self):
        self.catalog = load_catalog()
        self.model = ConductivityModel()
        self.lino3_model = LiNO3SolubilityModel()
        self.scales = {
            column: (
                float(self.catalog[column].quantile(0.05)),
                float(self.catalog[column].quantile(0.95)),
            )
            for column in (
                "dielectric_constant",
                "donor_number",
                "boiling_point_c",
                "viscosity_mpas",
                "flash_point_c",
            )
        }

    def _scale(self, column: str, value: float, inverse: bool = False) -> float:
        lo, hi = self.scales[column]
        score = float(np.clip((float(value) - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
        return 1.0 - score if inverse else score

    def _pair_is_allowed(
        self, a: pd.Series, b: pd.Series, options: RecommendationOptions, relaxed: bool = False
    ) -> bool:
        if relaxed:
            return True
        if options.exclude_high_hazard and max(a.hazard_prior, b.hazard_prior) >= 0.80:
            return False
        if max(a.viscosity_mpas, b.viscosity_mpas) > options.max_mixture_viscosity * 4:
            return False
        # Exclude two co-solvents that both have very low polarity; one component
        # must carry most of the salt-solvation burden.
        if max(a.dielectric_constant, b.dielectric_constant) < 12 and max(a.donor_number, b.donor_number) < 22:
            return False
        return True

    def _evaluate(
        self,
        salt: str,
        a: pd.Series,
        b: pd.Series,
        fraction_a: float,
        options: RecommendationOptions,
        relaxed: bool = False,
    ) -> dict:
        fraction_b = 1.0 - fraction_a
        dielectric = fraction_a * a.dielectric_constant + fraction_b * b.dielectric_constant
        donor = fraction_a * a.donor_number + fraction_b * b.donor_number
        viscosity = exp(
            fraction_a * log(max(a.viscosity_mpas, 0.05))
            + fraction_b * log(max(b.viscosity_mpas, 0.05))
        )
        # Flash point of a volatile binary mixture is not linear. This
        # conservative proxy stays much closer to the more volatile component.
        lower_flash = min(a.flash_point_c, b.flash_point_c)
        flash_spread = abs(a.flash_point_c - b.flash_point_c)
        volatile_fraction = fraction_a if a.flash_point_c == lower_flash else fraction_b
        flash = lower_flash + flash_spread * max(0.0, 1.0 - volatile_fraction) * 0.18
        tpsa_density = (
            fraction_a * a.tpsa / a.molecular_weight
            + fraction_b * b.tpsa / b.molecular_weight
        )

        dielectric_score = self._scale("dielectric_constant", dielectric)
        donor_score = self._scale("donor_number", donor)
        polarity_score = float(np.clip(tpsa_density / 0.45, 0.0, 1.0))

        # LiNO3 has high lattice energy and no direct CALiSol labels, so the
        # fallback emphasizes charge separation and Lewis-basic coordination.
        if salt == "LiNO3":
            affinity = (
                fraction_a * LINO3_AFFINITY.get(a["code"], 0.35)
                + fraction_b * LINO3_AFFINITY.get(b["code"], 0.35)
            )
            strong_solvator_fraction = (
                fraction_a if LINO3_AFFINITY.get(a["code"], 0.35) >= 0.78 else 0.0
            ) + (
                fraction_b if LINO3_AFFINITY.get(b["code"], 0.35) >= 0.78 else 0.0
            )
            solubility = _sigmoid(
                1.55 * dielectric_score
                + 2.65 * donor_score
                + 2.7 * affinity
                + 0.5 * polarity_score
                + 0.8 * min(strong_solvator_fraction, 0.55)
                - 3.45
            )
        else:
            solubility = _sigmoid(2.4 * dielectric_score + 2.0 * donor_score + polarity_score - 2.7)

        transport_proxy = sqrt(max(dielectric, 1.0)) * donor / max(viscosity, 0.05)
        transport_score = float(np.clip(log1p(transport_proxy) / 7.2, 0.0, 1.0))

        conductivity_score = transport_score

        oxidation = fraction_a * a.oxidation_prior + fraction_b * b.oxidation_prior
        reduction = fraction_a * a.reduction_prior + fraction_b * b.reduction_prior
        if options.application == "high_voltage":
            stability = 0.75 * oxidation + 0.25 * reduction
        elif options.application == "lithium_metal":
            stability = 0.25 * oxidation + 0.75 * reduction
        else:
            stability = 0.50 * oxidation + 0.50 * reduction

        volatility_penalty = self._scale(
            "boiling_point_c", min(a.boiling_point_c, b.boiling_point_c), inverse=True
        )
        low_temp = 0.65 * self._scale("viscosity_mpas", viscosity, inverse=True) + 0.35 * volatility_penalty
        safety = (
            0.55 * self._scale("flash_point_c", flash)
            + 0.45 * (1.0 - (fraction_a * a.hazard_prior + fraction_b * b.hazard_prior))
        )
        violations = []
        constraint_penalty = 0.0
        if flash < options.min_flash_point_c:
            shortfall = options.min_flash_point_c - flash
            violations.append(f"估算闪点低于要求 {shortfall:.1f} °C")
            constraint_penalty += min(0.28, 0.04 + shortfall / 500.0)
        if viscosity > options.max_mixture_viscosity:
            excess = viscosity - options.max_mixture_viscosity
            violations.append(f"估算黏度高于要求 {excess:.2f} mPa·s")
            constraint_penalty += min(0.25, 0.04 + excess / 40.0)
        if options.exclude_high_hazard and max(a.hazard_prior, b.hazard_prior) >= 0.80:
            violations.append("包含高危溶剂先验")
            constraint_penalty += 0.18
        if violations and not relaxed:
            return {}

        # High-melting solvents are useful in blends, but dominant fractions
        # near or below their melting point are poor room-temperature starts.
        phase_penalty = 0.0
        for solvent, fraction in ((a, fraction_a), (b, fraction_b)):
            if options.temperature_c < solvent.melting_point_c + 3:
                phase_penalty += max(0.0, fraction - 0.35) * 0.55

        weights = {
            key: max(0.0, float((options.weights or {}).get(key, default)))
            for key, default in DEFAULT_WEIGHTS.items()
        }
        weight_sum = sum(weights.values())
        if weight_sum <= 1e-12:
            weights = {key: 1.0 for key in DEFAULT_WEIGHTS}
            weight_sum = float(len(weights))
        total = (
            weights["solubility"] * solubility
            + weights["conductivity"] * conductivity_score
            + weights["safety"] * safety
            + weights["stability"] * stability
            + weights["low_temperature"] * low_temp
        ) / weight_sum
        total -= phase_penalty
        total -= constraint_penalty

        # Avoid nominally "optimal" 50/50 results caused only by smooth linear
        # mixing: reward complementary roles, but only modestly.
        complementary = abs(a.dielectric_constant - b.dielectric_constant) / 90.0
        total += min(0.035, 0.035 * complementary)

        catalog_coverage = int(a.training_code in self.model.supported_solvents) + int(
            b.training_code in self.model.supported_solvents
        )
        confidence = 0.42 + 0.06 * catalog_coverage
        basis = "分子描述符 + 溶剂化/传输启发式（外推）"
        if violations:
            confidence *= max(0.35, 1.0 - constraint_penalty * 1.8)
            basis += " · 放宽约束备选"

        reasons = []
        if dielectric >= 25:
            reasons.append("较高介电环境有利于盐的电荷分离")
        if donor >= 20:
            reasons.append("较强供电子能力有利于 Li⁺ 配位溶剂化")
        if viscosity <= 1.2:
            reasons.append("混合黏度较低，有利于离子迁移")
        if stability >= 0.72:
            reasons.append("与所选电池场景的稳定性先验较匹配")
        if flash >= 50:
            reasons.append("估算闪点较高，挥发/易燃风险相对较低")

        return {
            "solvent_a": a["code"],
            "solvent_a_name": a["name"],
            "solvent_b": b["code"],
            "solvent_b_name": b["name"],
            "ratio_a": round(100 * fraction_a),
            "ratio_b": round(100 * fraction_b),
            "score": round(float(np.clip(total, 0.0, 1.0)) * 100, 1),
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)) * 100, 1),
            "basis": basis,
            "predicted_conductivity": None,
            "conductivity_uncertainty": None,
            "predicted_solubility_mole_fraction": None,
            "confidence_factors": {},
            "properties": {
                "solubility_score": round(solubility * 100, 1),
                "conductivity_score": round(conductivity_score * 100, 1),
                "stability_score": round(stability * 100, 1),
                "safety_score": round(safety * 100, 1),
                "low_temperature_score": round(low_temp * 100, 1),
                "dielectric_constant": round(dielectric, 2),
                "viscosity_mpas": round(viscosity, 3),
                "flash_point_c": round(flash, 1),
                "donor_number": round(donor, 2),
            },
            "reasons": reasons[:3],
            "constraint_status": "relaxed" if violations else "feasible",
            "constraint_violations": violations,
            "sources": [url for url in (a.pubchem_url, b.pubchem_url) if url],
            "_transport_score": transport_score,
            "_heuristic_solubility_score": solubility,
            "_solubility_weight": weights["solubility"] / weight_sum,
            "_conductivity_weight": weights["conductivity"] / weight_sum,
            "_constraint_penalty": constraint_penalty,
            "_lino3_input": {
                "a": a,
                "b": b,
                "fraction_a": fraction_a,
                "temperature_c": options.temperature_c,
                "heuristic_score": solubility,
            },
            "_ml_input": {
                "salt": salt,
                "solvent_a": a.training_code,
                "solvent_b": b.training_code,
                "fraction_a": fraction_a,
                "temperature_k": options.temperature_c + 273.15,
                "concentration": options.concentration,
                "concentration_unit": options.concentration_unit,
                "ratio_type": "v",
            },
        }

    def _generate_candidates(
        self, salt: str, options: RecommendationOptions, relaxed: bool
    ) -> list[dict]:
        candidates = []
        for (_, a), (_, b) in combinations(self.catalog.iterrows(), 2):
            if not self._pair_is_allowed(a, b, options, relaxed=relaxed):
                continue
            for ratio in np.arange(0.10, 0.91, 0.05):
                result = self._evaluate(salt, a, b, float(ratio), options, relaxed=relaxed)
                if result:
                    candidates.append(result)
        return candidates

    @staticmethod
    def _select_diverse(candidates: list[dict], top_k: int) -> list[dict]:
        """Keep the list useful instead of letting one solvent occupy every row."""
        best_per_pair = {}
        for item in sorted(candidates, key=lambda value: value["score"], reverse=True):
            pair = tuple(sorted((item["solvent_a"], item["solvent_b"])))
            best_per_pair.setdefault(pair, item)
        pool = list(best_per_pair.values())
        pool.sort(key=lambda value: value["score"], reverse=True)

        selected = []
        solvent_uses: dict[str, int] = {}
        max_uses = max(2, int(np.ceil(top_k * 0.4)))
        for item in pool:
            a, b = item["solvent_a"], item["solvent_b"]
            if solvent_uses.get(a, 0) >= max_uses or solvent_uses.get(b, 0) >= max_uses:
                continue
            selected.append(item)
            solvent_uses[a] = solvent_uses.get(a, 0) + 1
            solvent_uses[b] = solvent_uses.get(b, 0) + 1
            if len(selected) >= top_k:
                return selected

        selected_pairs = {
            tuple(sorted((item["solvent_a"], item["solvent_b"]))) for item in selected
        }
        for item in pool:
            pair = tuple(sorted((item["solvent_a"], item["solvent_b"])))
            if pair in selected_pairs:
                continue
            selected.append(item)
            selected_pairs.add(pair)
            if len(selected) >= top_k:
                break
        return selected

    def recommend(self, options: RecommendationOptions) -> dict:
        salt = canonical_salt(options.salt)
        candidates = self._generate_candidates(salt, options, relaxed=False)
        feasible_count = len(candidates)
        used_relaxed_fallback = False
        if len(self._select_diverse(candidates, options.top_k)) < options.top_k and options.allow_relaxed_fallback:
            relaxed_candidates = self._generate_candidates(salt, options, relaxed=True)
            existing = {
                (item["solvent_a"], item["solvent_b"], item["ratio_a"]) for item in candidates
            }
            candidates.extend(
                item
                for item in relaxed_candidates
                if (item["solvent_a"], item["solvent_b"], item["ratio_a"]) not in existing
            )
            used_relaxed_fallback = any(
                item["constraint_status"] == "relaxed" for item in candidates
            )

        if salt == "LiNO3" and self.lino3_model.available:
            lino3_predictions = self.lino3_model.predict_many(
                [item["_lino3_input"] for item in candidates]
            )
            for item, prediction in zip(candidates, lino3_predictions):
                score_delta = item["_solubility_weight"] * (
                    prediction["score"] - item["_heuristic_solubility_score"]
                ) * 100
                item["score"] = round(
                    float(np.clip(item["score"] + score_delta, 0.0, 100.0)), 1
                )
                item["properties"]["solubility_score"] = round(prediction["score"] * 100, 1)
                item["predicted_solubility_mole_fraction"] = round(
                    prediction["mole_fraction"], 6
                )
                confidence = prediction["confidence"]
                if item["_constraint_penalty"] > 0:
                    confidence *= max(0.35, 1.0 - item["_constraint_penalty"] * 1.8)
                item["confidence"] = round(confidence, 1)
                item["confidence_factors"] = prediction["confidence_factors"]
                item["basis"] = "LiNO₃ 实测溶解度模型 + 分子描述符 + 物理约束"

        if salt in self.model.supported_salts:
            covered = [
                item
                for item in candidates
                if item["_ml_input"]["solvent_a"] in self.model.supported_solvents
                and item["_ml_input"]["solvent_b"] in self.model.supported_solvents
            ]
            predictions = self.model.predict_many([item["_ml_input"] for item in covered])
            for item, (prediction, uncertainty, confidence) in zip(covered, predictions):
                ml_score = float(np.clip(prediction / 15.0, 0.0, 1.0))
                score_delta = item["_conductivity_weight"] * (ml_score - item["_transport_score"]) * 100
                item["score"] = round(float(np.clip(item["score"] + score_delta, 0.0, 100.0)), 1)
                item["properties"]["conductivity_score"] = round(ml_score * 100, 1)
                item["predicted_conductivity"] = round(prediction, 3)
                item["conductivity_uncertainty"] = round(uncertainty, 3)
                if item["_constraint_penalty"] > 0:
                    confidence *= max(0.35, 1.0 - item["_constraint_penalty"] * 1.8)
                item["confidence"] = round(confidence, 1)
                item["basis"] = "CALiSol-23 电导率模型 + 物理约束"

        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected = self._select_diverse(candidates, options.top_k)
        selected.sort(
            key=lambda item: (item["score"], item["confidence"]),
            reverse=True,
        )

        for item in selected:
            for key in (
                "_transport_score",
                "_heuristic_solubility_score",
                "_solubility_weight",
                "_conductivity_weight",
                "_constraint_penalty",
                "_lino3_input",
                "_ml_input",
            ):
                item.pop(key, None)

        has_lino3_training = salt == "LiNO3" and self.lino3_model.available
        # LiNO3 now has direct pure-solvent labels, but the requested binary
        # mixtures and conductivity remain outside the labelled training domain.
        is_extrapolation = salt not in self.model.supported_salts
        return {
            "salt": salt,
            "salt_smiles": SALT_SMILES.get(salt),
            "temperature_c": options.temperature_c,
            "concentration": options.concentration,
            "concentration_unit": options.concentration_unit,
            "is_extrapolation": is_extrapolation,
            "training_coverage": {
                "conductivity_labels": salt in self.model.supported_salts,
                "solubility_labels": has_lino3_training,
                "binary_mixture_labels": False if salt == "LiNO3" else None,
            },
            "warning": (
                "LiNO₃ 已纳入实测溶解度训练集，但当前主要标签来自纯溶剂；二元配比仍属于模型插值/外推，且没有 LiNO₃ 电导率训练标签。"
                if has_lino3_training
                else "该盐不在电导率训练集内；结果是候选优先级和起始配比，不是已验证溶解度或最终配方。"
                if is_extrapolation
                else "结果用于实验前筛选；最终配方仍需溶解度、电导率、界面兼容性和安全测试确认。"
            ) + (
                " 满足全部约束的候选不足，列表中已补充低置信度的放宽约束备选。"
                if used_relaxed_fallback and any(
                    item["constraint_status"] == "relaxed" for item in selected
                )
                else ""
            ),
            "recommendations": selected,
            "search_space": {
                "solvents": len(self.catalog),
                "ratios_per_pair": 17,
                "evaluated_formulations": len(candidates),
                "feasible_formulations": feasible_count,
                "used_relaxed_fallback": used_relaxed_fallback,
            },
        }


def log1p(value: float) -> float:
    return log(1.0 + max(value, 0.0))

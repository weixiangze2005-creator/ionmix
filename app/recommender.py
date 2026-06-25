from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from math import exp, log, sqrt

import numpy as np
import pandas as pd

from app.catalog import load_catalog
from app.lino3_model import LiNO3SolubilityModel
from app.ml_model import ConductivityModel
from app.mixture_model import MixturePropertyModel


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
    score_threshold: float = 0.0
    max_results: int = 80
    max_components: int = 2
    return_all_above_threshold: bool = False
    weights: dict[str, float] | None = None
    allow_relaxed_fallback: bool = True


class FormulationRecommender:
    def __init__(self):
        self.catalog = load_catalog()
        self.model = ConductivityModel()
        self.lino3_model = LiNO3SolubilityModel()
        self.mixture_model = MixturePropertyModel()
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

    def _components_allowed(
        self, components: list[pd.Series], options: RecommendationOptions, relaxed: bool = False
    ) -> bool:
        if relaxed:
            return True
        if options.exclude_high_hazard and max(component.hazard_prior for component in components) >= 0.80:
            return False
        if max(component.viscosity_mpas for component in components) > options.max_mixture_viscosity * 4:
            return False
        # Exclude mixtures where all components are very low-polarity; one component
        # must carry most of the salt-solvation burden.
        if (
            max(component.dielectric_constant for component in components) < 12
            and max(component.donor_number for component in components) < 22
        ):
            return False
        return True

    def _evaluate(
        self,
        salt: str,
        components: list[tuple[pd.Series, float]],
        options: RecommendationOptions,
        relaxed: bool = False,
    ) -> dict:
        solvents = [component for component, _ in components]
        fractions = [float(fraction) for _, fraction in components]
        dielectric = sum(fraction * solvent.dielectric_constant for solvent, fraction in components)
        donor = sum(fraction * solvent.donor_number for solvent, fraction in components)
        viscosity = exp(sum(fraction * log(max(solvent.viscosity_mpas, 0.05)) for solvent, fraction in components))
        # Flash point of a volatile binary mixture is not linear. This
        # conservative proxy stays much closer to the more volatile component.
        lower_flash = min(solvent.flash_point_c for solvent in solvents)
        weighted_flash = sum(fraction * solvent.flash_point_c for solvent, fraction in components)
        flash = lower_flash + max(0.0, weighted_flash - lower_flash) * 0.18
        tpsa_density = sum(
            fraction * solvent.tpsa / solvent.molecular_weight
            for solvent, fraction in components
        )

        dielectric_score = self._scale("dielectric_constant", dielectric)
        donor_score = self._scale("donor_number", donor)
        polarity_score = float(np.clip(tpsa_density / 0.45, 0.0, 1.0))

        # LiNO3 has high lattice energy and no direct CALiSol labels, so the
        # fallback emphasizes charge separation and Lewis-basic coordination.
        if salt == "LiNO3":
            affinity = sum(
                fraction * LINO3_AFFINITY.get(solvent["code"], 0.35)
                for solvent, fraction in components
            )
            strong_solvator_fraction = sum(
                fraction
                for solvent, fraction in components
                if LINO3_AFFINITY.get(solvent["code"], 0.35) >= 0.78
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

        oxidation = sum(fraction * solvent.oxidation_prior for solvent, fraction in components)
        reduction = sum(fraction * solvent.reduction_prior for solvent, fraction in components)
        if options.application == "high_voltage":
            stability = 0.75 * oxidation + 0.25 * reduction
        elif options.application == "lithium_metal":
            stability = 0.25 * oxidation + 0.75 * reduction
        else:
            stability = 0.50 * oxidation + 0.50 * reduction

        volatility_penalty = self._scale(
            "boiling_point_c", min(solvent.boiling_point_c for solvent in solvents), inverse=True
        )
        low_temp = 0.65 * self._scale("viscosity_mpas", viscosity, inverse=True) + 0.35 * volatility_penalty
        safety = (
            0.55 * self._scale("flash_point_c", flash)
            + 0.45 * (1.0 - sum(fraction * solvent.hazard_prior for solvent, fraction in components))
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
        if options.exclude_high_hazard and max(solvent.hazard_prior for solvent in solvents) >= 0.80:
            violations.append("包含高危溶剂先验")
            constraint_penalty += 0.18
        if violations and not relaxed:
            return {}

        # High-melting solvents are useful in blends, but dominant fractions
        # near or below their melting point are poor room-temperature starts.
        phase_penalty = 0.0
        for solvent, fraction in components:
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
        complementary = float(np.std([solvent.dielectric_constant for solvent in solvents]) / 45.0)
        total += min(0.035, 0.035 * complementary)

        catalog_coverage = sum(
            int(solvent.training_code in self.model.supported_solvents)
            for solvent in solvents
        )
        confidence = 0.42 + 0.12 * (catalog_coverage / max(len(solvents), 1))
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

        component_payload = [
            {
                "code": solvent["code"],
                "name": solvent["name"],
                "ratio": round(100 * fraction),
                "pubchem_url": solvent.pubchem_url,
            }
            for solvent, fraction in components
        ]
        a = solvents[0]
        b = solvents[1]
        fraction_a = fractions[0]
        fraction_b = fractions[1]
        is_binary = len(components) == 2

        return {
            "components": component_payload,
            "component_count": len(components),
            "formula_key": "|".join(sorted(component["code"] for component in component_payload)),
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
            "sources": [component["pubchem_url"] for component in component_payload if component["pubchem_url"]],
            "_transport_score": transport_score,
            "_heuristic_solubility_score": solubility,
            "_solubility_weight": weights["solubility"] / weight_sum,
            "_conductivity_weight": weights["conductivity"] / weight_sum,
            "_constraint_penalty": constraint_penalty,
            "_lino3_input": {
                "components": components,
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
            } if is_binary else None,
            "_mixture_input": {
                "salt": salt,
                "temperature_c": options.temperature_c,
                "concentration": options.concentration,
                "concentration_unit": options.concentration_unit,
                "solvent_ratio_basis": "v",
                "components": [
                    (solvent["code"], fraction)
                    for solvent, fraction in components
                ],
            },
        }

    def _generate_candidates(
        self, salt: str, options: RecommendationOptions, relaxed: bool
    ) -> list[dict]:
        candidates = []
        for (_, a), (_, b) in combinations(self.catalog.iterrows(), 2):
            if not self._components_allowed([a, b], options, relaxed=relaxed):
                continue
            for ratio in np.arange(0.10, 0.91, 0.05):
                result = self._evaluate(
                    salt,
                    [(a, float(ratio)), (b, float(1.0 - ratio))],
                    options,
                    relaxed=relaxed,
                )
                if result:
                    candidates.append(result)
        if options.max_components >= 3:
            ternary_templates = {
                (0.50, 0.35, 0.15),
                (0.45, 0.35, 0.20),
                (0.40, 0.40, 0.20),
                (0.60, 0.25, 0.15),
            }
            for (_, a), (_, b), (_, c) in combinations(self.catalog.iterrows(), 3):
                solvents = [a, b, c]
                if not self._components_allowed(solvents, options, relaxed=relaxed):
                    continue
                for ratios in ternary_templates:
                    for ordered in set(permutations(ratios)):
                        result = self._evaluate(
                            salt,
                            list(zip(solvents, ordered)),
                            options,
                            relaxed=relaxed,
                        )
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
                if item["_ml_input"]
                and item["_ml_input"]["solvent_a"] in self.model.supported_solvents
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

        if self.mixture_model.available:
            mixture_predictions = self.mixture_model.predict_many(
                [item["_mixture_input"] for item in candidates]
            )
            for item, prediction in zip(candidates, mixture_predictions):
                if prediction["solubility_score"] is not None:
                    old_score = item["properties"]["solubility_score"] / 100.0
                    new_score = float(prediction["solubility_score"])
                    score_delta = item["_solubility_weight"] * (new_score - old_score) * 100
                    item["score"] = round(float(np.clip(item["score"] + score_delta, 0.0, 100.0)), 1)
                    item["properties"]["solubility_score"] = round(new_score * 100, 1)
                    item["predicted_solubility_mole_fraction"] = round(
                        float(prediction["solubility_mole_fraction"]), 6
                    )
                if prediction["conductivity_score"] is not None:
                    old_score = item["properties"]["conductivity_score"] / 100.0
                    new_score = float(prediction["conductivity_score"])
                    score_delta = item["_conductivity_weight"] * (new_score - old_score) * 100
                    item["score"] = round(float(np.clip(item["score"] + score_delta, 0.0, 100.0)), 1)
                    item["properties"]["conductivity_score"] = round(new_score * 100, 1)
                    item["predicted_conductivity"] = round(
                        float(prediction["conductivity"]), 3
                    )
                if prediction["used_targets"]:
                    confidence = float(prediction["confidence"])
                    if item["_constraint_penalty"] > 0:
                        confidence *= max(0.35, 1.0 - item["_constraint_penalty"] * 1.8)
                    item["confidence"] = round(float(max(item["confidence"], confidence)), 1)
                    item["confidence_factors"].update(prediction["confidence_factors"])
                    item["basis"] = item["basis"] + " + 配方级公开实验模型"

        candidates.sort(key=lambda item: item["score"], reverse=True)
        if options.return_all_above_threshold:
            threshold = float(options.score_threshold)
            best_per_formula = {}
            for item in candidates:
                if item["score"] < threshold:
                    continue
                best_per_formula.setdefault(item["formula_key"], item)
            selected = sorted(
                best_per_formula.values(),
                key=lambda item: (item["score"], item["confidence"]),
                reverse=True,
            )[: options.max_results]
        else:
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
                "_mixture_input",
                "formula_key",
            ):
                item.pop(key, None)

        has_lino3_training = salt == "LiNO3" and self.lino3_model.available
        mixture_metrics = self.mixture_model.metrics if self.mixture_model.available else {}
        has_binary_mixture_labels = bool(
            salt == "LiNO3"
            and mixture_metrics.get("solubility", {}).get("lino3_binary_rows", 0) > 0
        )
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
                "binary_mixture_labels": has_binary_mixture_labels if salt == "LiNO3" else None,
                "mixture_model": self.mixture_model.available,
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
                "max_components": options.max_components,
                "score_threshold": options.score_threshold,
                "returned_formulations": len(selected),
                "return_all_above_threshold": options.return_all_above_threshold,
                "evaluated_formulations": len(candidates),
                "feasible_formulations": feasible_count,
                "used_relaxed_fallback": used_relaxed_fallback,
            },
        }


def log1p(value: float) -> float:
    return log(1.0 + max(value, 0.0))

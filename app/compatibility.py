from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CompatibilityRule:
    """Declarative engineering rule applied after molecular generation.

    ``block`` rules remove a formulation from the searchable space. ``caution``
    rules keep it, add a transparent score penalty, and expose the reason to the
    frontend.  These rules represent conservative battery-engineering policy;
    they are deliberately separate from solubility and conductivity models.
    """

    rule_id: str
    title: str
    severity: str
    salts: frozenset[str] = frozenset()
    solvents: frozenset[str] = frozenset()
    applications: frozenset[str] = frozenset()
    max_concentration: float | None = None
    requires_safety_filter: bool = False
    message: str = ""
    rationale: str = ""
    sources: tuple[str, ...] = ()
    score_penalty: float = 0.0

    def matches(
        self,
        *,
        salt: str,
        solvent_codes: set[str],
        application: str,
        concentration: float,
        exclude_high_hazard: bool,
    ) -> bool:
        if self.salts and salt not in self.salts:
            return False
        if self.solvents and not (self.solvents & solvent_codes):
            return False
        if self.applications and application not in self.applications:
            return False
        if self.max_concentration is not None and concentration > self.max_concentration:
            return False
        if self.requires_safety_filter and not exclude_high_hazard:
            return False
        return True


# Keep the policy compact and auditable. A rule is added only when it has a
# clear engineering consequence and a source that explains the boundary.
COMPATIBILITY_RULES: tuple[CompatibilityRule, ...] = (
    CompatibilityRule(
        rule_id="lino3_dmso_conservative_exclusion",
        title="LiNO₃ / DMSO 默认工程排除",
        severity="block",
        salts=frozenset({"LiNO3"}),
        solvents=frozenset({"DMSO"}),
        message="已排除 LiNO₃ / DMSO：本工具默认不推荐需要专门界面保护或高浓度策略的 DMSO 电池体系。",
        rationale=(
            "LiNO₃ 可以溶于 DMSO，且有 Li–O₂ 研究使用该组合；但普通 DMSO 体系与锂金属的副反应和界面稳定性"
            "使其不适合作为通用起始配方，因此按项目要求采用保守硬过滤。"
        ),
        sources=(
            "https://doi.org/10.1002/adfm.202010627",
            "https://doi.org/10.1039/C3RA47372D",
        ),
    ),
    CompatibilityRule(
        rule_id="litfsi_low_concentration_al_corrosion",
        title="LiTFSI 高电压铝腐蚀风险",
        severity="caution",
        salts=frozenset({"LiTFSI"}),
        applications=frozenset({"high_voltage"}),
        message="LiTFSI 高电压体系存在铝集流体腐蚀风险，需通过高浓度、成膜盐或集流体保护进行验证。",
        rationale="LiTFSI 对铝集流体的腐蚀与盐浓度和钝化膜形成密切相关，不能仅凭溶剂性质判定可用。",
        sources=("https://doi.org/10.1016/j.jpowsour.2012.12.028",),
        score_penalty=0.06,
    ),
    CompatibilityRule(
        rule_id="lifsi_low_concentration_al_corrosion",
        title="LiFSI 高电压铝腐蚀风险",
        severity="caution",
        salts=frozenset({"LiFSI"}),
        applications=frozenset({"high_voltage"}),
        message="LiFSI 高电压体系存在铝集流体腐蚀风险，结果仅保留为需验证候选。",
        rationale="LiFSI 在高电压下可能引发铝集流体阳极溶解，高浓度或混盐策略可改变这一结论。",
        sources=("https://doi.org/10.1021/acsami.4c09083",),
        score_penalty=0.06,
    ),
    CompatibilityRule(
        rule_id="liclo4_organic_electrolyte_safety",
        title="LiClO₄ 有机电解液强氧化安全风险",
        severity="caution",
        salts=frozenset({"LiClO4"}),
        requires_safety_filter=True,
        message="LiClO₄ 与有机溶剂组合具有强氧化和爆炸风险，只应作为严格受控的研究候选。",
        rationale="高氯酸根与有机物在热、机械或高电流滥用条件下存在严重安全风险。",
        sources=("https://doi.org/10.1007/s41918-019-00060-4",),
        score_penalty=0.10,
    ),
)


def assess_compatibility(
    *,
    salt: str,
    solvent_codes: Iterable[str],
    application: str,
    concentration: float,
    exclude_high_hazard: bool,
) -> dict:
    """Return a JSON-ready assessment for one candidate formulation."""

    codes = {str(code) for code in solvent_codes}
    matched = [
        rule
        for rule in COMPATIBILITY_RULES
        if rule.matches(
            salt=salt,
            solvent_codes=codes,
            application=application,
            concentration=concentration,
            exclude_high_hazard=exclude_high_hazard,
        )
    ]
    blocking = [rule for rule in matched if rule.severity == "block"]
    cautions = [rule for rule in matched if rule.severity == "caution"]
    return {
        "blocked": bool(blocking),
        "status": "blocked" if blocking else "caution" if cautions else "passed",
        "rule_ids": [rule.rule_id for rule in matched],
        "messages": [rule.message for rule in matched],
        "rationales": [rule.rationale for rule in matched],
        "sources": sorted({source for rule in matched for source in rule.sources}),
        "score_penalty": min(0.25, sum(rule.score_penalty for rule in cautions)),
        "matched_rules": [
            {
                "id": rule.rule_id,
                "title": rule.title,
                "severity": rule.severity,
            }
            for rule in matched
        ],
    }


def public_rule_catalog() -> list[dict]:
    """Expose rule metadata without implementation-only matching fields."""

    return [
        {
            "id": rule.rule_id,
            "title": rule.title,
            "severity": rule.severity,
            "message": rule.message,
            "rationale": rule.rationale,
            "sources": list(rule.sources),
        }
        for rule in COMPATIBILITY_RULES
    ]

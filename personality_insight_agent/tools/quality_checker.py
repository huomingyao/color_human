"""
质量检查工具

借鉴 Nuwa-Skill 的质量检查清单(Phase 4 通过标准)，
在画像生成后自动检查各项指标。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckResult:
    check_name: str
    passed: bool
    detail: str = ""
    fix_suggestion: str = ""


@dataclass
class QualityReport:
    checks: list[CheckResult] = field(default_factory=list)
    total: int = 0
    passed_count: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed_count / self.total

    @property
    def all_passed(self) -> bool:
        return self.passed_count == self.total


def check_mental_models(mental_models: list) -> CheckResult:
    """检查心智模型数量和质量"""
    count = len(mental_models)
    if count < 1:
        return CheckResult("心智模型数量", False, f"仅{count}个，建议至少识别1个", "放宽triple_check标准或标注'思维模式不显著'")
    if count > 5:
        return CheckResult("心智模型数量", False, f"{count}个过多，可能未充分提炼", "将部分降级为'决策启发式'")
    return CheckResult("心智模型数量", True, f"{count}个，数量合理")


def check_evidence_coverage(skill_outputs: list[dict]) -> CheckResult:
    """检查各个结论是否有 evidence 支撑"""
    missing = []
    for i, output in enumerate(skill_outputs):
        if not output:
            continue
        # 递归检查 evidence 字段
        evidences = _collect_evidence_fields(output)
        empty_evidences = [k for k, v in evidences.items() if isinstance(v, list) and len(v) == 0]
        if empty_evidences:
            missing.append(f"Skill{i+1}: {', '.join(empty_evidences)}")

    if missing:
        return CheckResult(
            "证据覆盖", False,
            f"以下维度缺少证据: {'; '.join(missing)}",
            "补充原文引用或删除无证据的结论",
        )
    return CheckResult("证据覆盖", True, "所有结论均有原文证据支撑")


def check_honesty_boundary(honesty: dict) -> CheckResult:
    """检查诚实边界是否充分"""
    known = len(honesty.get("known", []))
    uncertain = len(honesty.get("uncertain", []))
    unknown = len(honesty.get("unknown", []))

    if known == 0:
        return CheckResult("诚实边界", False, "完全没有known维度", "至少列出有证据的维度")
    if uncertain == 0 and unknown == 0:
        return CheckResult("诚实边界", False, "没有标注不确定或未知的维度——过于自信", "诚实列出信息不足的维度")
    if uncertain + unknown < 2:
        return CheckResult("诚实边界", True, "基本完整，建议增加更多不确定标注")
    return CheckResult("诚实边界", True, f"完整: {known}项已知, {uncertain}项不确定, {unknown}项未知")


def check_contradictions(contradictions: list) -> CheckResult:
    """检查是否检测并保留了矛盾"""
    if len(contradictions) == 0:
        return CheckResult(
            "矛盾检测", False,
            "未检测到任何矛盾——可能过于平滑，真实人格总是有矛盾的",
            "重新审查 Skill2/3/4 之间是否有不一致的信号",
        )
    essential = [c for c in contradictions if isinstance(c, dict) and c.get("type") == "essential_tension"]
    if not essential:
        return CheckResult("矛盾检测", True, f"检测到{len(contradictions)}处矛盾，但缺乏本质性张力", "检查是否有更深层的价值观冲突")
    return CheckResult("矛盾检测", True, f"检测到{len(contradictions)}处矛盾（含{len(essential)}处本质性张力）")


def check_confidence_granularity(metadata: dict) -> CheckResult:
    """检查置信度是否分层标注"""
    overall = metadata.get("overall_confidence", 0)
    if overall == 0:
        return CheckResult("置信度", False, "未设置置信度", "运行置信度计算")
    if overall > 0.9:
        return CheckResult(
            "置信度", False,
            f"整体置信度{overall}过高，人格分析很难达到90%+",
            "下调置信度或标注'高置信度仅针对分析材料覆盖的范围'",
        )
    return CheckResult("置信度", True, f"整体置信度{overall:.0%}，合理范围")


def run_all_checks(
    mental_models: list,
    skill_outputs: list[dict],
    honesty: dict,
    contradictions: list,
    metadata: dict,
) -> QualityReport:
    """运行全部质量检查"""
    checks = [
        check_mental_models(mental_models),
        check_evidence_coverage(skill_outputs),
        check_honesty_boundary(honesty),
        check_contradictions(contradictions),
        check_confidence_granularity(metadata),
    ]

    report = QualityReport(
        checks=checks,
        total=len(checks),
        passed_count=sum(1 for c in checks if c.passed),
    )
    return report


def _collect_evidence_fields(d: dict, prefix: str = "") -> dict[str, list]:
    """递归收集所有 evidence 字段"""
    result = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if key == "evidence" and isinstance(value, list):
            result[full_key] = value
        elif isinstance(value, dict):
            result.update(_collect_evidence_fields(value, full_key))
    return result

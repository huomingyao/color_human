"""
双引擎验证 — Arbitrator (仲裁器)

当内源分析(三重验证)与外源证据(事实核查)发生冲突时，
仲裁器负责深度研判并最终裁定。

仲裁机制 (5种策略):
1. prefer_internal  — 内部分析更可信 (内部分析有更多上下文，外部证据片面)
2. prefer_external  — 外部证据更可信 (客观事实 > 主观分析)
3. flag_as_disputed — 标记争议，双方各自保留，不偏袒
4. deep_dive        — 指令Agent重新检索更深层证据
5. merge            — 双方可以调和 (不同角度的同一事实，换个说法而已)

决策逻辑:
- 外部证据为 NOT_FOUND → prefer_internal (没找到外部证据，只能信任内部)
- 内部置信度高 + 外部证据 LIKELY → merge (双方基本一致)
- 内部置信度低 + 外部证据 CONFIRMED/CONTRADICTED → prefer_external
- 双方都强且对立 → flag_as_disputed + deep_dive
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .models import (
    ArbitrationResult,
    ArbitrationStrategy,
    ConflictResolution,
    ConflictType,
    EvidenceStrength,
    FactCheckItem,
    VerificationResult,
)
from ..agentic_rag.utils import parse_json_response


class DualEngineArbitrator:
    """
    双引擎仲裁器 — 内部分析 vs 外部证据的最终裁判。

    用法:
        arb = DualEngineArbitrator()
        result = arb.arbitrate(verification_result, llm_callable)
        print(result.arbitration_summary)
    """

    def __init__(self):
        # 仲裁策略决策矩阵
        self.STRATEGY_MATRIX: dict[tuple[str, str], ArbitrationStrategy] = {
            # (internal_confidence_level, external_strength) → strategy
            ("high", "confirmed"):   ArbitrationStrategy.MERGE,
            ("high", "likely"):      ArbitrationStrategy.MERGE,
            ("high", "uncertain"):   ArbitrationStrategy.PREFER_INTERNAL,
            ("high", "contradicted"):ArbitrationStrategy.FLAG_AS_DISPUTED,
            ("high", "not_found"):   ArbitrationStrategy.PREFER_INTERNAL,

            ("medium", "confirmed"):   ArbitrationStrategy.MERGE,
            ("medium", "likely"):      ArbitrationStrategy.MERGE,
            ("medium", "uncertain"):   ArbitrationStrategy.FLAG_AS_DISPUTED,
            ("medium", "contradicted"):ArbitrationStrategy.DEEP_DIVE,
            ("medium", "not_found"):   ArbitrationStrategy.PREFER_INTERNAL,

            ("low", "confirmed"):      ArbitrationStrategy.PREFER_EXTERNAL,
            ("low", "likely"):         ArbitrationStrategy.PREFER_EXTERNAL,
            ("low", "uncertain"):      ArbitrationStrategy.FLAG_AS_DISPUTED,
            ("low", "contradicted"):   ArbitrationStrategy.PREFER_EXTERNAL,
            ("low", "not_found"):      ArbitrationStrategy.FLAG_AS_DISPUTED,
        }

    # ========================================================================
    # 主入口: 仲裁
    # ========================================================================

    def arbitrate(
        self,
        verification: VerificationResult,
        llm_callable: Optional[Callable[[str, str], str]] = None,
        skill_outputs: Optional[dict[str, Any]] = None,
    ) -> ArbitrationResult:
        """
        执行仲裁流程。

        Args:
            verification: 事实核查结果 (含所有 disputed items)
            llm_callable: LLM调用
            skill_outputs: 原始Skill输出 (用于深度研判时回溯)

        Returns:
            ArbitrationResult: 完整仲裁报告
        """
        conflicts: list[ConflictResolution] = []

        # 只对有争议的条目进行仲裁
        disputable_items = verification.disputes_requiring_arbitration
        if not disputable_items:
            # 也纳入 uncertain 的条目
            disputable_items = [
                item for item in verification.checked_items
                if item.verification_result in ("contradicted", "uncertain")
            ]

        for item in disputable_items:
            resolution = self._arbitrate_single(item, llm_callable, skill_outputs)
            conflicts.append(resolution)

        # 统计
        resolved = sum(1 for c in conflicts if c.strategy != ArbitrationStrategy.FLAG_AS_DISPUTED)
        disputed = sum(1 for c in conflicts if c.strategy == ArbitrationStrategy.FLAG_AS_DISPUTED)

        strategy_counts: dict[str, int] = {}
        for c in conflicts:
            s = c.strategy.value
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        # 生成建议
        recommendations = self._generate_recommendations(conflicts)
        honesty_additions = self._generate_honesty_additions(conflicts)

        # 生成摘要
        summary = self._generate_arbitration_summary(conflicts, resolved, disputed)

        return ArbitrationResult(
            arbitration_id=f"arb_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            conflicts=conflicts,
            total_conflicts=len(conflicts),
            resolved_count=resolved,
            disputed_count=disputed,
            strategy_usage=strategy_counts,
            recommendations=recommendations,
            honesty_boundary_additions=honesty_additions,
            arbitrator_notes="",
            arbitration_summary=summary,
        )

    # ========================================================================
    # 单条仲裁
    # ========================================================================

    def _arbitrate_single(
        self,
        item: FactCheckItem,
        llm_callable: Optional[Callable[[str, str], str]] = None,
        skill_outputs: Optional[dict[str, Any]] = None,
    ) -> ConflictResolution:
        """对单个争议条目进行仲裁"""
        # Step 1: 确定冲突类型
        conflict_type = self._classify_conflict(item)

        # Step 2: 选择仲裁策略
        strategy = self._select_strategy(item)

        # Step 3: 执行仲裁
        if strategy == ArbitrationStrategy.DEEP_DIVE and llm_callable:
            # 深度研判: 让 LLM 分析双方论点
            resolution, final_verdict = self._deep_dive(item, llm_callable)
        elif strategy == ArbitrationStrategy.MERGE:
            resolution, final_verdict = self._merge_positions(item, llm_callable)
        elif strategy == ArbitrationStrategy.PREFER_EXTERNAL:
            resolution, final_verdict = self._prefer_external(item)
        elif strategy == ArbitrationStrategy.PREFER_INTERNAL:
            resolution, final_verdict = self._prefer_internal(item)
        else:  # FLAG_AS_DISPUTED
            resolution, final_verdict = self._flag_disputed(item)

        # Step 4: 计算对画像的影响
        impact = self._assess_impact(item, strategy)

        # Step 5: 调整置信度
        adjusted = self._adjust_confidence(item, strategy)

        return ConflictResolution(
            conflict_id=f"conflict_{item.item_id}",
            conflict_type=conflict_type,
            internal_position=f"[{item.source}] {item.claim}\n分析: {item.internal_analysis[:300]}",
            external_position=f"证据强度: {item.external_strength.value}\n证据: {item.external_evidence[:300]}",
            strategy=strategy,
            resolution=resolution,
            final_verdict=final_verdict,
            impact_on_profile=impact,
            adjusted_confidence=adjusted,
            evidence_chain=[item.internal_analysis, item.external_evidence],
            reviewer_notes="",
        )

    # ========================================================================
    # Step 1: 冲突分类
    # ========================================================================

    def _classify_conflict(self, item: FactCheckItem) -> ConflictType:
        """分类冲突类型"""
        claim_lower = item.claim.lower()

        # 事实类关键词 → FACTUAL
        factual_kw = ["决策", "收购", "投资", "离职", "创办", "出售", "事件", "时间", "地点"]
        if any(kw in claim_lower for kw in factual_kw):
            return ConflictType.FACTUAL

        # 性格/能力类 → INTERPRETIVE
        trait_kw = ["性格", "能力", "特质", "特征", "倾向", "偏好", "擅长"]
        if any(kw in claim_lower for kw in trait_kw):
            return ConflictType.INTERPRETIVE

        # 时间类 → TEMPORAL
        time_kw = ["变化", "转变", "以前", "后来", "现在", "早期", "最近", "逐渐"]
        if any(kw in claim_lower for kw in time_kw):
            return ConflictType.TEMPORAL

        # 场景类 → CONTEXTUAL
        context_kw = ["场景", "场合", "环境", "情境", "背景下", "面对"]
        if any(kw in claim_lower for kw in context_kw):
            return ConflictType.CONTEXTUAL

        return ConflictType.INTERPRETIVE

    # ========================================================================
    # Step 2: 策略选择
    # ========================================================================

    def _select_strategy(self, item: FactCheckItem) -> ArbitrationStrategy:
        """
        根据内部置信度和外部证据强度选择仲裁策略。

        决策矩阵(INTERNAL × EXTERNAL):
        - 高置信度 + 无外部证据 → PREFER_INTERNAL
        - 高置信度 + 外部矛盾 → FLAG_AS_DISPUTED (需要更深入研究)
        - 低置信度 + 外部明确 → PREFER_EXTERNAL
        - 中置信度 + 外部矛盾 → DEEP_DIVE
        """
        # 置信度分级
        if item.internal_confidence >= 0.7:
            conf_level = "high"
        elif item.internal_confidence >= 0.4:
            conf_level = "medium"
        else:
            conf_level = "low"

        ext_strength = item.external_strength.value

        key = (conf_level, ext_strength)
        return self.STRATEGY_MATRIX.get(
            key,
            ArbitrationStrategy.FLAG_AS_DISPUTED,  # 默认标记争议
        )

    # ========================================================================
    # Step 3: 执行各类仲裁策略
    # ========================================================================

    def _deep_dive(
        self,
        item: FactCheckItem,
        llm_callable: Callable[[str, str], str],
    ) -> tuple[str, str]:
        """深度研判: 让 LLM 综合双方论据给出判断"""
        try:
            prompt = f"""=== 争议条目 ===
声明: {item.claim}
来源: {item.source}

=== 内部分析 (三重验证) ===
置信度: {item.internal_confidence}
分析: {item.internal_analysis[:1000]}

=== 外部证据 ===
证据强度: {item.external_strength.value}
证据: {item.external_evidence[:1000]}

请作为仲裁者，回答:
1. 双方论点各自的有效性
2. 证据是否真正矛盾，还是从不同角度描述同一事实？
3. 最终裁定 (采纳内部/采纳外部/标记争议/调和双方)

输出JSON:
{
    "analysis": "仲裁分析",
    "verdict": "最终裁定"
}"""

            resp = llm_callable(
                "你是公正的仲裁专家。综合双方论据，做出判断。只输出JSON。",
                prompt,
            )
            parsed = self._parse_json_response(resp)
            return parsed.get("analysis", "深度研判未能得出结论"), parsed.get("verdict", "标记为争议")
        except Exception:
            return "深度研判失败", "标记为争议，建议人工审查"

    def _merge_positions(
        self,
        item: FactCheckItem,
        llm_callable: Optional[Callable[[str, str], str]] = None,
    ) -> tuple[str, str]:
        """调和双方: 找到共识"""
        resolution = (
            f"内部分析({item.source}, 置信度{item.internal_confidence})与"
            f"外部证据(强度{item.external_strength.value})基本一致，可相互印证。"
        )
        verdict = f"确认声明: {item.claim[:150]}"
        return resolution, verdict

    def _prefer_external(self, item: FactCheckItem) -> tuple[str, str]:
        """采纳外部"""
        resolution = (
            f"内部分析置信度较低({item.internal_confidence})，"
            f"而外部证据强度为{item.external_strength.value}。"
            f"优先采纳外部证据。"
        )
        verdict = f"根据外部证据修订: {item.external_evidence[:200]}"
        return resolution, verdict

    def _prefer_internal(self, item: FactCheckItem) -> tuple[str, str]:
        """采纳内部"""
        resolution = (
            f"外部证据不足(强度={item.external_strength.value})，"
            f"内部分析置信度较高({item.internal_confidence})。"
            f"暂时维持内部分析结论，并在诚实边界中标注外部验证不足。"
        )
        verdict = f"维持内部分析: {item.claim[:150]}"
        return resolution, verdict

    def _flag_disputed(self, item: FactCheckItem) -> tuple[str, str]:
        """标记争议"""
        resolution = (
            f"内部分析(置信度{item.internal_confidence})与外部证据"
            f"(强度{item.external_strength.value})存在矛盾，"
            f"目前无法确定哪方更可信。标记为争议，双方保留。"
        )
        verdict = f"争议未解: {item.claim[:150]} — 双方论据均记录在案"
        return resolution, verdict

    # ========================================================================
    # Step 4-5: 影响评估与置信度调整
    # ========================================================================

    def _assess_impact(self, item: FactCheckItem, strategy: ArbitrationStrategy) -> str:
        """评估对画像的影响程度"""
        if strategy == ArbitrationStrategy.PREFER_EXTERNAL:
            if item.category == "trait":
                return "moderate"  # 性格特征被修正
            elif item.category in ("claim", "event", "decision"):
                return "major"     # 重要声明被推翻
        elif strategy == ArbitrationStrategy.FLAG_AS_DISPUTED:
            return "moderate"      # 争议降低了可信度
        elif strategy == ArbitrationStrategy.DEEP_DIVE:
            return "minor"
        return "none"

    def _adjust_confidence(self, item: FactCheckItem, strategy: ArbitrationStrategy) -> float:
        """根据仲裁结果调整置信度"""
        base = item.internal_confidence

        adjustments = {
            ArbitrationStrategy.PREFER_INTERNAL: 0.0,     # 不变
            ArbitrationStrategy.PREFER_EXTERNAL: -0.25,    # 下调
            ArbitrationStrategy.FLAG_AS_DISPUTED: -0.15,   # 下调
            ArbitrationStrategy.DEEP_DIVE: -0.05,           # 微调
            ArbitrationStrategy.MERGE: +0.05,               # 上调 (印证)
        }

        delta = adjustments.get(strategy, -0.1)
        return max(0.1, min(0.95, base + delta))  # 限制在 [0.1, 0.95]

    # ========================================================================
    # 报告生成
    # ========================================================================

    def _generate_recommendations(self, conflicts: list[ConflictResolution]) -> list[str]:
        """基于仲裁结果生成画像修改建议"""
        recs = []

        for c in conflicts:
            if c.strategy == ArbitrationStrategy.PREFER_EXTERNAL:
                recs.append(f"修正: {c.final_verdict[:200]}")
            elif c.strategy == ArbitrationStrategy.FLAG_AS_DISPUTED:
                recs.append(f"标注争议: {c.final_verdict[:200]}")
            elif c.impact_on_profile == "major":
                recs.append(f"重大修正: {c.final_verdict[:200]}")

        if not recs:
            recs.append("画像结论与外部证据基本一致，无需重大修改")

        return recs

    def _generate_honesty_additions(self, conflicts: list[ConflictResolution]) -> list[str]:
        """从仲裁结果提取应加入诚实边界的内容"""
        additions = []

        for c in conflicts:
            if c.strategy == ArbitrationStrategy.FLAG_AS_DISPUTED:
                additions.append(
                    f"关于「{c.conflict_id}」，内部分析与外部证据存在矛盾，"
                    f"此维度的结论可靠性有限。"
                )
            elif c.strategy == ArbitrationStrategy.PREFER_INTERNAL:
                additions.append(
                    f"关于「{c.conflict_id}」，外部验证证据不足，"
                    f"结论主要基于内部分析，可能不够全面。"
                )
            elif c.impact_on_profile == "major":
                additions.append(
                    f"关于「{c.conflict_id}」，经外部核查后已修正，"
                    f"原始分析可能存在偏差。"
                )

        return additions

    def _generate_arbitration_summary(
        self,
        conflicts: list[ConflictResolution],
        resolved: int,
        disputed: int,
    ) -> str:
        """生成仲裁摘要"""
        summary = f"""=== 双引擎仲裁报告 ===

总冲突数: {len(conflicts)}
已解决: {resolved}
标记争议: {disputed}

仲裁策略分布:
"""
        strategy_counts: dict[str, int] = {}
        for c in conflicts:
            s = c.strategy.value
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        for strategy, count in strategy_counts.items():
            summary += f"  - {strategy}: {count} 条\n"

        summary += "\n各冲突裁定:\n"
        for c in conflicts:
            summary += f"\n  [{c.conflict_id}] ({c.conflict_type.value})\n"
            summary += f"  策略: {c.strategy.value}\n"
            summary += f"  裁定: {c.final_verdict[:200]}\n"
            summary += f"  影响: {c.impact_on_profile}\n"

        return summary

    # ========================================================================
    # 便捷方法
    # ========================================================================

    def quick_arbitrate(
        self,
        verification: VerificationResult,
    ) -> ArbitrationResult:
        """快速仲裁 (不调用 LLM，仅用规则)"""
        return self.arbitrate(verification, llm_callable=None)

    def _parse_json_response(self, resp: str) -> dict[str, Any]:
        return parse_json_response(resp)

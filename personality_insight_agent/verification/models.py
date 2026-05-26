"""
双引擎验证 — 数据模型

定义事实核查、仲裁和验证报告的完整数据类型。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# 枚举
# ============================================================================


class EvidenceStrength(str, Enum):
    """证据强度"""
    CONFIRMED = "confirmed"          # 多源交叉确认
    LIKELY = "likely"                # 有证据支持但不完全确定
    UNCERTAIN = "uncertain"          # 证据不足
    CONTRADICTED = "contradicted"    # 被证据推翻
    NOT_FOUND = "not_found"          # 在知识库中未找到相关信息


class ArbitrationStrategy(str, Enum):
    """仲裁策略"""
    PREFER_INTERNAL = "prefer_internal"    # 内部分析更可信 (有更多上下文)
    PREFER_EXTERNAL = "prefer_external"    # 外部证据更可信 (客观事实)
    FLAG_AS_DISPUTED = "flag_as_disputed"  # 标记争议，不偏袒任何一方
    DEEP_DIVE = "deep_dive"                # 需要更深度的调查
    MERGE = "merge"                        # 双方可以调和 (不同角度的同一事实)


class ConflictType(str, Enum):
    """冲突类型"""
    FACTUAL = "factual"          # 事实层面: 说A vs 实际是B
    INTERPRETIVE = "interpretive" # 解读层面: 对同一事实的不同解读
    TEMPORAL = "temporal"       # 时间层面: 过去说A后来做B
    CONTEXTUAL = "contextual"   # 场景层面: 场景X说A场景Y说B


# ============================================================================
# 事实核查
# ============================================================================


class FactCheckItem(BaseModel):
    """单个待核查的事实声明"""
    item_id: str = Field(..., description="唯一标识")
    claim: str = Field(..., description="原始声明 (从Skill2/3/4的输出中提取)")
    source: str = Field(..., description="声明的来源 (内部推理/原文引用)")
    category: str = Field(default="claim", description="声明类别: claim/event/decision/trait")

    # 内外部分析
    internal_analysis: str = Field(default="", description="内部分析结论 (三重验证)")
    internal_confidence: float = Field(default=0.5, description="内部分析置信度")

    # 外部核查
    external_evidence: str = Field(default="", description="外部核查找到的证据")
    external_strength: EvidenceStrength = Field(default=EvidenceStrength.NOT_FOUND)
    external_sources: List[str] = Field(default_factory=list, description="外部证据来源")

    # 核查结论
    verification_result: str = Field(default="pending", description="verified/contradicted/uncertain")


class VerificationResult(BaseModel):
    """一轮事实核查的完整结果"""
    verification_id: str = Field(..., description="批次ID")
    checked_items: List[FactCheckItem] = Field(default_factory=list, description="核查条目列表")

    # 统计
    total_claims: int = Field(default=0)
    verified_count: int = Field(default=0)
    contradicted_count: int = Field(default=0)
    uncertain_count: int = Field(default=0)

    # 仲裁触发
    disputes_requiring_arbitration: List[FactCheckItem] = Field(
        default_factory=list,
        description="需要仲裁的争议条目"
    )

    # 整体评估
    overall_credibility: float = Field(default=1.0, description="整体可信度 (0-1)")
    summary: str = Field(default="", description="核查摘要")


# ============================================================================
# 仲裁结果
# ============================================================================


class ConflictResolution(BaseModel):
    """单个冲突的解决结果"""
    conflict_id: str = Field(..., description="冲突标识")
    conflict_type: ConflictType = Field(..., description="冲突类型")

    # 冲突双方
    internal_position: str = Field(..., description="内部分析的立场")
    external_position: str = Field(..., description="外部证据的立场")

    # 仲裁
    strategy: ArbitrationStrategy = Field(..., description="采用的仲裁策略")
    resolution: str = Field(..., description="仲裁决策说明")
    final_verdict: str = Field(..., description="最终裁定")

    # 影响
    impact_on_profile: str = Field(default="none", description="对画像的影响: none/minor/moderate/major")
    adjusted_confidence: float = Field(default=0.0, description="调整后的置信度")

    # 审计
    evidence_chain: List[str] = Field(default_factory=list, description="证据链")
    reviewer_notes: str = Field(default="", description="仲裁者备注")


class ArbitrationResult(BaseModel):
    """完整的仲裁报告"""
    arbitration_id: str = Field(..., description="仲裁批次ID")
    conflicts: List[ConflictResolution] = Field(default_factory=list)

    # 统计
    total_conflicts: int = Field(default=0)
    resolved_count: int = Field(default=0)
    disputed_count: int = Field(default=0)  # 标记为争议但无法解决

    # 策略分布
    strategy_usage: Dict[str, int] = Field(default_factory=dict)

    # 建议
    recommendations: List[str] = Field(
        default_factory=list,
        description="给最终画像的建议修改"
    )
    honesty_boundary_additions: List[str] = Field(
        default_factory=list,
        description="应补充到诚实边界的内容"
    )

    # 元信息
    arbitrator_notes: str = Field(default="", description="仲裁者综合备注")
    arbitration_summary: str = Field(default="", description="仲裁摘要 (供报告使用)")

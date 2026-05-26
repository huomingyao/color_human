"""
双引擎验证 — Fact Check Agent (外部事实核查)

在女娲原生"三重验证"(跨域复现/生成力/排他性)之后介入，
从知识库中核实具体事件和观点的事实基础。

流程:
1. 从 Skill 2/3/4 的输出中提取可验证的声明
2. 对每个声明，用 RAG 回知识库检索证据
3. 对比内部分析 vs 外部证据
4. 标记 verified / contradicted / uncertain
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from .models import (
    EvidenceStrength,
    FactCheckItem,
    VerificationResult,
)
from ..agentic_rag.models import RetrievalResult
from ..agentic_rag.utils import parse_json_response


class FactCheckAgent:
    """
    外部事实核查 Agent。

    职责:
    - 从画像输出中提取可验证声明
    - 调用 RAG 回知识库核实
    - 标记验证状态

    用法:
        checker = FactCheckAgent(tree, vector_search_fn)
        result = checker.verify(
            skill_outputs={"skill2": ..., "skill3": ..., "skill4": ...},
            llm_callable=my_llm,
        )
    """

    def __init__(
        self,
        tree=None,
        vector_search_fn: Optional[Callable[[str, str, int], list[dict[str, Any]]]] = None,
    ):
        """
        Args:
            tree: Corpus2Skill 知识目录树
            vector_search_fn: (query, category, top_k) -> list[dict] 向量检索
        """
        self.tree = tree
        self.vector_search_fn = vector_search_fn

    # ========================================================================
    # 主入口
    # ========================================================================

    def verify(
        self,
        skill_outputs: dict[str, Any],
        llm_callable: Callable[[str, str], str],
        original_text: str = "",
    ) -> VerificationResult:
        """
        对画像输出进行事实核查。

        Args:
            skill_outputs: {"skill2": {...}, "skill3": {...}, "skill4": {...}}
            llm_callable: LLM 调用
            original_text: 原始文本 (用于文本检索后备)

        Returns:
            VerificationResult: 核查结果
        """
        # Step 1: 提取可验证声明
        claims = self._extract_claims(skill_outputs, llm_callable)

        # Step 2: 逐个核查
        for claim in claims:
            # 2a. 从知识库检索证据
            evidence = self._retrieve_evidence(claim.claim, llm_callable)

            # 2b. 对比声明 vs 证据
            strength, sources, ev_text = self._evaluate_evidence(
                claim, evidence, llm_callable
            )

            claim.external_evidence = ev_text
            claim.external_strength = strength
            claim.external_sources = sources

            # 2c. 判定
            claim.verification_result = self._determine_result(claim, strength)

        # Step 3: 生成核查报告
        verified = sum(1 for c in claims if c.verification_result == "verified")
        contradicted = sum(1 for c in claims if c.verification_result == "contradicted")
        uncertain = sum(1 for c in claims if c.verification_result == "uncertain")

        disputes = [c for c in claims if c.verification_result == "contradicted"]

        # 计算整体可信度
        total = len(claims) or 1
        credibility = (verified * 1.0 + uncertain * 0.5 + contradicted * 0.0) / total

        summary = self._generate_summary(claims, verified, contradicted, uncertain, credibility)

        return VerificationResult(
            verification_id=f"verify_{len(claims)}_{hash(str(skill_outputs)) % 10000}",
            checked_items=claims,
            total_claims=len(claims),
            verified_count=verified,
            contradicted_count=contradicted,
            uncertain_count=uncertain,
            disputes_requiring_arbitration=disputes,
            overall_credibility=round(credibility, 2),
            summary=summary,
        )

    # ========================================================================
    # Step 1: 声明提取
    # ========================================================================

    def _extract_claims(
        self,
        skill_outputs: dict[str, Any],
        llm_callable: Callable[[str, str], str],
    ) -> list[FactCheckItem]:
        """从 Skill 输出中提取可验证的事实声明"""
        claims = []

        # 从 Skill2 提取认知相关声明
        skill2 = skill_outputs.get("skill2", {})
        cognitive = skill2.get("cognitive_profile", {})
        mental_models = cognitive.get("mental_models", [])
        for i, mm in enumerate(mental_models):
            name = mm.get("name", "") if isinstance(mm, dict) else str(mm)
            evidence_text = json.dumps(mm.get("evidence", []), ensure_ascii=False) if isinstance(mm, dict) else ""
            if name:
                claims.append(FactCheckItem(
                    item_id=f"mm_{i}",
                    claim=f"心智模型: {name}",
                    source="Skill2 内部分析",
                    category="trait",
                    internal_analysis=evidence_text[:500],
                    internal_confidence=0.75,
                ))

        # 从 Skill3 提取语言风格声明
        skill3 = skill_outputs.get("skill3", {})
        lang = skill3.get("language_style", {})
        sig_patterns = lang.get("signature_patterns", [])
        for i, sp in enumerate(sig_patterns):
            sp_text = sp if isinstance(sp, str) else json.dumps(sp, ensure_ascii=False)
            claims.append(FactCheckItem(
                item_id=f"sig_{i}",
                claim=f"语言特征: {sp_text[:200]}",
                source="Skill3 句式分析",
                category="trait",
                internal_analysis=sp_text[:500],
                internal_confidence=0.70,
            ))

        # 从 Skill4 提取性格声明
        skill4 = skill_outputs.get("skill4", {})
        personality = skill4.get("personality", {})
        big_five = personality.get("big_five", {})
        for trait, info in big_five.items():
            if isinstance(info, dict):
                score = info.get("score", 0)
                evidence = info.get("evidence", [])
                ev_str = json.dumps(evidence, ensure_ascii=False) if evidence else ""
                claims.append(FactCheckItem(
                    item_id=f"bf_{trait}",
                    claim=f"大五-{trait}: 得分 {score}",
                    source="Skill4 性格推断",
                    category="trait",
                    internal_analysis=ev_str[:500],
                    internal_confidence=info.get("confidence", 0.5),
                ))

        mbti = personality.get("mbti", {})
        if mbti:
            claims.append(FactCheckItem(
                item_id="mbti_type",
                claim=f"MBTI类型: {mbti.get('type', '?')}",
                source="Skill4 MBTI推断",
                category="trait",
                internal_analysis=f"置信度: {mbti.get('confidence', 0.5)}",
                internal_confidence=mbti.get("confidence", 0.5),
            ))

        # 用 LLM 辅助提取显性声明 (如有 LLM)
        if llm_callable and len(claims) > 5:
            claims = self._llm_filter_claims(llm_callable, claims, skill_outputs)

        return claims

    def _llm_filter_claims(
        self,
        llm_callable: Callable[[str, str], str],
        claims: list[FactCheckItem],
        skill_outputs: dict[str, Any],
    ) -> list[FactCheckItem]:
        """用 LLM 过滤和增强待核查声明"""
        try:
            claims_desc = "\n".join(f"- [{c.item_id}] {c.claim}" for c in claims)
            prompt = f"""以下是人物画像分析产出的声明列表:

{claims_desc}

请选出最重要的5-8个声明进行事实核查。优先级:
1. 可直接与原文/事实对比的声明 (如决策记录、已知事件)
2. 置信度较低的声明 (优先核查)
3. 可验证性高的声明 (有明确对错)

输出JSON数组: ["item_id1", "item_id2", ...]"""

            resp = llm_callable("你是事实核查策略专家。只输出JSON数组。", prompt)
            json_match = re.search(r'\[[\s\S]*\]', resp)
            if json_match:
                selected_ids = set(json.loads(json_match.group()))
                return [c for c in claims if c.item_id in selected_ids]
        except Exception:
            pass

        return claims

    # ========================================================================
    # Step 2: 证据检索
    # ========================================================================

    def _retrieve_evidence(
        self,
        claim: str,
        llm_callable: Callable[[str, str], str],
    ) -> RetrievalResult:
        """用 RAG 回知识库检索证据"""
        if self.vector_search_fn:
            # 用向量检索
            chunks = self.vector_search_fn(claim, "all", 5)
            return RetrievalResult(
                query=claim,
                chunks=chunks if chunks else [],
                retrieval_method="vector",
            )
        else:
            # 退化为 LLM 知识库检索
            return RetrievalResult(
                query=claim,
                chunks=[],
                retrieval_method="none",
            )

    # ========================================================================
    # Step 3: 证据评估
    # ========================================================================

    def _evaluate_evidence(
        self,
        claim: FactCheckItem,
        evidence: RetrievalResult,
        llm_callable: Callable[[str, str], str],
    ) -> tuple[EvidenceStrength, list[str], str]:
        """对比声明与检索到的证据"""
        if not evidence.chunks:
            return EvidenceStrength.NOT_FOUND, [], "在知识库中未找到相关证据"

        # 收集证据文本
        ev_texts = []
        sources = []
        for chunk in evidence.chunks:
            text = chunk.get("content", "") or chunk.get("text", "") or ""
            source = chunk.get("source", "") or chunk.get("chunk_id", "")
            if text:
                ev_texts.append(text)
                sources.append(source)

        combined_evidence = "\n\n".join(ev_texts[:5])

        if not llm_callable:
            # 无 LLM 时降级为规则判断
            strength = self._rule_based_evaluation(claim.claim, combined_evidence)
            return strength, sources, combined_evidence[:1000]

        # LLM 评估
        try:
            system_prompt = """你是一个严谨的事实核查员。请对比以下声明和证据。

输出JSON:
{
    "strength": "confirmed/likely/uncertain/contradicted",
    "reasoning": "判断理由 (1-2句话)",
    "key_evidence": "关键证据摘要"
}

判断标准:
- confirmed: 证据明确支持声明
- likely: 证据倾向支持但不完全确定
- uncertain: 证据不足或模棱两可
- contradicted: 证据与声明直接矛盾"""

            user_prompt = f"""待核查声明: {claim.claim}
声明来源: {claim.source}
内部分析: {claim.internal_analysis[:500]}

知识库检索证据:
{combined_evidence[:3000]}

请对比并输出JSON判定。"""

            resp = llm_callable(system_prompt, user_prompt)
            parsed = self._parse_json_response(resp)

            strength_map = {
                "confirmed": EvidenceStrength.CONFIRMED,
                "likely": EvidenceStrength.LIKELY,
                "uncertain": EvidenceStrength.UNCERTAIN,
                "contradicted": EvidenceStrength.CONTRADICTED,
            }
            strength = strength_map.get(
                parsed.get("strength", "uncertain"),
                EvidenceStrength.UNCERTAIN,
            )

            reasoning = parsed.get("reasoning", "")
            key_evidence = parsed.get("key_evidence", "")

            return strength, sources, f"判定: {strength.value}\n理由: {reasoning}\n关键证据: {key_evidence}"

        except Exception:
            return EvidenceStrength.UNCERTAIN, sources, combined_evidence[:1000]

    def _rule_based_evaluation(self, claim: str, evidence: str) -> EvidenceStrength:
        """规则层面的证据评估 (无LLM时使用)"""
        if not evidence.strip():
            return EvidenceStrength.NOT_FOUND

        claim_words = set(claim.lower().split())
        evidence_lower = evidence.lower()

        # 关键词命中率
        hits = sum(1 for w in claim_words if w in evidence_lower)
        hit_rate = hits / max(len(claim_words), 1)

        if hit_rate > 0.5:
            return EvidenceStrength.LIKELY
        elif hit_rate > 0.2:
            return EvidenceStrength.UNCERTAIN
        else:
            return EvidenceStrength.NOT_FOUND

    # ========================================================================
    # Step 4: 判定
    # ========================================================================

    def _determine_result(
        self,
        claim: FactCheckItem,
        strength: EvidenceStrength,
    ) -> str:
        """根据证据强度判定验证结果"""
        if strength == EvidenceStrength.CONFIRMED or strength == EvidenceStrength.LIKELY:
            return "verified"
        elif strength == EvidenceStrength.CONTRADICTED:
            return "contradicted"
        else:
            return "uncertain"

    # ========================================================================
    # 汇总
    # ========================================================================

    def _generate_summary(
        self,
        claims: list[FactCheckItem],
        verified: int,
        contradicted: int,
        uncertain: int,
        credibility: float,
    ) -> str:
        """生成核查摘要"""
        summary = f"""事实核查完成:
- 总声明数: {len(claims)}
- 验证通过: {verified}
- 被推翻: {contradicted}
- 不确定: {uncertain}
- 整体可信度: {credibility:.0%}
"""
        if contradicted > 0:
            summary += "\n⚠️ 存在被外部证据推翻的声明，已标记为争议，将进入仲裁程序。\n"
            for c in claims:
                if c.verification_result == "contradicted":
                    summary += f"  - [{c.item_id}] {c.claim[:100]}\n"

        return summary

    def _parse_json_response(self, resp: str) -> dict[str, Any]:
        return parse_json_response(resp)

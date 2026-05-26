"""
Agentic RAG — Reflector (反思与迭代)

核心职责:
- 评估当前检索结果的信息充分性
- 检测多源信息中的冲突
- 识别信息缺口
- 决策是否继续检索以及下一步方向
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from .models import (
    AgentAction,
    ReflectionResult,
    RetrievalResult,
    ResearchLoopConfig,
)
from .utils import parse_json_array


class ResearchReflector:
    """
    反思器 — 对检索结果进行评估和反思。

    用法:
        reflector = ResearchReflector(config)
        reflection = reflector.reflect(results, original_instruction, llm_callable)
        if reflection.should_continue:
            next_queries = reflection.suggested_queries
    """

    def __init__(self, config: Optional[ResearchLoopConfig] = None):
        self.config = config or ResearchLoopConfig()

    # ========================================================================
    # 主入口: 反思
    # ========================================================================

    def reflect(
        self,
        results: list[RetrievalResult],
        original_instruction: str,
        llm_callable: Optional[Callable[[str, str], str]] = None,
    ) -> ReflectionResult:
        """
        对检索结果进行全面反思。

        Args:
            results: 本次循环的所有检索结果
            original_instruction: 原始研究指令
            llm_callable: LLM调用函数

        Returns:
            ReflectionResult: 包含充分性判断、冲突、缺口和建议
        """
        # 1. 评估信息充分性
        sufficient, quality = self._evaluate_sufficiency(results)

        # 2. 冲突检测
        conflicts = self._detect_conflicts(results, llm_callable) if self.config.conflict_detection else []

        # 3. 缺口分析
        gaps = self._analyze_gaps(results, original_instruction, llm_callable) if self.config.gap_analysis else []

        # 4. 决策是否继续
        should_continue = self._decide_continue(
            sufficient, quality, conflicts, gaps, len(results)
        )

        # 5. 生成建议
        suggested_actions = self._suggest_actions(should_continue, conflicts, gaps)
        suggested_queries = self._generate_refinement_queries(conflicts, gaps, original_instruction)

        # 6. 生成反思日志
        reflection_log = self._build_reflection_log(
            sufficient, quality, conflicts, gaps, should_continue
        )

        return ReflectionResult(
            information_sufficient=sufficient,
            quality_score=quality,
            conflicts_found=conflicts,
            gaps=gaps,
            should_continue=should_continue,
            suggested_actions=suggested_actions,
            suggested_queries=suggested_queries,
            reflection_log=reflection_log,
        )

    # ========================================================================
    # 1. 信息充分性评估
    # ========================================================================

    def _evaluate_sufficiency(self, results: list[RetrievalResult]) -> tuple[bool, float]:
        """评估检索结果是否足够"""
        if not results:
            return False, 0.0

        # 维度计算
        total_chunks = sum(len(r.chunks) for r in results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        avg_density = sum(r.information_density for r in results) / len(results)
        categories_covered = set()
        sources = set()
        for r in results:
            categories_covered.update(r.categories_covered)
            sources.update(r.sources)

        # 充分性评分
        chunk_score = min(total_chunks / 10, 1.0)  # 至少10个块
        category_score = min(len(categories_covered) / 3, 1.0)  # 至少3个分类
        source_score = min(len(sources) / 5, 1.0)  # 至少5个来源

        quality = (
            chunk_score * 0.25
            + avg_relevance * 0.30
            + avg_density * 0.20
            + category_score * 0.15
            + source_score * 0.10
        )

        # 是否充分
        sufficient = (
            quality >= self.config.early_stop_quality
            and total_chunks >= 5
            and avg_relevance >= self.config.min_relevance
        )

        return sufficient, quality

    # ========================================================================
    # 2. 冲突检测
    # ========================================================================

    def _detect_conflicts(
        self,
        results: list[RetrievalResult],
        llm_callable: Optional[Callable[[str, str], str]] = None,
    ) -> list[dict[str, Any]]:
        """检测检索结果中的信息冲突"""
        conflicts = []

        # 收集所有文段
        all_texts = []
        for result in results:
            for chunk in result.chunks:
                text = chunk.get("content", "") or chunk.get("text", "") or ""
                source = chunk.get("source", "") or chunk.get("chunk_id", "")
                if text:
                    all_texts.append({"text": text, "source": source})

        if len(all_texts) < 2:
            return conflicts

        # 如果有 LLM，委托冲突检测
        if llm_callable:
            return self._llm_conflict_detection(llm_callable, all_texts)

        # 无 LLM 时做简单的规则检测
        # 检查立场/观点关键词的极性不一致
        conflicts = self._rule_based_conflict_detection(all_texts)

        return conflicts

    def _llm_conflict_detection(
        self,
        llm_callable: Callable[[str, str], str],
        texts: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """使用 LLM 进行冲突检测"""
        # 取前 5 个代表性文段
        samples = texts[:5]
        sample_text = "\n\n---\n\n".join(
            f"[来源: {t['source']}]\n{t['text'][:500]}"
            for t in samples
        )

        try:
            prompt = f"""分析以下文本片段，检测是否存在信息冲突：

{sample_text}

冲突类型包括:
1. temporal(时间性): 同一件事在不同时间的说法不同 (可能是自然变化)
2. spatial(场景性): 不同场景下表现不同 (可能是合理的场景差异)
3. essential(本质性): 核心价值观/方法论层面的矛盾 (最有价值的信号)

请以 JSON 格式输出检测到的冲突:
[{{"type": "temporal/spatial/essential", "description": "...", "source_a": "...", "source_b": "...", "severity": "high/medium/low"}}]

如果没有冲突，输出空数组 []。只输出JSON。"""

            resp = llm_callable("你是信息冲突检测专家。只输出JSON。", prompt)
            return parse_json_array(resp)
        except Exception:
            pass

        return []

    def _rule_based_conflict_detection(
        self,
        texts: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """规则层面冲突检测（无LLM时使用）"""
        conflicts = []

        # 极性词检测
        positive_patterns = [
            r'(成功|正确|有效|优秀|最佳|最优|必须|一定要)',
        ]
        negative_patterns = [
            r'(失败|错误|无效|糟糕|最差|不能|绝不|放弃)',
        ]

        # 找到包含极性对立的两段文本
        pos_texts = []
        neg_texts = []

        for t in texts:
            text = t["text"]
            if any(re.search(p, text) for p in positive_patterns):
                pos_texts.append(t)
            if any(re.search(p, text) for p in negative_patterns):
                neg_texts.append(t)

        # 如果针对相同主题出现对立评价
        if pos_texts and neg_texts:
            # 简单检查是否可能在说同一件事
            for pt in pos_texts[:3]:
                for nt in neg_texts[:3]:
                    conflicts.append({
                        "type": "essential",
                        "description": f"检测到正负面评价共存，可能表示内在矛盾",
                        "source_a": pt["source"],
                        "source_b": nt["source"],
                        "severity": "medium",
                    })
                    break
                break

        return conflicts

    # ========================================================================
    # 3. 缺口分析
    # ========================================================================

    def _analyze_gaps(
        self,
        results: list[RetrievalResult],
        instruction: str,
        llm_callable: Optional[Callable[[str, str], str]] = None,
    ) -> list[dict[str, str]]:
        """分析信息缺口"""
        gaps = []

        # 检查覆盖的维度
        covered = set()
        for r in results:
            covered.update(r.categories_covered)

        expected_categories = [
            "writings", "conversations", "expressions",
            "external_views", "decisions", "timeline",
        ]
        missing = [c for c in expected_categories if c not in covered]

        for cat in missing:
            gaps.append({
                "dimension": cat,
                "description": f"缺少 {cat} 维度的信息",
                "suggested_query": f"{instruction} (在{cat}分类中搜索)",
            })

        # 检查深度: 如果总chunk数太少
        total_chunks = sum(len(r.chunks) for r in results)
        if total_chunks < 5:
            gaps.append({
                "dimension": "depth",
                "description": f"检索结果过少(仅{total_chunks}个相关块)，信息可能不足",
                "suggested_query": f"放宽查询: {instruction}",
            })

        return gaps

    # ========================================================================
    # 4. 继续/停止决策
    # ========================================================================

    def _decide_continue(
        self,
        sufficient: bool,
        quality: float,
        conflicts: list[dict[str, Any]],
        gaps: list[dict[str, str]],
        round_count: int,
    ) -> bool:
        """决定是否继续检索"""
        # 质量足够 → 停止
        if sufficient and quality >= self.config.early_stop_quality:
            return False

        # 有未解决的高严重性冲突 → 继续
        if any(c.get("severity") == "high" for c in conflicts):
            return True

        # 有重要信息缺口 → 继续
        if len(gaps) > 2:
            return True

        # 已经很多轮了 → 停止
        if round_count >= self.config.max_rounds:
            return False

        # 默认: 如果质量还不够高，继续
        return quality < self.config.min_relevance

    # ========================================================================
    # 5. 建议生成
    # ========================================================================

    def _suggest_actions(
        self,
        should_continue: bool,
        conflicts: list[dict[str, Any]],
        gaps: list[dict[str, str]],
    ) -> list[AgentAction]:
        """建议下一步动作"""
        actions = []

        if should_continue:
            if conflicts:
                actions.append(AgentAction.REFLECT)
                actions.append(AgentAction.SEARCH)
            elif gaps:
                actions.append(AgentAction.SEARCH)
                if len(gaps) > 3:
                    actions.append(AgentAction.REPLAN)
            else:
                actions.append(AgentAction.SEARCH)
        else:
            actions.append(AgentAction.SYNTHESIZE)
            actions.append(AgentAction.STOP)

        return actions

    def _generate_refinement_queries(
        self,
        conflicts: list[dict[str, Any]],
        gaps: list[dict[str, str]],
        instruction: str,
    ) -> list[str]:
        """生成优化后的检索查询"""
        queries = []

        # 从冲突中生成验证查询
        for conflict in conflicts:
            desc = conflict.get("description", "")
            if desc:
                queries.append(f"验证: {desc}")

        # 从缺口中生成补充查询
        for gap in gaps:
            sq = gap.get("suggested_query", "")
            if sq:
                queries.append(sq)

        # 如果没有任何新查询，至少给一个
        if not queries:
            queries.append(f"换个角度检索: {instruction}")

        return queries[:5]

    # ========================================================================
    # 6. 反思日志
    # ========================================================================

    def _build_reflection_log(
        self,
        sufficient: bool,
        quality: float,
        conflicts: list[dict[str, Any]],
        gaps: list[dict[str, str]],
        should_continue: bool,
    ) -> str:
        """生成反思日志"""
        log = f"""=== 反思评估 ===
信息充分性: {'足够' if sufficient else '不足'}
质量评分: {quality:.2f}
发现冲突: {len(conflicts)} 处
信息缺口: {len(gaps)} 个
决策: {'继续检索' if should_continue else '停止检索，综合结果'}
"""
        if conflicts:
            log += "\n冲突详情:\n"
            for c in conflicts:
                log += f"  - [{c.get('type', '?')}] {c.get('description', '')[:100]}\n"

        if gaps:
            log += "\n信息缺口:\n"
            for g in gaps:
                log += f"  - [{g.get('dimension', '?')}] {g.get('description', '')[:100]}\n"

        return log

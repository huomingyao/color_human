"""
Agentic RAG — Research Agent (升级版研究者)

整合 自主规划 → 深度检索 → 反思迭代 的完整循环，
将女娲的被动"收集者"升级为主动"研究智能体"。

与 Nuwa 6 Agent 并行模式的兼容:
- 本 Agent 可以替代原有的任何单个采集 Agent
- 多个 AgenticResearchAgent 可并行运行 (如同传统6 Agent)
- 但每个内部都具备自主规划和迭代能力

用法:
    agent = AgenticResearchAgent(
        tree=corpus_tree,
        vector_search_fn=my_faiss_search,
        config=ResearchLoopConfig(max_rounds=5),
    )
    result = agent.research(
        instruction="研究此人的重大商业决策模式",
        llm_callable=my_llm,
    )
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .models import (
    AgentAction,
    AgentState,
    ReflectionResult,
    ResearchDepth,
    ResearchLoopConfig,
    ResearchPlan,
    ResearchTask,
    RetrievalResult,
)
from .planner import ResearchPlanner
from .retriever import DeepRetriever
from .reflector import ResearchReflector
from ..corpus2skill.models import SkillDirectoryTree
from ..corpus2skill.navigator import KnowledgeNavigator


class AgenticResearchAgent:
    """
    升级版研究 Agent — 自主规划、深度检索、反思迭代。

    状态机: IDLE → PLANNING → RETRIEVING ⇄ REFLECTING → SYNTHESIZING → DONE

    可配置为对应 Nuwa 6 Agent 的任一维度:
    - Agent 1 (著作): research("分析此人的著作和系统性长文中的核心论点", ...)
    - Agent 2 (对话): research("分析此人在对话和访谈中的即兴思维", ...)
    - Agent 5 (决策): research("分析此人的重大决策记录", ...)
    """

    def __init__(
        self,
        tree: Optional[SkillDirectoryTree] = None,
        vector_search_fn: Optional[Callable[[str, str, int], list[dict[str, Any]]]] = None,
        config: Optional[ResearchLoopConfig] = None,
        agent_name: str = "ResearchAgent",
    ):
        """
        Args:
            tree: Corpus2Skill 知识目录树 (可选，但有它才能用导航模式)
            vector_search_fn: (query, category, top_k) -> list[dict] 向量检索函数
            config: 循环控制参数
            agent_name: Agent 名称 (便于在并行场景下识别)
        """
        self.tree = tree
        self.vector_search_fn = vector_search_fn
        self.config = config or ResearchLoopConfig()
        self.agent_name = agent_name

        # 子组件
        self.planner = ResearchPlanner()
        self.retriever = DeepRetriever(tree, vector_search_fn) if tree else None
        self.reflector = ResearchReflector(self.config)
        self.navigator = KnowledgeNavigator(tree) if tree else None

        # 状态
        self.state: AgentState = AgentState.IDLE
        self.history: list[dict[str, Any]] = []

        # 研究产出
        self.collected_results: list[RetrievalResult] = []
        self.reflection_history: list[ReflectionResult] = []

    # ========================================================================
    # 主入口: 研究循环
    # ========================================================================

    def research(
        self,
        instruction: str,
        llm_callable: Callable[[str, str], str],
        research_depth: ResearchDepth = ResearchDepth.STANDARD,
        knowledge_tree_summary: str = "",
    ) -> dict[str, Any]:
        """
        执行完整的研究循环。

        Args:
            instruction: 研究指令 (例如 "分析此人的决策模式")
            llm_callable: LLM 调用函数
            research_depth: 研究深度
            knowledge_tree_summary: 知识树摘要

        Returns:
            {
                "success": bool,
                "summary": str,
                "collected_results": list[RetrievalResult],
                "reflections": list[ReflectionResult],
                "total_rounds": int,
                "total_chunks": int,
                "quality": float,
                "state_history": list[str],
            }
        """
        self.state = AgentState.PLANNING
        self.collected_results = []
        self.reflection_history = []
        self.state_history: list[str] = []
        if self.retriever:
            self.retriever.reset_seen()

        start_time = time.time()

        # ---- Phase 1: 规划 ----
        self.state = AgentState.PLANNING
        self._log_state()
        plan = self.planner.create_plan(
            instruction=instruction,
            llm_callable=llm_callable,
            knowledge_tree_summary=knowledge_tree_summary,
            research_depth=research_depth,
        )

        # ---- Phase 2: 执行检索循环 ----
        round_count = 0
        max_rounds = plan.max_rounds

        while round_count < max_rounds:
            round_count += 1

            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout_seconds:
                break

            # 2a. 检索
            self.state = AgentState.RETRIEVING
            self._log_state()

            tasks_to_run = [t for t in plan.tasks if not t.completed]
            if not tasks_to_run:
                break

            for task in tasks_to_run[:3]:  # 每轮最多执行3个任务
                for query in task.search_queries[:2]:  # 每个任务最多2个查询
                    if self.retriever:
                        result = self.retriever.search(
                            query=query,
                            categories=task.target_categories,
                            top_k=5,
                            method="hybrid" if self.tree else "vector",
                        )
                    else:
                        # 退化为 LLM 直调
                        result = self._direct_llm_retrieval(query, task, llm_callable)

                    self.collected_results.append(result)
                    task.completed = True

            # 2b. 反思
            self.state = AgentState.REFLECTING
            self._log_state()

            reflection = self.reflector.reflect(
                results=self.collected_results,
                original_instruction=instruction,
                llm_callable=llm_callable,
            )
            self.reflection_history.append(reflection)

            # 2c. 决策
            if not reflection.should_continue:
                break

            if reflection.information_sufficient and reflection.quality_score >= self.config.early_stop_quality:
                break

            # 2d. 重新规划 (如有需要)
            if AgentAction.REPLAN in reflection.suggested_actions:
                self.state = AgentState.REPLANNING
                self._log_state()
                plan = self.planner.create_plan(
                    instruction=instruction,
                    llm_callable=llm_callable,
                    knowledge_tree_summary=knowledge_tree_summary,
                    research_depth=research_depth,
                    previous_results=self.collected_results,
                )

        # ---- Phase 3: 综合 ----
        self.state = AgentState.SYNTHESIZING
        self._log_state()

        summary = self._synthesize(instruction, llm_callable)

        self.state = AgentState.DONE
        self._log_state()

        return {
            "success": True,
            "summary": summary,
            "collected_results": self.collected_results,
            "reflections": self.reflection_history,
            "total_rounds": round_count,
            "total_chunks": sum(len(r.chunks) for r in self.collected_results),
            "quality": self.reflection_history[-1].quality_score if self.reflection_history else 0.0,
            "state_history": self.state_history,
        }

    # ========================================================================
    # 综合输出
    # ========================================================================

    def _synthesize(
        self,
        instruction: str,
        llm_callable: Callable[[str, str], str],
    ) -> str:
        """综合所有检索结果为文本摘要"""
        if not self.collected_results:
            return "未检索到相关信息。"

        # 收集所有文段
        all_chunks = []
        for result in self.collected_results:
            for chunk in result.chunks:
                text = chunk.get("content", "") or chunk.get("text", "") or ""
                source = chunk.get("source", "") or chunk.get("chunk_id", "")
                if text:
                    all_chunks.append(f"[来源: {source}]\n{text}")

        combined_text = "\n\n---\n\n".join(all_chunks[:15])  # 最多15个段

        if not llm_callable:
            return combined_text[:2000]

        try:
            system_prompt = """你是一个研究综合专家。请基于提供的检索结果，以客观、结构化的方式回答问题。

要求:
1. 基于证据回答，不编造
2. 标注信息来源
3. 如果证据相互矛盾，指出矛盾并分析可能原因
4. 如果信息不足，明确说明
5. 使用分点和编号组织回答"""
            user_prompt = f"研究问题: {instruction}\n\n检索结果:\n{combined_text[:5000]}"

            return llm_callable(system_prompt, user_prompt)
        except Exception as e:
            return f"(综合失败: {e})\n\n原始结果:\n{combined_text[:1000]}"

    # ========================================================================
    # 降级: 无向量库时的直接 LLM 检索
    # ========================================================================

    def _direct_llm_retrieval(
        self,
        query: str,
        task: ResearchTask,
        llm_callable: Callable[[str, str], str],
    ) -> RetrievalResult:
        """没有向量库时，直接用 LLM 做"检索"(本质是让 LLM 基于训练数据回答)"""
        try:
            system_prompt = f"""你正在执行研究任务: {task.description}
请基于你的知识库回答以下查询。如果不知道，请明确说明。

请输出:
1. 相关知识点 (分条列出)
2. 信息来源的文献/出处 (如能回忆)
3. 不确定的部分 (诚实标注)"""

            response = llm_callable(system_prompt, query)

            return RetrievalResult(
                query=query,
                chunks=[{"content": response, "source": "LLM_knowledge", "chunk_id": f"llm_{hash(query)}"}],
                sources=["LLM_knowledge"],
                categories_covered=task.target_categories,
                relevance_score=0.6,
                retrieval_method="direct_llm",
            )
        except Exception as e:
            return RetrievalResult(
                query=query,
                chunks=[],
                sources=[],
                relevance_score=0.0,
                retrieval_method="direct_llm",
            )

    # ========================================================================
    # 并行兼容: 适配 Nuwa 6 Agent 模式的批处理接口
    # ========================================================================

    @classmethod
    def run_parallel_agents(
        cls,
        instructions: list[dict[str, str]],  # [{"name": "Agent1", "instruction": "..."}]
        llm_callable: Callable[[str, str], str],
        tree: Optional[SkillDirectoryTree] = None,
        vector_search_fn: Optional[Callable] = None,
        depth: ResearchDepth = ResearchDepth.STANDARD,
    ) -> list[dict[str, Any]]:
        """
        批量运行多个研究 Agent (对应 Nuwa 6 Agent 并行采集模式)。

        每个 Agent 独立分配任务，6个 Agent 分别对应不同的内容维度。
        在 Python 层面是顺序执行，在 Agent 框架中可并行 spawn。

        Args:
            instructions: 每个 Agent 的指令列表
            llm_callable: LLM调用函数
            tree: 共享的知识树
            vector_search_fn: 共享的向量检索函数
            depth: 研究深度

        Returns:
            每个 Agent 的结果列表
        """
        results = []
        for instr in instructions:
            agent = cls(
                tree=tree,
                vector_search_fn=vector_search_fn,
                agent_name=instr["name"],
            )
            result = agent.research(
                instruction=instr["instruction"],
                llm_callable=llm_callable,
                research_depth=depth,
            )
            result["agent_name"] = instr["name"]
            results.append(result)

        return results

    # ========================================================================
    # 内部工具
    # ========================================================================

    def _log_state(self):
        """记录状态迁移"""
        self.state_history.append(self.state.value)

    def get_research_report(self) -> str:
        """生成研究过程的完整报告"""
        report = f"""=== {self.agent_name} 研究报告 ===

状态序列: {' → '.join(self.state_history)}
总轮数: {len(self.reflection_history)}
总检索结果: {len(self.collected_results)}
总文档块: {sum(len(r.chunks) for r in self.collected_results)}
最终质量: {self.reflection_history[-1].quality_score:.2f} (如存在)

=== 检索详情 ===
"""
        for i, r in enumerate(self.collected_results):
            report += f"\n检索 {i+1}: {r.query}\n"
            report += f"  方法: {r.retrieval_method}\n"
            report += f"  结果数: {len(r.chunks)}\n"
            report += f"  相关性: {r.relevance_score:.2f}\n"
            report += f"  来源: {', '.join(r.sources[:3])}\n"

        if self.reflection_history:
            report += f"\n=== 最后一次反思 ===\n{self.reflection_history[-1].reflection_log}"

        return report

    def to_nuwa_research_format(self) -> dict[str, Any]:
        """
        将研究结果转换为 Nuwa 兼容的调研文件格式。

        用于将 Agentic RAG 的研究结果写回 0X-xxx.md 调研文件。
        """
        summary = ""
        for result in self.collected_results:
            summary += f"\n## 检索查询: {result.query}\n\n"
            for chunk in result.chunks:
                text = chunk.get("content", "") or chunk.get("text", "") or ""
                source = chunk.get("source", "") or ""
                summary += f"- 来源: {source}\n  {text[:500]}\n\n"

        conflicts = []
        for refl in self.reflection_history:
            for c in refl.conflicts_found:
                conflicts.append(f"- [{c.get('type')}] {c.get('description')}")

        return {
            "summary": summary,
            "sources_count": sum(len(r.sources) for r in self.collected_results),
            "chunks_count": sum(len(r.chunks) for r in self.collected_results),
            "conflicts": conflicts,
            "information_gaps": [
                g.get("description", "")
                for refl in self.reflection_history
                for g in refl.gaps
            ],
            "quality_score": self.reflection_history[-1].quality_score if self.reflection_history else 0.0,
            "reflection_log": self.reflection_history[-1].reflection_log if self.reflection_history else "",
        }

"""
Orchestrator V2 — 整合三大组件的升级版编排器

在原有 5-Skill Pipeline 基础上，新增:
- Phase 0.5: Corpus2Skill 知识树加载 (如有知识库)
- Phase 0.6: Agentic RAG 自主研究 (替代传统被动检索)
- Phase 6: 双引擎验证 (事实核查 + 仲裁)
- Phase 7: 最终输出 (含仲裁报告)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .orchestrator_base import BaseOrchestrator
from .models import (
    InputType,
    OrchestratorOutput,
    PipelineStep,
)
from .corpus2skill.builder import CorpusTreeBuilder
from .corpus2skill.models import SkillDirectoryTree
from .corpus2skill.navigator import KnowledgeNavigator
from .agentic_rag.models import ResearchDepth, ResearchLoopConfig
from .agentic_rag.research_agent import AgenticResearchAgent
from .verification.fact_checker import FactCheckAgent
from .verification.arbitrator import DualEngineArbitrator
from .verification.models import ArbitrationResult, VerificationResult


class PersonalityInsightOrchestratorV2(BaseOrchestrator):
    """升级版编排器 — 集成 Corpus2Skill + Agentic RAG + 双引擎验证"""

    def __init__(self):
        super().__init__()
        self.corpus_builder = CorpusTreeBuilder()
        self.tree: Optional[SkillDirectoryTree] = None
        self.navigator: Optional[KnowledgeNavigator] = None
        self.fact_checker: Optional[FactCheckAgent] = None
        self.arbitrator = DualEngineArbitrator()
        self.rag_config = ResearchLoopConfig(
            max_rounds=5,
            early_stop_quality=0.85,
            timeout_seconds=600,
        )

    # ========================================================================
    # 模式1: 纯文本分析
    # ========================================================================

    def analyze(
        self,
        llm_callable,
        raw_text=None,
        person_id=None,
        options=None,
    ):
        options = options or {}
        steps: list[PipelineStep] = []

        input_type = self.classify_input(raw_text, person_id)
        degradation = self.check_degradation(raw_text or "")

        if degradation.get("degradation_needed"):
            return OrchestratorOutput(
                input_type=input_type,
                pipeline_used=[],
                degradation_triggered=True,
                degradation_reason=degradation["reason"],
            )

        cleaned_text, material_quality, s1_dict = self.run_skill1(
            llm_callable, raw_text, person_id, input_type, options
        )

        step1 = PipelineStep(step_name="Skill1_MaterialRetrieval", status="done")
        step1.quality = material_quality
        steps.append(step1)

        if material_quality < self.QUALITY_DEGRADE_THRESHOLD:
            return OrchestratorOutput(
                input_type=input_type,
                pipeline_used=["Skill1"],
                pipeline_steps=steps,
                degradation_triggered=True,
                degradation_reason=f"材料质量过低({material_quality:.2f})",
            )

        s2_dict, output = self.run_pipeline(
            llm_callable, cleaned_text, material_quality, s1_dict, person_id, input_type, steps
        )

        # Phase 6: 双引擎验证
        if self.fact_checker and options.get("enable_verification", True):
            self._run_verification(output, s2_dict, steps)

        return output

    # ========================================================================
    # 模式2: 含私有知识库的完整分析
    # ========================================================================

    def analyze_with_corpus(
        self,
        llm_callable: Callable,
        raw_text=None,
        person_id=None,
        corpus_dir=None,
        vector_search_fn: Optional[Callable] = None,
        research_depth: ResearchDepth = ResearchDepth.STANDARD,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        完整模式 — Corpus2Skill + Agentic RAG + 5-Skill Pipeline + 双引擎验证。

        Returns:
            {
                "orchestrator_output": OrchestratorOutput,
                "corpus_tree": SkillDirectoryTree | None,
                "agentic_research": list[dict],
                "verification": VerificationResult | None,
                "arbitration": ArbitrationResult | None,
                "extended_report": str,
            }
        """
        options = options or {}
        start_time = time.time()

        # Phase 0.5: 构建知识树
        tree = self._build_corpus_tree(llm_callable, corpus_dir, person_id)

        # Phase 0.6: Agentic RAG 自主研究
        research_results = self._run_agentic_research(
            llm_callable, person_id, tree, vector_search_fn, research_depth
        )

        # Phase 1-5 + 6: 标准 Pipeline
        orch_result = self.analyze(
            llm_callable, raw_text, person_id, options
        )

        # 生成扩展报告
        extended_report = self._generate_extended_report(
            orch_result=orch_result,
            research_results=research_results,
            elapsed=time.time() - start_time,
        )

        return {
            "orchestrator_output": orch_result,
            "corpus_tree": tree,
            "agentic_research": research_results,
            "verification": getattr(orch_result, "verification", None),
            "arbitration": getattr(orch_result, "arbitration", None),
            "extended_report": extended_report,
        }

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _build_corpus_tree(
        self,
        llm_callable,
        corpus_dir,
        person_id,
    ):
        if not corpus_dir:
            return None
        try:
            tree = self.corpus_builder.build(
                llm_callable=llm_callable,
                source_dir=corpus_dir,
                person_id=person_id or "unknown",
            )
            self.tree = tree
            self.navigator = KnowledgeNavigator(tree)
            self.fact_checker = FactCheckAgent(tree=tree)
            return tree
        except Exception as e:
            print(f"[OrchestratorV2] 知识树构建失败: {e}")
            return None

    def _run_agentic_research(
        self,
        llm_callable,
        person_id,
        tree,
        vector_search_fn,
        research_depth,
    ):
        research_results = []
        if not tree:
            return research_results

        nuwa_agent_instructions = [
            {"name": "Agent1_Writings", "instruction": f"研究 {person_id or '目标人物'} 的著作、系统长文、博客中的核心论点、方法论、反复出现的主题。"},
            {"name": "Agent2_Conversations", "instruction": f"研究 {person_id or '目标人物'} 在访谈、播客、对话中的即兴反应。"},
            {"name": "Agent3_Expressions", "instruction": f"研究 {person_id or '目标人物'} 的碎片表达，分析高频用词、语气模式。"},
            {"name": "Agent4_ExternalViews", "instruction": f"研究外部对 {person_id or '目标人物'} 的评价和批评。"},
            {"name": "Agent5_Decisions", "instruction": f"研究 {person_id or '目标人物'} 的重大决策记录。"},
            {"name": "Agent6_Timeline", "instruction": f"研究 {person_id or '目标人物'} 的人生时间线。"},
        ]

        for agent_instr in nuwa_agent_instructions:
            try:
                agent = AgenticResearchAgent(
                    tree=tree,
                    vector_search_fn=vector_search_fn,
                    agent_name=agent_instr["name"],
                    config=self.rag_config,
                )
                result = agent.research(
                    instruction=agent_instr["instruction"],
                    llm_callable=llm_callable,
                    research_depth=research_depth,
                )
                research_results.append(result)
            except Exception as e:
                research_results.append({
                    "agent_name": agent_instr["name"],
                    "success": False,
                    "error": str(e),
                })

        return research_results

    def _run_verification(
        self,
        output: OrchestratorOutput,
        s2_dict: dict,
        steps: list[PipelineStep],
    ):
        if not output.final_output:
            return

        skill_outputs = {
            "skill2": output.final_output.model_dump().get("cognitive", {}),
            "skill3": output.final_output.model_dump().get("language_emotion", {}),
            "skill4": output.final_output.model_dump().get("personality", {}),
        }

        verification = self.fact_checker.verify(
            skill_outputs=skill_outputs,
            llm_callable=getattr(self, "_llm_callable", None),
        )

        step6 = PipelineStep(step_name="Phase6_DualEngineVerification", status="done")
        step6.quality = verification.overall_credibility if verification else 0.7
        steps.append(step6)

        if verification.disputes_requiring_arbitration or verification.contradicted_count > 0:
            arbitration = self.arbitrator.arbitrate(
                verification=verification,
                llm_callable=getattr(self, "_llm_callable", None),
                skill_outputs=skill_outputs,
            )
            output.arbitration = arbitration
            if arbitration.honesty_boundary_additions:
                hb = output.final_output.honesty_boundary
                existing = list(hb.uncertain) if hb.uncertain else []
                existing.extend(arbitration.honesty_boundary_additions)
                hb.uncertain = existing

        output.verification = verification

    def _generate_extended_report(
        self,
        orch_result,
        research_results,
        elapsed,
    ) -> str:
        lines = []
        lines.append(f"# 完整画像报告 (V2 增强版)")
        lines.append(f"> 耗时: {elapsed:.1f}s\n")

        if orch_result.final_output:
            out = orch_result.final_output
            lines.append("## 一、核心画像")
            lines.append(f"- **思维风格**: {out.core_profile.thinking_style}")
            lines.append(f"- **性格快照**: {out.core_profile.personality_snapshot}")
            lines.append(f"- **整体置信度**: {out.metadata.overall_confidence:.0%}\n")

        if research_results:
            lines.append("## 二、Agentic RAG 研究摘要")
            for rr in research_results:
                lines.append(f"### {rr.get('agent_name', 'Unknown')}")
                lines.append(f"- 状态: {'成功' if rr.get('success') else '失败'}")
                lines.append(f"- 检索轮数: {rr.get('total_rounds', 0)}")
                lines.append(f"- 质量评分: {rr.get('quality', 0.0):.2f}\n")

        if orch_result.final_output:
            lines.append("## 三、5-Skill Pipeline 分析结果\n")
            lines.append(orch_result.final_output.report)
            lines.append("")

        if orch_result.verification:
            v = orch_result.verification
            lines.append("## 四、双引擎验证报告")
            lines.append(f"- 验证通过: {v.verified_count}, 被推翻: {v.contradicted_count}, 不确定: {v.uncertain_count}")
            lines.append(f"- 整体可信度: {v.overall_credibility:.0%}\n")

        lines.append("## 五、Pipeline 元信息")
        lines.append(f"- 执行顺序: {' → '.join(orch_result.pipeline_used)}")
        for step in orch_result.pipeline_steps:
            icon = "✓" if step.status == "done" else "⚠"
            lines.append(f"- {icon} **{step.step_name}**: quality={step.quality:.2f}")

        return "\n".join(lines)
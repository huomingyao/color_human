"""
Orchestrator (主控编排器)

借鉴 Nuwa-Skill 的 Phase 0-5 流程设计，协调 5 个 Skill 的顺序执行。

核心职责:
- Phase 0: 输入类型判断，路由决策
- Phase 1-4: 顺序调用 Skill 1→2→3→4
- Phase 5: 合成最终画像 (Skill 5) + 质量验证
- 检查点机制: 每个 Skill 输出质量检查
- 降级策略: 材料不足时降低分析深度
- 矛盾处理: 保留而非抹平
"""

from .orchestrator_base import BaseOrchestrator
from .models import (
    InputType,
    OrchestratorOutput,
    PipelineStep,
)


class PersonalityInsightOrchestrator(BaseOrchestrator):
    """
    人物画像分析主控编排器

    用法:
        orchestrator = PersonalityInsightOrchestrator()
        result = orchestrator.analyze(
            llm_callable=my_llm_function,
            raw_text="一段聊天记录...",
            person_id="zhangsan"
        )
    """

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

        if degradation["degradation_needed"]:
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
                degradation_reason=f"材料质量过低({material_quality:.2f})，建议补充素材",
            )

        _, output = self.run_pipeline(
            llm_callable, cleaned_text, material_quality, s1_dict, person_id, input_type, steps
        )
        return output

    def analyze_sync(self, llm_callable, raw_text, person_id=None):
        return self.analyze(llm_callable, raw_text=raw_text, person_id=person_id)
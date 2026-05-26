"""
Orchestrator 基础类

提取 V1/V2 的共享逻辑:
- Skill 实例管理
- Phase 0 分流与降级
- Phase 1-5 标准 Pipeline
- 阈值配置
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .models import (
    InputType,
    OrchestratorOutput,
    PipelineStep,
    Skill1Input,
)
from .skills.skill1_material_retrieval import Skill1MaterialRetrieval
from .skills.skill2_cognitive_profile import Skill2CognitiveProfile
from .skills.skill3_language_emotion import Skill3LanguageEmotion
from .skills.skill4_personality_inference import Skill4PersonalityInference
from .skills.skill5_profile_synthesis import Skill5ProfileSynthesis


class BaseOrchestrator:
    """编排器基类 — 封装 5-Skill Pipeline 的共享逻辑"""

    def __init__(self):
        self.skill1 = Skill1MaterialRetrieval()
        self.skill2 = Skill2CognitiveProfile()
        self.skill3 = Skill3LanguageEmotion()
        self.skill4 = Skill4PersonalityInference()
        self.skill5 = Skill5ProfileSynthesis()

        self.QUALITY_DEGRADE_THRESHOLD = 0.4

    # ========================================================================
    # 共享的 Phase 0-5 Pipeline
    # ========================================================================

    def run_pipeline(
        self,
        llm_callable: Callable[[str, str], str],
        cleaned_text: str,
        material_quality: float,
        s1_dict: dict[str, Any],
        person_id: Optional[str],
        input_type: InputType,
        steps: list[PipelineStep],
    ) -> tuple[dict[str, Any], OrchestratorOutput]:
        """
        执行 Skill 2→3→4→5 的标准 Pipeline。

        返回 (final_output_dict, output) 元组。
        子类可在返回前追加 Phase 6 验证等逻辑。
        """
        # Phase 2
        step2 = PipelineStep(step_name="Skill2_CognitiveProfile", status="running")
        steps.append(step2)
        s2_result = self.skill2.run(llm_callable, cleaned_text, material_quality)
        s2_dict = s2_result.model_dump()
        s2_quality = s2_result.quality
        step2.status = "done"
        step2.quality = s2_quality

        # Phase 3
        step3 = PipelineStep(step_name="Skill3_LanguageEmotion", status="running")
        steps.append(step3)
        s3_result = self.skill3.run(llm_callable, cleaned_text, material_quality)
        s3_dict = s3_result.model_dump()
        s3_quality = s3_result.quality
        step3.status = "done"
        step3.quality = s3_quality

        # Phase 4
        step4 = PipelineStep(step_name="Skill4_PersonalityInference", status="running")
        steps.append(step4)
        s4_result = self.skill4.run(
            llm_callable,
            cognitive_output=s2_dict,
            language_output=s3_dict,
            material_quality=material_quality,
        )
        s4_dict = s4_result.model_dump()
        s4_quality = s4_result.quality
        step4.status = "done"
        step4.quality = s4_quality

        # Phase 5
        step5 = PipelineStep(step_name="Skill5_ProfileSynthesis", status="running")
        steps.append(step5)
        consistency_pass = s4_result.consistency_check.intra_framework == "pass"
        overall_confidence = Skill5ProfileSynthesis.compute_confidence(
            material_quality, s2_quality, s3_quality, s4_quality, consistency_pass
        )
        s5_result = self.skill5.run(
            llm_callable,
            person_id=person_id or "anonymous",
            skill2_output=s2_dict,
            skill3_output=s3_dict,
            skill4_output=s4_dict,
            skill1_output=s1_dict,
            material_quality=material_quality,
        )
        if abs(s5_result.metadata.overall_confidence - overall_confidence) > 0.2:
            s5_result.metadata.overall_confidence = overall_confidence
        step5.status = "done"
        step5.quality = overall_confidence

        output = OrchestratorOutput(
            input_type=input_type,
            pipeline_used=["Skill1", "Skill2", "Skill3", "Skill4", "Skill5"],
            pipeline_steps=steps,
            final_output=s5_result,
            degradation_triggered=False,
        )
        return s2_dict, output

    # ========================================================================
    # Phase 0 工具
    # ========================================================================

    @staticmethod
    def classify_input(
        raw_text: Optional[str],
        person_id: Optional[str],
    ) -> InputType:
        if person_id and raw_text:
            return InputType.HYBRID
        elif person_id and not raw_text:
            return InputType.KNOWLEDGE_ONLY
        else:
            return InputType.TEXT_ONLY

    @staticmethod
    def check_degradation(text: str) -> dict[str, Any]:
        assessment = Skill1MaterialRetrieval.quick_assess(text)
        if assessment["effective_words"] < 200:
            return {
                "degradation_needed": True,
                "reason": f"有效字数仅{assessment['effective_words']}字，不足以进行分析。",
                "level": "critical",
            }
        elif assessment["effective_words"] < 1000:
            return {
                "degradation_needed": False,
                "reason": f"有效字数仅{assessment['effective_words']}字，置信度上限0.6。",
                "level": "warning",
                "max_confidence": 0.6,
            }
        return {"degradation_needed": False}

    # ========================================================================
    # Skill1 公共处理
    # ========================================================================

    @staticmethod
    def run_skill1(
        llm_callable: Callable[[str, str], str],
        raw_text: Optional[str],
        person_id: Optional[str],
        input_type: InputType,
        options: dict[str, Any],
    ) -> tuple[str, float, dict[str, Any]]:
        """
        执行 Skill1，返回 (cleaned_text, quality, s1_dict)。

        统一 V1/V2 的 Skill1 处理逻辑。
        """
        if input_type in (InputType.KNOWLEDGE_ONLY, InputType.HYBRID):
            s1_input = Skill1Input(
                person_id=person_id,
                raw_text=raw_text,
                input_type=input_type,
                options=options,
            )
            s1_result = Skill1MaterialRetrieval().run(llm_callable, s1_input)
            cleaned_text = s1_result.material.get("cleaned_text", raw_text or "")
            material_quality = s1_result.quality_score
            s1_dict = s1_result.model_dump()
        else:
            cleaned_text = Skill1MaterialRetrieval.pre_clean(raw_text or "")
            assessment = Skill1MaterialRetrieval.quick_assess(cleaned_text)
            material_quality = assessment["approx_quality"]
            s1_dict = {
                "material": {"cleaned_text": cleaned_text},
                "quality_score": material_quality,
                "limitations": [],
            }
        return cleaned_text, material_quality, s1_dict
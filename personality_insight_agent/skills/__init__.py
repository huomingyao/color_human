"""
Skills 模块 - 5个独立分析技能

每个 Skill 遵循统一接口:
    - build_system_prompt() -> str
    - build_user_prompt(input_data) -> str
    - parse_output(llm_response: str) -> PydanticModel
    - run(llm_callable, input_data) -> PydanticModel
"""

from .skill1_material_retrieval import Skill1MaterialRetrieval
from .skill2_cognitive_profile import Skill2CognitiveProfile
from .skill3_language_emotion import Skill3LanguageEmotion
from .skill4_personality_inference import Skill4PersonalityInference
from .skill5_profile_synthesis import Skill5ProfileSynthesis

__all__ = [
    "Skill1MaterialRetrieval",
    "Skill2CognitiveProfile",
    "Skill3LanguageEmotion",
    "Skill4PersonalityInference",
    "Skill5ProfileSynthesis",
]

"""
核心数据模型 —— 定义所有 Skill 的输入输出结构

所有模型基于 Pydantic，确保类型安全和 JSON Schema 兼容。
可直接序列化为 OpenAI Function Calling / MCP Tool 的 parameters schema。
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ============================================================================
# 枚举类型
# ============================================================================


class InputType(str, Enum):
    TEXT_ONLY = "text_only"
    KNOWLEDGE_ONLY = "knowledge_only"
    HYBRID = "hybrid"
    COMPARISON = "comparison"


class InfoProcessingStyle(str, Enum):
    INTUITIVE = "intuitive"
    ANALYTICAL = "analytical"
    BALANCED = "balanced"


class DecisionMode(str, Enum):
    DATA_DRIVEN = "data_driven"
    EXPERIENCE_BASED = "experience_based"
    INTUITION_FIRST = "intuition_first"


class RiskOrientation(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class RegulatoryFrame(str, Enum):
    PREVENTION = "prevention"
    PROMOTION = "promotion"


class CreativityType(str, Enum):
    EVOLUTIONARY = "evolutionary"
    REVOLUTIONARY = "revolutionary"


class AttributionStyle(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    BALANCED = "balanced"


class TimeOrientation(str, Enum):
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    PAST_FUTURE = "past_with_future"


class Directness(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Verbosity(str, Enum):
    CONCISE = "concise"
    MODERATE = "moderate"
    VERBOSE = "verbose"


class EmotionExpression(str, Enum):
    OVERT = "overt"
    COVERT = "covert"
    SUPPRESSED = "suppressed"


class MetaphorUsage(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QuestionStyle(str, Enum):
    INQUISITIVE = "inquisitive"
    RHETORICAL = "rhetorical"
    MINIMAL = "minimal"


class AttachmentStyle(str, Enum):
    SECURE = "secure"
    ANXIOUS = "anxious"
    AVOIDANT = "avoidant"


class PDPStyle(str, Enum):
    TIGER = "tiger"
    PEACOCK = "peacock"
    KOALA = "koala"
    OWL = "owl"
    CHAMELEON = "chameleon"


# ============================================================================
# Skill 1: 素材检索与清洗
# ============================================================================


class SourceInfo(BaseModel):
    type: str = Field(..., description="来源类型: wechat_chat, blog_article, interview, etc.")
    period: Optional[str] = Field(None, description="时间范围")
    message_count: Optional[int] = Field(None, description="消息/条目数")
    chars: int = Field(..., description="有效字符数")


class TextChunk(BaseModel):
    id: int
    source: str
    topic: str = Field("general", description="话题标签")
    text: str
    word_count: int


class Skill1Input(BaseModel):
    person_id: Optional[str] = Field(None, description="人物唯一标识")
    raw_text: Optional[str] = Field(None, description="原始文本输入")
    input_type: InputType = Field(InputType.TEXT_ONLY)
    options: Dict[str, Any] = Field(default_factory=dict)


class Skill1Output(BaseModel):
    skill: str = Field(default="Skill1_MaterialRetrieval")
    material: Dict[str, Any] = Field(..., description="清洗后的材料")
    chunks: List[TextChunk] = Field(default_factory=list)
    source_summary: Dict[str, Any] = Field(default_factory=dict)
    quality_score: float = Field(ge=0.0, le=1.0)
    limitations: List[str] = Field(default_factory=list)


# ============================================================================
# Skill 2: 认知与思维特征提取
# ============================================================================


class TripleCheck(BaseModel):
    cross_domain: bool = Field(False, description="跨场景复现: ≥2个不同话题/场景中出现")
    generative: bool = Field(False, description="生成力: 能预测在新问题上的反应")
    exclusive: bool = Field(False, description="排他性: 不是所有人都有的通用特征")


class MentalModel(BaseModel):
    name: str
    description: str
    cross_domain_evidence: List[str] = Field(default_factory=list)
    generativity: str = ""
    exclusivity: str = ""
    triple_check: TripleCheck = Field(default_factory=TripleCheck)


class InfoProcessing(BaseModel):
    style: InfoProcessingStyle = InfoProcessingStyle.BALANCED
    detail_orientation: int = Field(default=3, ge=1, le=5)
    evidence: List[str] = Field(default_factory=list)


class DecisionPattern(BaseModel):
    mode: DecisionMode = DecisionMode.DATA_DRIVEN
    deliberation_level: int = Field(default=3, ge=1, le=5, description="1=快速 5=反复权衡")
    evidence: List[str] = Field(default_factory=list)


class RiskAttitude(BaseModel):
    orientation: RiskOrientation = RiskOrientation.MODERATE
    frame: RegulatoryFrame = RegulatoryFrame.PREVENTION
    evidence: List[str] = Field(default_factory=list)


class Creativity(BaseModel):
    type: CreativityType = CreativityType.EVOLUTIONARY
    playfulness: int = Field(default=3, ge=1, le=5)
    evidence: List[str] = Field(default_factory=list)


class Skill2Output(BaseModel):
    skill: str = Field(default="Skill2_CognitiveProfile")
    cognitive_profile: Dict[str, Any] = Field(default_factory=dict)
    mental_models: List[MentalModel] = Field(default_factory=list)
    attribution_style: AttributionStyle = AttributionStyle.BALANCED
    time_orientation: TimeOrientation = TimeOrientation.PRESENT
    quality: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: List[str] = Field(default_factory=list)


# ============================================================================
# Skill 3: 语言风格与情感模式分析
# ============================================================================


class SentenceFingerprint(BaseModel):
    """借鉴 Nuwa-Skill 的句式指纹"""
    avg_sentence_length: float = 0.0
    question_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    analogy_density: float = Field(default=0.0, description="每千字类比数")
    first_person_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    certainty_tone: str = "moderate"


class StyleTags(BaseModel):
    formality: int = Field(default=3, ge=1, le=5)
    abstractness: int = Field(default=3, ge=1, le=5)
    cautiousness: int = Field(default=3, ge=1, le=5)
    narrative_data: int = Field(default=3, ge=1, le=5, description="1=纯叙事 5=纯数据")


class Emotion(BaseModel):
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=0.0, le=1.0)
    dominance: float = Field(default=0.0, ge=0.0, le=1.0)
    expression_tendency: EmotionExpression = EmotionExpression.COVERT


class Rhetoric(BaseModel):
    metaphor_usage: MetaphorUsage = MetaphorUsage.MEDIUM
    question_style: QuestionStyle = QuestionStyle.MINIMAL
    humor_tendency: int = Field(default=1, ge=1, le=5)


class Skill3Output(BaseModel):
    skill: str = Field(default="Skill3_LanguageEmotion")
    language_style: Dict[str, Any] = Field(default_factory=dict)
    sentence_fingerprint: SentenceFingerprint = Field(default_factory=SentenceFingerprint)
    style_tags: StyleTags = Field(default_factory=StyleTags)
    emotion: Emotion = Field(default_factory=Emotion)
    rhetoric: Rhetoric = Field(default_factory=Rhetoric)
    signature_patterns: List[str] = Field(default_factory=list)
    quality: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: List[str] = Field(default_factory=list)


# ============================================================================
# Skill 4: 性格特质推断
# ============================================================================


class BigFiveScore(BaseModel):
    score: int = Field(default=3, ge=1, le=5)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    signals: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class BigFive(BaseModel):
    openness: BigFiveScore = Field(default_factory=BigFiveScore)
    conscientiousness: BigFiveScore = Field(default_factory=BigFiveScore)
    extraversion: BigFiveScore = Field(default_factory=BigFiveScore)
    agreeableness: BigFiveScore = Field(default_factory=BigFiveScore)
    neuroticism: BigFiveScore = Field(default_factory=BigFiveScore)


class MBTI(BaseModel):
    type: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    breakdown: Dict[str, int] = Field(default_factory=dict)
    alternative: str = ""


class Motivation(BaseModel):
    achievement: Dict[str, Any] = Field(default_factory=lambda: {"score": 3, "rank": 1})
    affiliation: Dict[str, Any] = Field(default_factory=lambda: {"score": 3, "rank": 2})
    power: Dict[str, Any] = Field(default_factory=lambda: {"score": 3, "rank": 3})


class PDPResult(BaseModel):
    style: str = ""
    sub_style: str = ""
    tiger_score: int = Field(default=3, ge=1, le=5)
    peacock_score: int = Field(default=3, ge=1, le=5)
    koala_score: int = Field(default=3, ge=1, le=5)
    owl_score: int = Field(default=3, ge=1, le=5)
    chameleon_score: int = Field(default=3, ge=1, le=5)


class ConsistencyCheck(BaseModel):
    intra_framework: str = "pass"
    inter_framework: str = ""
    signal_coverage: str = ""
    contradictions: List[str] = Field(default_factory=list)


class Skill4Output(BaseModel):
    skill: str = Field(default="Skill4_PersonalityInference")
    personality: Dict[str, Any] = Field(default_factory=dict)
    big_five: BigFive = Field(default_factory=BigFive)
    mbti: MBTI = Field(default_factory=MBTI)
    motivation: Motivation = Field(default_factory=Motivation)
    pdp: PDPResult = Field(default_factory=PDPResult)
    consistency_check: ConsistencyCheck = Field(default_factory=ConsistencyCheck)
    quality: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: Dict[str, str] = Field(default_factory=dict)


# ============================================================================
# Skill 5: 画像合成与报告生成
# ============================================================================


class CoreProfile(BaseModel):
    thinking_style: str = ""
    personality_snapshot: str = ""
    communication_essence: str = ""
    core_motivation: str = ""


class Contradiction(BaseModel):
    type: str = Field(..., description="temporal / contextual / essential_tension")
    description: str = ""
    evidence: List[str] = Field(default_factory=list)


class HonestyBoundary(BaseModel):
    """借鉴 Nuwa-Skill 的诚实边界"""
    known: List[str] = Field(default_factory=list)
    uncertain: List[str] = Field(default_factory=list)
    unknown: List[str] = Field(default_factory=list)
    material_limitations: List[str] = Field(default_factory=list)


class AnalysisMetadata(BaseModel):
    analysis_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = "2.0"
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    material_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Insights(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    growth_areas: List[str] = Field(default_factory=list)
    blind_spots: List[str] = Field(default_factory=list)


class Skill5Output(BaseModel):
    skill: str = Field(default="Skill5_ProfileSynthesis")
    person_id: str = ""
    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata)
    core_profile: CoreProfile = Field(default_factory=CoreProfile)
    cognitive: Dict[str, Any] = Field(default_factory=dict)
    language_emotion: Dict[str, Any] = Field(default_factory=dict)
    personality: Dict[str, Any] = Field(default_factory=dict)
    insights: Insights = Field(default_factory=Insights)
    contradictions: List[Contradiction] = Field(default_factory=list)
    honesty_boundary: HonestyBoundary = Field(default_factory=HonestyBoundary)
    report: str = ""


# ============================================================================
# Orchestrator 输入输出
# ============================================================================


class OrchestratorInput(BaseModel):
    person_id: Optional[str] = Field(None)
    raw_text: Optional[str] = Field(None)
    input_type: InputType = InputType.TEXT_ONLY
    compare_ids: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)


class PipelineStep(BaseModel):
    step_name: str
    status: str = Field(default="pending")  # pending / running / done / failed
    quality: float = Field(default=0.0, ge=0.0, le=1.0)


class OrchestratorOutput(BaseModel):
    input_type: InputType
    pipeline_used: List[str] = Field(default_factory=list)
    pipeline_steps: List[PipelineStep] = Field(default_factory=list)
    final_output: Optional[Skill5Output] = None
    degradation_triggered: bool = False
    degradation_reason: str = ""
    errors: List[str] = Field(default_factory=list)

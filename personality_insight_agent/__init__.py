"""
Personality Insight Agent V2 - 基于心理学框架的人物画像分析系统

借鉴 Nuwa-Skill 的 Pipeline 架构，集成三大前沿组件:
- Corpus2Skill: 层级知识目录树 + 离线流水线
- Agentic RAG: 自主规划 → 深度检索 → 反思迭代
- 双引擎验证: 事实核查 + 外部仲裁

快速使用 (纯文本):
    from personality_insight_agent import analyze_personality
    result = analyze_personality(my_llm_func, chat_text, "person_name")

快速使用 (含私有知识库):
    from personality_insight_agent import analyze_with_private_corpus
    result = analyze_with_private_corpus(
        llm_callable=my_llm,
        corpus_dir="d:/my_docs/",
        person_id="zhangsan",
        vector_search_fn=my_faiss_search,
    )
"""

__version__ = "2.0.0"

# 便捷函数
from .main import analyze_personality, quick_analyze, compare_personalities

# 核心类
from .orchestrator import PersonalityInsightOrchestrator
from .orchestrator_v2 import PersonalityInsightOrchestratorV2
from .models import Skill5Output

# 三大组件
from .corpus2skill import (
    KnowledgeTreeNode,
    CorpusIndex,
    SkillDirectoryTree,
    CorpusTreeBuilder,
    KnowledgeNavigator,
)
from .agentic_rag import (
    ResearchPlan,
    ResearchTask,
    AgenticResearchAgent,
    ResearchPlanner,
    DeepRetriever,
    ResearchReflector,
)
from .verification import (
    VerificationResult,
    FactCheckItem,
    ArbitrationResult,
    FactCheckAgent,
    DualEngineArbitrator,
)

# V2 完整分析入口
from .main import analyze_with_private_corpus


__all__ = [
    # 版本
    "__version__",
    # V1
    "PersonalityInsightOrchestrator",
    "analyze_personality",
    "quick_analyze",
    "compare_personalities",
    # V2
    "PersonalityInsightOrchestratorV2",
    "analyze_with_private_corpus",
    "Skill5Output",
    # Corpus2Skill
    "KnowledgeTreeNode",
    "CorpusIndex",
    "SkillDirectoryTree",
    "CorpusTreeBuilder",
    "KnowledgeNavigator",
    # Agentic RAG
    "ResearchPlan",
    "ResearchTask",
    "AgenticResearchAgent",
    "ResearchPlanner",
    "DeepRetriever",
    "ResearchReflector",
    # 双引擎验证
    "VerificationResult",
    "FactCheckItem",
    "ArbitrationResult",
    "FactCheckAgent",
    "DualEngineArbitrator",
]

"""
Agentic RAG — 组件二：升级"研究者"

将女娲的蒸馏Agent从被动的"收集者"升级为能自主规划、使用工具、并反思迭代的"智能体"。

├── models.py         — Agent 规划/反思/状态数据模型
├── planner.py        — 任务分解 + 检索路径规划 (自主规划)
├── retriever.py      — 深度检索 (多跳、迭代、重排序)
├── reflector.py      — 反思与冲突检测 (反思迭代)
└── research_agent.py — 升级版研究Agent (整合规划-检索-反思循环)
"""

from .models import (
    ResearchPlan,
    ResearchTask,
    RetrievalResult,
    ReflectionResult,
    AgentState,
    AgentAction,
    ResearchLoopConfig,
)
from .planner import ResearchPlanner
from .retriever import DeepRetriever
from .reflector import ResearchReflector
from .research_agent import AgenticResearchAgent

__all__ = [
    "ResearchPlan",
    "ResearchTask",
    "RetrievalResult",
    "ReflectionResult",
    "AgentState",
    "AgentAction",
    "ResearchLoopConfig",
    "ResearchPlanner",
    "DeepRetriever",
    "ResearchReflector",
    "AgenticResearchAgent",
]

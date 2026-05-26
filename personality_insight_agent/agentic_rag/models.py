"""
Agentic RAG 数据模型

定义自主规划 → 深度检索 → 反思迭代循环中所有的数据类型。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# 枚举
# ============================================================================


class AgentState(str, Enum):
    """研究Agent的状态机"""
    IDLE = "idle"
    PLANNING = "planning"         # 正在分解任务
    RETRIEVING = "retrieving"     # 正在检索
    REFLECTING = "reflecting"     # 正在反思
    REPLANNING = "replanning"     # 根据反思重新规划
    SYNTHESIZING = "synthesizing" # 综合结果
    DONE = "done"
    ERROR = "error"


class AgentAction(str, Enum):
    """Agent 可执行的动作"""
    PLAN = "plan"              # 制定研究计划
    SEARCH = "search"          # 向量检索
    NAVIGATE = "navigate"      # 知识树导航
    READ_CHUNK = "read_chunk"  # 读取具体文档块
    REFLECT = "reflect"        # 反思当前结果
    REPLAN = "replan"          # 调整计划
    SYNTHESIZE = "synthesize"  # 综合输出
    STOP = "stop"              # 信息足够，停止


class ResearchDepth(str, Enum):
    """研究深度"""
    QUICK = "quick"       # 单次检索，不要循环
    STANDARD = "standard" # 2-3轮循环
    DEEP = "deep"         # 最多5轮，有冲突必须迭代


# ============================================================================
# 研究计划
# ============================================================================


class ResearchTask(BaseModel):
    """研究计划中的单个任务"""
    task_id: str = Field(..., description="任务唯一标识")
    description: str = Field(..., description="任务描述")

    # 检索目标
    target_categories: List[str] = Field(
        default_factory=list,
        description="目标内容分类: writings/conversations/expressions/external/decisions/timeline"
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="具体的检索查询列表"
    )

    # 导航路径
    knowledge_path: List[str] = Field(
        default_factory=list,
        description="在知识树中的导航路径，如 ['root', 'decisions', 'doc_001']"
    )

    # 依赖
    depends_on: List[str] = Field(
        default_factory=list,
        description="依赖的task_id列表，必须先完成依赖任务"
    )
    priority: int = Field(default=5, description="优先级 1(highest)-10(lowest)")

    # 状态
    completed: bool = False
    result_summary: str = ""


class ResearchPlan(BaseModel):
    """完整的研究计划 — Planner 的输出"""
    plan_id: str = Field(default_factory=lambda: f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    original_query: str = Field(..., description="原始研究问题/指令")

    # 分解后的任务
    tasks: List[ResearchTask] = Field(default_factory=list, description="任务列表")

    # 策略
    research_depth: ResearchDepth = Field(default=ResearchDepth.STANDARD)
    estimated_rounds: int = Field(default=3, description="预计检索轮数")
    max_rounds: int = Field(default=5, description="最大轮数上限")

    # 知识源
    knowledge_tree_id: Optional[str] = Field(default=None, description="Corpus2Skill 树ID")
    enable_web_search: bool = Field(default=False, description="是否启用网络搜索补充")

    # 元信息
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    rationale: str = Field(default="", description="规划理由")


# ============================================================================
# 检索结果
# ============================================================================


class RetrievalResult(BaseModel):
    """单次检索的结果"""
    query: str = Field(..., description="检索查询")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="检索到的文档块")

    # 来源统计
    sources: List[str] = Field(default_factory=list, description="来源文件列表")
    categories_covered: List[str] = Field(default_factory=list, description="覆盖的内容分类")

    # 质量评估
    relevance_score: float = Field(default=0.0, description="整体相关性评分 (0-1)")
    information_density: float = Field(default=0.0, description="信息密度 (0-1)")
    novelty_score: float = Field(default=0.0, description="新颖性评分 (vs 已有结果, 0-1)")

    # 元数据
    retrieval_method: str = Field(default="vector", description="vector/navigate/hybrid")
    retrieval_time_ms: int = Field(default=0)


# ============================================================================
# 反思结果
# ============================================================================


class ReflectionResult(BaseModel):
    """Reflector 的反思输出"""
    # 整体评估
    information_sufficient: bool = Field(default=False, description="信息是否足够")
    quality_score: float = Field(default=0.0, description="整体质量 (0-1)")

    # 冲突检测
    conflicts_found: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="""
        发现的冲突列表:
        [{
            "type": "temporal/spatial/essential",
            "description": "冲突描述",
            "source_a": "来源A的引用",
            "source_b": "来源B的引用",
            "severity": "high/medium/low"
        }]
        """
    )

    # 信息缺口
    gaps: List[Dict[str, str]] = Field(
        default_factory=list,
        description="[{'dimension': ..., 'description': ..., 'suggested_query': ...}]"
    )

    # 下一步行动
    should_continue: bool = Field(default=True, description="是否应继续检索")
    suggested_actions: List[AgentAction] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list)

    # 反思日志
    reflection_log: str = Field(default="", description="反思过程的文字记录")


# ============================================================================
# 研究循环配置
# ============================================================================


class ResearchLoopConfig(BaseModel):
    """研究循环的配置参数"""
    max_rounds: int = Field(default=5, description="最大检索轮数")
    min_relevance: float = Field(default=0.4, description="最低相关性阈值")
    min_information_density: float = Field(default=0.3, description="最低信息密度")
    novelty_threshold: float = Field(default=0.2, description="新颖性阈值: 低于此值认为重复")
    conflict_detection: bool = Field(default=True, description="是否检测冲突")
    gap_analysis: bool = Field(default=True, description="是否分析信息缺口")
    early_stop_quality: float = Field(default=0.85, description="达到此质量分可提前停止")

    # 并行设置
    parallel_agents: int = Field(default=1, description="并行检索Agent数量(此处为单Agent内循环)")

    # 超时
    timeout_seconds: int = Field(default=300, description="单Agent超时(秒)")

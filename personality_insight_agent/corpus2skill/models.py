"""
Corpus2Skill 数据模型

定义层级知识目录树中所有节点类型:
- SkillDirectoryTree: 整棵树的容器
- CorpusIndex (INDEX.md): 中间索引节点，提供子节点概览
- KnowledgeTreeNode: 叶子节点，对应具体文档
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# 枚举定义
# ============================================================================


class NodeType(str, Enum):
    ROOT = "root"          # SKILL.md — 整棵树的根
    INDEX = "index"        # INDEX.md — 中间索引节点
    LEAF = "leaf"          # 具体文档节点


class ContentCategory(str, Enum):
    """内容分类 — 与 Nuwa 的 6 Agent 维度对齐"""
    WRITINGS = "writings"           # 著作与系统思考
    CONVERSATIONS = "conversations" # 对话与即兴思考
    EXPRESSIONS = "expressions"     # 碎片表达与风格
    EXTERNAL = "external_views"     # 他者视角
    DECISIONS = "decisions"         # 决策记录
    TIMELINE = "timeline"           # 时间线
    GENERIC = "generic"             # 通用/未分类


class EvidenceLevel(str, Enum):
    """证据等级"""
    PRIMARY = "primary"       # 一手 : 本人著作、演讲、访谈原文
    SECONDARY = "secondary"   # 二手 : 他人转述、评论
    TERTIARY = "tertiary"     # 推测 : AI推理、统计分析


# ============================================================================
# 知识树节点
# ============================================================================


class KnowledgeTreeNode(BaseModel):
    """
    知识树中的单个节点。

    可以是根节点(SKILL.md)、中间索引节点(INDEX.md)或叶子节点(文档)。
    叶子节点持有指向 FAISS 向量库中对应文档块的 ID 列表。
    """
    node_id: str = Field(..., description="唯一标识，如 'writings/book_01'")
    node_type: NodeType = Field(..., description="节点类型")
    title: str = Field(..., description="节点标题")
    description: str = Field(default="", description="节点一句话描述")

    # 层级关系
    parent_id: Optional[str] = Field(default=None, description="父节点ID")
    children: List[KnowledgeTreeNode] = Field(default_factory=list, description="子节点列表")

    # 元数据
    category: ContentCategory = Field(default=ContentCategory.GENERIC)
    evidence_level: EvidenceLevel = Field(default=EvidenceLevel.PRIMARY)
    source_path: Optional[str] = Field(default=None, description="原始文件路径")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # 叶子节点特有: 指向向量库中 chunk ID
    chunk_ids: List[str] = Field(default_factory=list, description="FAISS chunk ID 列表")
    estimated_words: int = Field(default=0, description="估算字数")

    # 统计 (中间节点/根节点)
    summary: str = Field(default="", description="此节点的内容摘要 (非叶子时自动生成)")
    key_topics: List[str] = Field(default_factory=list, description="关键主题/标签")
    key_entities: List[str] = Field(default_factory=list, description="关键实体 (人名/地名/术语)")


class CorpusIndex(BaseModel):
    """
    INDEX.md 的内容模型 — 中间索引节点的详细概览。

    记录其下所有子节点的聚合信息，供 Agent 决定"下一步去哪里"。
    """
    index_id: str = Field(..., description="索引标识")
    title: str = Field(..., description="索引标题")

    # 子节点摘要
    total_children: int = Field(default=0)
    total_words: int = Field(default=0)
    category_distribution: Dict[str, int] = Field(default_factory=dict)

    # 导航提示 — Agent 用来决策的关键信息
    navigation_hints: List[str] = Field(
        default_factory=list,
        description="关键导航提示，如 '商业决策相关内容集中在decisions/子目录下'"
    )

    # 子节点概览 (每个子节点一行)
    children_overview: List[Dict[str, str]] = Field(
        default_factory=list,
        description="[{'id': ..., 'title': ..., 'summary': ..., 'word_count': ...}]"
    )

    # 交叉引用
    cross_references: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="主题→相关节点ID列表的映射"
    )


class SkillDirectoryTree(BaseModel):
    """
    完整的层级知识目录树 — Corpus2Skill 离线流水线的最终产物。

    ├── SKILL.md (root)
    │   ├── INDEX.md — 著作与系统思考
    │   │   ├── doc_001: 《XXX》第一章
    │   │   └── doc_002: YYY 博客合集
    │   ├── INDEX.md — 对话与即兴思考
    │   │   ├── doc_003: 播客访谈 transcript
    │   │   └── doc_004: 内部会议记录
    │   ├── INDEX.md — 决策记录
    │   │   ├── doc_005: 2024年战略会议纪要
    │   │   └── doc_006: 投资备忘录
    │   └── ...
    """
    tree_id: str = Field(..., description="知识树唯一ID，通常为 person_id")
    root: KnowledgeTreeNode = Field(..., description="根节点")

    # 全局索引
    index_registry: Dict[str, CorpusIndex] = Field(
        default_factory=dict,
        description="所有 INDEX.md 的注册表，key=index_id"
    )

    # 全局文档注册表
    doc_registry: Dict[str, KnowledgeTreeNode] = Field(
        default_factory=dict,
        description="所有叶子节点的注册表，key=node_id，用于 O(1) 查找"
    )

    # 元信息
    total_documents: int = Field(default=0)
    total_words: int = Field(default=0)
    build_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_directory: str = Field(default="", description="原始资料目录路径")

    # 向量库映射: category → vector_store_path
    vector_store_map: Dict[str, str] = Field(
        default_factory=dict,
        description="内容分类 → FAISS 向量库路径的映射"
    )


# ============================================================================
# 导航上下文
# ============================================================================


class NavigationContext(BaseModel):
    """
    Agent 导航时的上下文窗口。

    记录 Agent 当前在知识树中的位置、历史路径，支持后退和跳转。
    """
    current_node_id: str = Field(..., description="当前所在节点")
    path_history: List[str] = Field(default_factory=list, description="访问路径历史")

    # 当前视图: Agent "看到"的内容
    visible_children: List[Dict[str, str]] = Field(
        default_factory=list,
        description="当前节点的直接子节点概览"
    )

    # 检索到的内容块
    retrieved_chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="从当前上下文检索到的文档块"
    )

    # 导航建议
    suggested_next: List[str] = Field(
        default_factory=list,
        description="建议下一步查看的节点ID列表"
    )

    def record_visit(self, node_id: str):
        """记录导航路径"""
        self.path_history.append(self.current_node_id)
        self.current_node_id = node_id

    def go_back(self) -> Optional[str]:
        """返回上一节点"""
        if self.path_history:
            return self.path_history.pop()
        return None


class NavigationPlan(BaseModel):
    """
    Agentic RAG Planner 生成的导航计划。

    将高级指令(如"重大商业决策")分解为知识树中的具体导航路径。
    """
    instruction: str = Field(..., description="原始指令")

    # 分解后的子任务，每个子任务对应知识树的一条路径
    steps: List[NavigationStep] = Field(
        default_factory=list,
        description="导航步骤序列"
    )

    estimated_depth: int = Field(default=3, description="预计遍历深度")
    fallback_nodes: List[str] = Field(
        default_factory=list,
        description="降级方案: 当主路径信息不足时的备选节点"
    )


class NavigationStep(BaseModel):
    """导航计划中的单步"""
    step_order: int = Field(..., description="步骤序号")
    category: ContentCategory = Field(..., description="目标内容分类")
    target_nodes: List[str] = Field(default_factory=list, description="目标节点ID列表")
    search_query: str = Field(default="", description="在此分类下的检索查询")
    rationale: str = Field(default="", description="为什么需要这一步")
    is_optional: bool = Field(default=False, description="是否为可选步骤")

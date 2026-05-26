"""
Corpus2Skill — 组件一：构筑"记忆宫殿"

核心理念：从"检索"到"导航"。将用户提供的私有资料自动处理为层级化知识目录树，
让 Agent 能"看到"知识的宏观结构，按逻辑路径高效导航。

├── builder.py    — 离线流水线：批量文档 → 层级知识树
├── navigator.py  — 在线导航：路径规划 + 上下文窗口管理
└── models.py     — 知识树节点/索引数据模型
"""

from .models import KnowledgeTreeNode, CorpusIndex, SkillDirectoryTree
from .builder import CorpusTreeBuilder
from .navigator import KnowledgeNavigator

__all__ = [
    "KnowledgeTreeNode",
    "CorpusIndex",
    "SkillDirectoryTree",
    "CorpusTreeBuilder",
    "KnowledgeNavigator",
]

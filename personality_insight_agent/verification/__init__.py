"""
双引擎验证 — 组件三：确保客观性

核心理念: 引入外部事实核查，与女娲原生"三重验证"机制形成"双引擎驱动"的闭环。

├── models.py      — 仲裁记录/验证报告数据模型
├── fact_checker.py — 外部事实核查 Agent (RAG回知识库核实)
└── arbitrator.py   — 内外部冲突仲裁机制
"""

from .models import (
    VerificationResult,
    FactCheckItem,
    ArbitrationResult,
    ConflictResolution,
    ArbitrationStrategy,
)
from .fact_checker import FactCheckAgent
from .arbitrator import DualEngineArbitrator

__all__ = [
    "VerificationResult",
    "FactCheckItem",
    "ArbitrationResult",
    "ConflictResolution",
    "ArbitrationStrategy",
    "FactCheckAgent",
    "DualEngineArbitrator",
]

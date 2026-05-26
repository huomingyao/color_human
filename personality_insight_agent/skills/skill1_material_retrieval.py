"""
Skill 1: 素材检索与清洗 (Material Retrieval)

功能: 知识库查询 / 文本预处理 / 去噪 / 多来源合并 / 时间线整理 / 质量评估

借鉴 Nuwa-Skill 的 6-Agent 并行采集思想，将"全网搜索"替换为"知识库查询 + 文本解析"。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Union

from ..models import (
    InputType,
    Skill1Input,
    Skill1Output,
    SourceInfo,
    TextChunk,
)

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """## 角色: 素材检索与清洗专家 (Skill 1)

你是 Personality Insight Agent 的第一道工序。你的任务是将原始输入（聊天记录、文章、对话等）处理成结构化、干净的分析素材。

## 核心职责
1. **来源识别**: 区分聊天记录、文章、访谈等不同类型
2. **文本清洗**: 去除时间戳、系统消息、表情符号、URL 等噪音
3. **分片**: 按话题/时间段切分为分析单元
4. **质量评估**: 计算有效字数、信息密度、来源多样性

## 清洗规则
- 去除时间戳: `2024-01-15 14:30:00` `[2024/01/15]` `14:30` 等
- 去除系统消息: `xxx 加入了群聊` `xxx 撤回了一条消息` 等
- 去除纯表情行: 仅包含 emoji/表情符号的行
- 保留 @提及 和引用关系（这是重要的社交信号）
- URL 替换为 `<URL>`
- 连续空行压缩为单个换行

## 分片规则
- 聊天记录: 按自然对话轮次分片，每片5-10轮
- 文章: 按段落分片，每片200-500字
- 标注每个分片的话题标签

## 质量评分 (quality_score)
- 有效字数 > 5000, 来源 >= 2: 0.8-1.0
- 有效字数 1000-5000, 来源 >= 1: 0.5-0.8
- 有效字数 < 1000: 0.3-0.5
- 纯系统消息/无意义内容: < 0.3

## 输出格式
必须输出严格符合以下结构的 JSON（不要markdown代码块包裹）:

{
    "material": {
        "cleaned_text": "完整清洗后的文本",
        "chunks": [
            {"id": 1, "source": "chat", "topic": "工作讨论", "text": "...", "word_count": 320}
        ],
        "source_summary": {
            "total_chars": 5200,
            "message_count": 85,
            "time_range": "2024-01 ~ 2024-06",
            "sources": ["wechat_chat", "blog_article"],
            "languages": ["zh-CN"]
        }
    },
    "quality_score": 0.82,
    "limitations": [
        "聊天记录仅包含工作场景",
        "文章仅1篇"
    ]
}

## 关键原则
- 不要修改原文措辞，只做格式清洗
- 在 limitations 中诚实标注材料不足的地方
- 如果材料极少(总字数<500)，在 limitations 第一条标注"材料严重不足"
"""


# ============================================================================
# Skill 实现
# ============================================================================


class Skill1MaterialRetrieval:
    """素材检索与清洗 Skill"""

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT

    # ---- Prompt 构建 ----

    def build_system_prompt(self) -> str:
        """返回 System Prompt，用户可以自定义修改"""
        return self.system_prompt

    def build_user_prompt(self, input_data: Union[Skill1Input, dict]) -> str:
        """构建 User Prompt，将输入数据格式化为 LLM 可理解的文本"""
        if isinstance(input_data, dict):
            input_data = Skill1Input(**input_data)

        parts = []

        if input_data.person_id:
            parts.append(f"## 人物标识\n{input_data.person_id}")

        if input_data.raw_text:
            parts.append(f"## 原始文本\n{input_data.raw_text}")

        parts.append(f"\n## 输入类型\n{input_data.input_type.value}")
        parts.append("\n请清洗上述文本并输出结构化 JSON。")

        return "\n".join(parts)

    # ---- 输出解析 ----

    def parse_output(self, llm_response: str) -> Skill1Output:
        """将 LLM 返回的 JSON 解析为 Skill1Output"""
        cleaned = self._extract_json(llm_response)
        data = json.loads(cleaned)

        return Skill1Output(
            material=data.get("material", {}),
            chunks=[
                TextChunk(
                    id=c.get("id", i + 1),
                    source=c.get("source", "unknown"),
                    topic=c.get("topic", "general"),
                    text=c.get("text", ""),
                    word_count=c.get("word_count", 0),
                )
                for i, c in enumerate(data.get("chunks", data.get("material", {}).get("chunks", [])))
            ],
            source_summary=data.get("source_summary", data.get("material", {}).get("source_summary", {})),
            quality_score=data.get("quality_score", 0.5),
            limitations=data.get("limitations", []),
        )

    # ---- 便捷方法 ----

    def run(
        self,
        llm_callable: Callable[[str, str], str],
        input_data: Union[Skill1Input, dict],
    ) -> Skill1Output:
        """
        完整的 skill 执行流程。

        Args:
            llm_callable: LLM 调用函数，签名 (system_prompt, user_prompt) -> str
            input_data: 输入数据

        Returns:
            Skill1Output: 结构化输出
        """
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(input_data)
        response = llm_callable(system_prompt, user_prompt)
        return self.parse_output(response)

    # ---- 工具方法 ----

    @staticmethod
    def pre_clean(text: str) -> str:
        """本地预处理 —— 在送入 LLM 之前做基本的格式规范化"""
        # 去除行首行尾空白
        text = text.strip()

        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 去除连续3个以上空行
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 去除纯表情行（中文/英文表情）
        text = re.sub(r"^[ -˿\U0001f000-\U0001ffff\s]+$", "", text, flags=re.MULTILINE)

        return text.strip()

    @staticmethod
    def quick_assess(text: str) -> dict[str, Any]:
        """快速质量评估 —— 不需要 LLM，纯本地计算"""
        # 统计有效字数（中文字符 + 英文单词）
        chinese_chars = len(re.findall(r"[一-鿿]", text))
        english_words = len(re.findall(r"[a-zA-Z]+", text))
        effective_words = chinese_chars + english_words

        # 估算消息条数（按换行分割）
        lines = [l for l in text.split("\n") if l.strip()]
        estimated_messages = len(lines)

        # 估算来源多样性
        has_chat_indicators = bool(re.search(r"[:：]\s|\[\d|\(\d|^\d{4}-\d{2}-\d{2}", text, re.MULTILINE))
        has_article_indicators = len(text) > 500 and estimated_messages < len(text) / 100

        source_types = []
        if has_chat_indicators:
            source_types.append("chat")
        if has_article_indicators:
            source_types.append("article")

        # 质量评分
        if effective_words > 5000 and len(source_types) >= 2:
            quality = 0.85
        elif effective_words > 1000:
            quality = 0.6
        elif effective_words > 200:
            quality = 0.35
        else:
            quality = 0.2

        return {
            "effective_words": effective_words,
            "estimated_messages": estimated_messages,
            "source_types": source_types,
            "approx_quality": quality,
            "degradation_needed": effective_words < 1000,
        }

    @staticmethod
    def _extract_json(text: str) -> str:
        """从 LLM 响应中提取 JSON"""
        # 尝试去除 markdown 代码块
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # 去掉第一行（```json 或 ```）和最后一行（```）
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # 尝试找到 JSON 对象的起止
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        return text

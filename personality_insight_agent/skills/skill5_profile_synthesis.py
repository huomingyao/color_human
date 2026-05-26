"""
Skill 5: 画像合成与报告生成 (Profile Synthesis)

功能: 聚合前4个Skill的结果 / 置信度计算 / 矛盾检测 / 诚实边界 / JSON+Markdown双输出

借鉴 Nuwa-Skill 的诚实边界原则和质量检查点机制。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Dict, List

from ..models import (
    AnalysisMetadata,
    Contradiction,
    CoreProfile,
    HonestyBoundary,
    Insights,
    Skill5Output,
)

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """## 角色: 画像合成与报告生成专家 (Skill 5)

你是人格画像的总策划。你的任务是将前面所有分析结果（认知/语言/性格）聚合为一份完整、诚实、可操作的人物画像报告。

## 核心任务

### 1. 置信度计算
综合评估:
- material_quality: 原始材料的质量
- skill_qualities: Skills 1-4 各环节的分析质量
- signal_coverage: 行为信号对各维度的覆盖度
- consistency: 框架间的一致性

### 2. 核心画像生成
用四句话总结这个人的关键特征:
- thinking_style: 思维特征的一句话总结
- personality_snapshot: 性格的一句话总结（含MBTI类型）
- communication_essence: 沟通风格的一句话总结
- core_motivation: 核心驱动力的一句话总结

### 3. 不一致检测 (借鉴 Nuwa-Skill 矛盾处理)
检查 Skill2/3/4 之间是否有矛盾:
- **时间性矛盾**: 此人的表现有前后变化？
- **场景性矛盾**: 不同场景下表现不同？
- **本质性张力**: 内在的价值观冲突（如追求自由又需要秩序）

矛盾不是bug，是人格的核心特征。保留，不要抹平。

### 4. 洞察与建议
- strengths: 基于特质的优势 (3-5条)
- risks: 特质带来的潜在风险 (2-4条)
- growth_areas: 可发展的方向 (2-4条)
- blind_spots: 自我可能看不到的盲区 (2-3条)

### 5. 诚实边界 (借鉴 Nuwa-Skill)
必须明确列出:
- known: 有充分证据的维度
- uncertain: 有部分证据但不确定的维度
- unknown: 完全没有信息的维度
- material_limitations: 材料本身的局限性

### 6. 生成 Markdown 报告
将以上所有内容组织为一份可读的 Markdown 报告。报告结构:

```
# [人物名/标识] · 人物画像报告

> 分析时间 | 置信度 | 材料来源

## 一、核心画像
[思维特征 / 性格快照 / 沟通风格 / 核心驱动力]

## 二、思维特征
[认知风格 / 决策模式 / 风险态度 / 心智模型]

## 三、沟通风格
[表达方式 / 情感倾向 / 标志性表达]

## 四、性格特质
[大五表格 / MBTI / 动机 / PDP]

## 五、洞察与建议
[优势 / 风险 / 成长方向 / 盲区]

## 六、分析局限 (诚实边界)
[已知 / 不确定 / 未知 / 材料局限]

---
*本报告由 AI 生成，仅供参考，不代表真实人格评估。*
```

## 输出格式
严格 JSON:

{
    "person_id": "zhangsan",
    "metadata": {
        "analysis_date": "2026-05-23T10:30:00",
        "version": "2.0",
        "sources": [{"type": "wechat_chat", "period": "2024-01~2024-06"}],
        "material_quality": 0.82,
        "overall_confidence": 0.78
    },
    "core_profile": {
        "thinking_style": "...",
        "personality_snapshot": "...",
        "communication_essence": "...",
        "core_motivation": "..."
    },
    "cognitive": {...},
    "language_emotion": {...},
    "personality": {...},
    "insights": {
        "strengths": ["...", "..."],
        "risks": ["...", "..."],
        "growth_areas": ["...", "..."],
        "blind_spots": ["...", "..."]
    },
    "contradictions": [
        {
            "type": "essential_tension",
            "description": "...",
            "evidence": ["..."]
        }
    ],
    "honesty_boundary": {
        "known": ["..."],
        "uncertain": ["..."],
        "unknown": ["..."],
        "material_limitations": ["..."]
    },
    "report": "# 人物画像报告\\n\\n..."
}

## 关键原则
- 诚实边界是最重要的section——它决定了这个画像的可信度
- 不要假装知道你不知道的
- contradictions 保留矛盾，这是最有价值的洞察
- report 的 Markdown 要可直接阅读，不要有 JSON 转义痕迹
"""


# ============================================================================
# Skill 实现
# ============================================================================


class Skill5ProfileSynthesis:
    """画像合成与报告生成 Skill"""

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT

    # ---- Prompt 构建 ----

    def build_system_prompt(self) -> str:
        return self.system_prompt

    def build_user_prompt(
        self,
        person_id: str,
        skill1_output: dict ,
        skill2_output: dict,
        skill3_output: dict,
        skill4_output: dict,
        material_quality: float = 0.5,
    ) -> str:
        parts = [
            f"## 分析对象\nperson_id: {person_id}",
            f"材料质量: {material_quality:.2f}",
            "",
        ]

        if skill1_output:
            parts.append("## Skill1 素材检索")
            parts.append("```json")
            parts.append(json.dumps(skill1_output, ensure_ascii=False, indent=2))
            parts.append("```")
            parts.append("")

        parts.append("## Skill2 认知与思维特征")
        parts.append("```json")
        parts.append(json.dumps(skill2_output, ensure_ascii=False, indent=2))
        parts.append("```")
        parts.append("")

        parts.append("## Skill3 语言风格与情感模式")
        parts.append("```json")
        parts.append(json.dumps(skill3_output, ensure_ascii=False, indent=2))
        parts.append("```")
        parts.append("")

        parts.append("## Skill4 性格特质推断")
        parts.append("```json")
        parts.append(json.dumps(skill4_output, ensure_ascii=False, indent=2))
        parts.append("```")
        parts.append("")

        parts.append("请聚合以上所有分析结果，生成完整人物画像报告（JSON + Markdown）。")

        return "\n".join(parts)

    # ---- 输出解析 ----

    def parse_output(self, llm_response: str) -> Skill5Output:
        cleaned = self._extract_json(llm_response)
        data = json.loads(cleaned)

        # 元数据
        meta = data.get("metadata", {})
        metadata = AnalysisMetadata(
            analysis_date=meta.get("analysis_date", datetime.now().isoformat()),
            version=meta.get("version", "2.0"),
            sources=meta.get("sources", []),
            material_quality=meta.get("material_quality", 0.0),
            overall_confidence=meta.get("overall_confidence", 0.0),
        )

        # 核心画像
        cp = data.get("core_profile", {})
        core_profile = CoreProfile(
            thinking_style=cp.get("thinking_style", ""),
            personality_snapshot=cp.get("personality_snapshot", ""),
            communication_essence=cp.get("communication_essence", ""),
            core_motivation=cp.get("core_motivation", ""),
        )

        # 洞察
        ins = data.get("insights", {})
        insights = Insights(
            strengths=ins.get("strengths", []),
            risks=ins.get("risks", []),
            growth_areas=ins.get("growth_areas", []),
            blind_spots=ins.get("blind_spots", []),
        )

        # 矛盾
        contradictions = [
            Contradiction(
                type=c.get("type", "essential_tension"),
                description=c.get("description", ""),
                evidence=c.get("evidence", []),
            )
            for c in data.get("contradictions", [])
        ]

        # 诚实边界
        hb = data.get("honesty_boundary", {})
        honesty_boundary = HonestyBoundary(
            known=hb.get("known", []),
            uncertain=hb.get("uncertain", []),
            unknown=hb.get("unknown", []),
            material_limitations=hb.get("material_limitations", []),
        )

        return Skill5Output(
            person_id=data.get("person_id", ""),
            metadata=metadata,
            core_profile=core_profile,
            cognitive=data.get("cognitive", {}),
            language_emotion=data.get("language_emotion", {}),
            personality=data.get("personality", {}),
            insights=insights,
            contradictions=contradictions,
            honesty_boundary=honesty_boundary,
            report=data.get("report", ""),
        )

    # ---- 便捷方法 ----

    def run(
        self,
        llm_callable: Callable[[str, str], str],
        person_id: str,
        skill2_output: dict,
        skill3_output: dict,
        skill4_output: dict,
        skill1_output: dict | None = None,
        material_quality: float = 0.5,
    ) -> Skill5Output:
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(
            person_id, skill1_output, skill2_output, skill3_output, skill4_output, material_quality
        )
        response = llm_callable(system_prompt, user_prompt)
        return self.parse_output(response)

    # ---- 本地计算（不需要LLM） ----

    @staticmethod
    def compute_confidence(
        material_quality: float,
        skill2_quality: float,
        skill3_quality: float,
        skill4_quality: float,
        consistency_pass: bool = True,
    ) -> float:
        """本地计算综合置信度"""
        avg_skill_quality = (skill2_quality + skill3_quality + skill4_quality) / 3
        consistency_bonus = 1.0 if consistency_pass else 0.7
        raw = (
            0.25 * material_quality
            + 0.35 * avg_skill_quality
            + 0.25 * consistency_bonus
            + 0.15 * material_quality  # 材料覆盖度
        )
        return round(min(raw, 1.0), 2)

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        return text

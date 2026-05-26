"""
Skill 2: 认知与思维特征提取 (Cognitive Profile)

功能: 信息处理风格 / 决策模式 / 风险态度 / 心智模型识别 / 归因风格 / 时间取向

借鉴 Nuwa-Skill 的三重验证(跨域复现/生成力/排他性)识别稳定的思维模式。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Union

from ..models import (
    AttributionStyle,
    Creativity,
    CreativityType,
    DecisionMode,
    DecisionPattern,
    InfoProcessing,
    InfoProcessingStyle,
    MentalModel,
    RegulatoryFrame,
    RiskAttitude,
    RiskOrientation,
    Skill2Output,
    TimeOrientation,
    TripleCheck,
)

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """## 角色: 认知与思维特征分析师 (Skill 2)

你是认知心理学专家。你的任务是从文本材料中提取一个人的思维模式，包括信息处理风格、决策方式、风险态度，以及识别其核心心智模型。

## 分析维度

### 1. 信息处理风格
- intuitive: 依赖直觉和感受 ("我感觉" "直觉上" "好像")
- analytical: 依赖逻辑和数据 ("数据显示" "分析来看" "逻辑上")
- balanced: 两者兼顾

### 2. 决策模式
- data_driven: 数据驱动 ("先看数据" "跑个AB测试")
- experience_based: 经验驱动 ("以前遇到过" "我的经验是")
- intuition_first: 直觉先行 ("直觉告诉我不对" "凭感觉判断")
- deliberation_level: 1=快速拍板 5=反复权衡 ("我再想想" 频率越高，分越高)

### 3. 风险态度
- conservative: 保守 ("稳妥" "先验证" "风险可控")
- moderate: 平衡 ("试试但留后路")
- aggressive: 激进 ("没问题" "大不了" "亏了再说")
- frame: prevention(预防聚焦, "别出错") vs promotion(促进聚焦, "抓住机会")

### 4. 创造性
- evolutionary: 改良型 ("优化" "迭代" "改进")
- revolutionary: 颠覆型 ("颠覆" "重新定义" "从零开始")

### 5. 归因风格
- internal: 内归因 ("我要负责" "我应该更努力")
- external: 外归因 ("环境所迫" "没办法" "运气不好")
- balanced: 两者平衡

### 6. 时间取向
- past: 回顾过去
- present: 关注当下
- future: 面向未来
- past_with_future: 用过去经验规划未来

### 7. 心智模型识别 (借鉴 Nuwa-Skill 三重验证)

候选思维模式必须通过三重验证才能被认定为"心智模型":

**验证1: 跨场景复现** — 此模式在 ≥2 个不同话题/场景中出现?
  YES → 继续; NO → 降级为"偶发表现"

**验证2: 生成力** — 用此模式能否预测此人在新问题上的反应?
  YES → 继续; NO → 只是特定场景表现

**验证3: 排他性** — 这个模式不是所有人都有的通用特征?
  YES → 确认为心智模型; NO → 通用认知特征

三重全过 → 核心心智模型
通过1-2重 → 辅助认知特征
0重通过 → 丢弃

## 输出格式
严格 JSON (不要markdown代码块):

{
    "cognitive_profile": {
        "info_processing": {
            "style": "balanced_analytical",
            "detail_orientation": 4,
            "evidence": ["原文片段1", "原文片段2"]
        },
        "decision_pattern": {
            "mode": "deliberative",
            "deliberation_level": 4,
            "evidence": ["原文片段..."]
        },
        "risk_attitude": {
            "orientation": "moderate_conservative",
            "frame": "prevention",
            "evidence": ["原文片段..."]
        },
        "creativity": {
            "type": "evolutionary",
            "playfulness": 3,
            "evidence": ["原文片段..."]
        },
        "attribution_style": "internal_balanced",
        "time_orientation": "future_oriented_with_past_reference"
    },
    "mental_models": [
        {
            "name": "渐进验证",
            "description": "不确定时先小步试，验证后再扩大",
            "cross_domain_evidence": ["产品决策中...", "投资决策中..."],
            "generativity": "能预测此人在任何新领域都会选择MVP式推进",
            "exclusivity": "区别于'先做大规划再执行'型的人",
            "triple_check": {
                "cross_domain": true,
                "generative": true,
                "exclusive": true
            }
        }
    ],
    "quality": 0.85,
    "uncertainty": [
        "缺少高压决策场景的材料",
        "..."
    ]
}

## 关键原则
- 每条结论必须有原文 evidence 支撑
- 不要过度推断——只输出有信号支撑的结论
- 心智模型宁缺毋滥(1-3个即可)
- 在 uncertainty 中诚实标注推断不确定的地方
"""


# ============================================================================
# Skill 实现
# ============================================================================


class Skill2CognitiveProfile:
    """认知与思维特征提取 Skill"""

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT

    # ---- Prompt 构建 ----

    def build_system_prompt(self) -> str:
        return self.system_prompt

    def build_user_prompt(
        self,
        cleaned_text: str,
        material_quality: float,
        chunks: list[dict] | None = None,
    ) -> str:
        """构建 User Prompt"""
        parts = [
            "## 清洗后的文本材料",
            f"材料质量评分: {material_quality:.2f}",
            "",
            cleaned_text,
        ]

        if chunks:
            parts.append("\n## 文本分片概览")
            for c in chunks:
                parts.append(f"- 分片{c.get('id', '?')} [{c.get('source', '?')}|{c.get('topic', '?')}]: {c.get('word_count', 0)}字")

        parts.append("\n请基于上述文本，提取此人的认知与思维特征，按 JSON 格式输出。")
        return "\n".join(parts)

    # ---- 输出解析 ----

    def parse_output(self, llm_response: str) -> Skill2Output:
        cleaned = self._extract_json(llm_response)
        data = json.loads(cleaned)
        cp = data.get("cognitive_profile", {})

        # 解析信息处理
        ip_data = cp.get("info_processing", {})
        style_map = {
            "intuitive": InfoProcessingStyle.INTUITIVE,
            "analytical": InfoProcessingStyle.ANALYTICAL,
            "balanced": InfoProcessingStyle.BALANCED,
            "balanced_analytical": InfoProcessingStyle.BALANCED,
        }
        info_processing = InfoProcessing(
            style=style_map.get(ip_data.get("style", "balanced"), InfoProcessingStyle.BALANCED),
            detail_orientation=ip_data.get("detail_orientation", 3),
            evidence=ip_data.get("evidence", []),
        )

        # 解析决策模式
        dp_data = cp.get("decision_pattern", {})
        dm_map = {
            "data_driven": DecisionMode.DATA_DRIVEN,
            "experience_based": DecisionMode.EXPERIENCE_BASED,
            "intuition_first": DecisionMode.INTUITION_FIRST,
            "deliberative": DecisionMode.DATA_DRIVEN,
        }
        decision_pattern = DecisionPattern(
            mode=dm_map.get(dp_data.get("mode", "data_driven"), DecisionMode.DATA_DRIVEN),
            deliberation_level=dp_data.get("deliberation_level", 3),
            evidence=dp_data.get("evidence", []),
        )

        # 解析风险态度
        ra_data = cp.get("risk_attitude", {})
        ro_map = {
            "conservative": RiskOrientation.CONSERVATIVE,
            "moderate": RiskOrientation.MODERATE,
            "moderate_conservative": RiskOrientation.CONSERVATIVE,
            "aggressive": RiskOrientation.AGGRESSIVE,
        }
        risk_attitude = RiskAttitude(
            orientation=ro_map.get(ra_data.get("orientation", "moderate"), RiskOrientation.MODERATE),
            frame=RegulatoryFrame.PREVENTION if ra_data.get("frame") == "prevention" else RegulatoryFrame.PROMOTION,
            evidence=ra_data.get("evidence", []),
        )

        # 解析创造性
        cr_data = cp.get("creativity", {})
        creativity = Creativity(
            type=CreativityType.EVOLUTIONARY if cr_data.get("type") == "evolutionary" else CreativityType.REVOLUTIONARY,
            playfulness=cr_data.get("playfulness", 3),
            evidence=cr_data.get("evidence", []),
        )

        # 解析归因风格
        attr_map = {
            "internal": AttributionStyle.INTERNAL,
            "internal_balanced": AttributionStyle.BALANCED,
            "external": AttributionStyle.EXTERNAL,
            "balanced": AttributionStyle.BALANCED,
        }
        attribution_style = attr_map.get(cp.get("attribution_style", "balanced"), AttributionStyle.BALANCED)

        # 解析时间取向
        to_map = {
            "past": TimeOrientation.PAST,
            "present": TimeOrientation.PRESENT,
            "future": TimeOrientation.FUTURE,
            "future_oriented_with_past_reference": TimeOrientation.PAST_FUTURE,
        }
        time_orientation = to_map.get(cp.get("time_orientation", "present"), TimeOrientation.PRESENT)

        # 解析心智模型
        mental_models = []
        for mm in data.get("mental_models", []):
            tc = mm.get("triple_check", {})
            mental_models.append(
                MentalModel(
                    name=mm.get("name", ""),
                    description=mm.get("description", ""),
                    cross_domain_evidence=mm.get("cross_domain_evidence", []),
                    generativity=mm.get("generativity", ""),
                    exclusivity=mm.get("exclusivity", ""),
                    triple_check=TripleCheck(
                        cross_domain=tc.get("cross_domain", False),
                        generative=tc.get("generative", False),
                        exclusive=tc.get("exclusive", False),
                    ),
                )
            )

        return Skill2Output(
            cognitive_profile=cp,
            mental_models=mental_models,
            attribution_style=attribution_style,
            time_orientation=time_orientation,
            quality=data.get("quality", 0.5),
            uncertainty=data.get("uncertainty", []),
        )

    # ---- 便捷方法 ----

    def run(
        self,
        llm_callable: Callable[[str, str], str],
        cleaned_text: str,
        material_quality: float = 0.5,
        chunks: list[dict] | None = None,
    ) -> Skill2Output:
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(cleaned_text, material_quality, chunks)
        response = llm_callable(system_prompt, user_prompt)
        return self.parse_output(response)

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

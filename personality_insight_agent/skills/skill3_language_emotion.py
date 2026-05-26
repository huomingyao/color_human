"""
Skill 3: 语言风格与情感模式分析 (Language & Emotion)

功能: 句式指纹 / 风格标签 / 情感倾向 / 修辞特征 / 个性化表达模式

借鉴 Nuwa-Skill 的表达DNA量化方法（句式指纹 + 风格标签体系）。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from ..models import (
    Emotion,
    EmotionExpression,
    MetaphorUsage,
    QuestionStyle,
    Rhetoric,
    SentenceFingerprint,
    Skill3Output,
    StyleTags,
)

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """## 角色: 语言风格与情感模式分析师 (Skill 3)

你是语言心理学家。你的任务是从文本中量化分析一个人的语言表达风格和情感模式。

## 分析维度

### 1. 句式指纹 (借鉴 Nuwa-Skill 表达DNA量化)

从文本中采样分析:
- **平均句长**: 总字数 / 总句数。中文通常15-35字/句为正常
- **疑问句比例**: 疑问句数 / 总句数。高比例表示探索型表达
- **类比密度**: 类比/隐喻数 / 千字。高密度表示形象思维倾向
- **第一人称率**: "我/我们" / 总词数。反映自我意识强度
- **确定性语气**: "一定/肯定/显然/绝对" vs "可能/也许/大概/或许" vs 中性。低频的确定性词 → moderate

### 2. 风格标签 (借鉴 Nuwa-Skill 表达DNA标签)

每条1-5分:
- **正式度** (formality): 1=口语化("嘛" "呗" "啦") ↔ 5=正式("鉴于" "综上所述")
- **抽象度** (abstractness): 1=具象("排队" "皱眉" "点击") ↔ 5=抽象("范式" "底层逻辑")
- **审慎度** (cautiousness): 1=断言("就是" "必须") ↔ 5=谨慎("可能" "建议")
- **叙事性** (narrative_data): 1=纯叙事(讲故事) ↔ 5=纯数据(列数字)

### 3. 情感维度 (VAD模型)
- **valence** (愉悦度): -1=极度负面 ~ 1=极度正面。正常对话通常-0.3~0.5
- **arousal** (唤醒度): 0~1。感叹号、程度副词密度越高，唤醒度越高
- **dominance** (控制感): 0~1。"应该/必须" → 高控制; "可能/也许" → 低控制

### 4. 情感表达倾向
- overt: 情感外露 ("太棒了!!!" "气死我了")
- covert: 含蓄表达 ("还行" = 不够好; "有道理" = 不完全同意)
- suppressed: 几乎不表达情感

### 5. 修辞特征
- **隐喻使用**: high/medium/low
- **提问方式**: inquisitive(追问型) / rhetorical(反问型) / minimal(少提问)
- **幽默倾向**: 1-5

### 6. 标志性表达模式
识别此人的个人化表达习惯，例如:
- "每次开头先说'其实...'"
- "在否定前先说'你说的有道理'作为缓冲"
- "频繁使用'我觉得'而非'我认为'"

## 输出格式
严格 JSON:

{
    "language_style": {
        "sentence_fingerprint": {
            "avg_sentence_length": 28.5,
            "question_ratio": 0.12,
            "analogy_density": 3.2,
            "first_person_ratio": 0.08,
            "certainty_tone": "moderate"
        },
        "style_tags": {
            "formality": 3,
            "abstractness": 4,
            "cautiousness": 4,
            "narrative_data": 3
        },
        "emotion": {
            "valence": 0.2,
            "arousal": 0.35,
            "dominance": 0.55,
            "expression_tendency": "covert"
        },
        "rhetoric": {
            "metaphor_usage": "medium",
            "question_style": "inquisitive",
            "humor_tendency": 2
        }
    },
    "signature_patterns": [
        "高频使用'其实'作为转折引导",
        "偏好'我觉得'而非'我认为'"
    ],
    "quality": 0.80,
    "uncertainty": [
        "冲突场景的语言样本不足"
    ]
}

## 分析原则
- 从原文中摘取3-5个典型句子作为示例
- 风格标签要有对比参照（与"普通社交对话"对比）
- 标志性表达模式必须是此人独有的，不能是通用表达习惯
"""


# ============================================================================
# Skill 实现
# ============================================================================


class Skill3LanguageEmotion:
    """语言风格与情感模式分析 Skill"""

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT

    # ---- Prompt 构建 ----

    def build_system_prompt(self) -> str:
        return self.system_prompt

    def build_user_prompt(self, cleaned_text: str, material_quality: float = 0.5) -> str:
        parts = [
            "## 清洗后的文本材料",
            f"材料质量评分: {material_quality:.2f}",
            "",
            cleaned_text,
            "",
            "请基于上述文本，量化分析此人的语言风格和情感模式，按 JSON 格式输出。",
        ]
        return "\n".join(parts)

    # ---- 输出解析 ----

    def parse_output(self, llm_response: str) -> Skill3Output:
        cleaned = self._extract_json(llm_response)
        data = json.loads(cleaned)
        ls = data.get("language_style", {})

        # 句式指纹
        sf = ls.get("sentence_fingerprint", {})
        sentence_fingerprint = SentenceFingerprint(
            avg_sentence_length=sf.get("avg_sentence_length", 0),
            question_ratio=sf.get("question_ratio", 0),
            analogy_density=sf.get("analogy_density", 0),
            first_person_ratio=sf.get("first_person_ratio", 0),
            certainty_tone=sf.get("certainty_tone", "moderate"),
        )

        # 风格标签
        st = ls.get("style_tags", {})
        style_tags = StyleTags(
            formality=st.get("formality", 3),
            abstractness=st.get("abstractness", 3),
            cautiousness=st.get("cautiousness", 3),
            narrative_data=st.get("narrative_data", 3),
        )

        # 情感
        em = ls.get("emotion", {})
        exp_map = {
            "overt": EmotionExpression.OVERT,
            "covert": EmotionExpression.COVERT,
            "suppressed": EmotionExpression.SUPPRESSED,
        }
        emotion = Emotion(
            valence=em.get("valence", 0),
            arousal=em.get("arousal", 0),
            dominance=em.get("dominance", 0),
            expression_tendency=exp_map.get(em.get("expression_tendency", "covert"), EmotionExpression.COVERT),
        )

        # 修辞
        rh = ls.get("rhetoric", {})
        mu_map = {"high": MetaphorUsage.HIGH, "medium": MetaphorUsage.MEDIUM, "low": MetaphorUsage.LOW}
        qs_map = {"inquisitive": QuestionStyle.INQUISITIVE, "rhetorical": QuestionStyle.RHETORICAL, "minimal": QuestionStyle.MINIMAL}
        rhetoric = Rhetoric(
            metaphor_usage=mu_map.get(rh.get("metaphor_usage", "medium"), MetaphorUsage.MEDIUM),
            question_style=qs_map.get(rh.get("question_style", "minimal"), QuestionStyle.MINIMAL),
            humor_tendency=rh.get("humor_tendency", 1),
        )

        return Skill3Output(
            language_style=ls,
            sentence_fingerprint=sentence_fingerprint,
            style_tags=style_tags,
            emotion=emotion,
            rhetoric=rhetoric,
            signature_patterns=data.get("signature_patterns", []),
            quality=data.get("quality", 0.5),
            uncertainty=data.get("uncertainty", []),
        )

    # ---- 便捷方法 ----

    def run(
        self,
        llm_callable: Callable[[str, str], str],
        cleaned_text: str,
        material_quality: float = 0.5,
    ) -> Skill3Output:
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(cleaned_text, material_quality)
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

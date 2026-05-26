"""
Skill 4: 性格特质推断 (Personality Inference)

功能: 大五人格(OCEAN) / MBTI(16P) / 动机理论(McClelland) / PDP行为风格 / 多框架交叉验证

借鉴 Nuwa-Skill 的三重验证，在多个心理学框架间互相校验，不一致时标记而非强行统一。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from ..models import (
    BigFive,
    BigFiveScore,
    ConsistencyCheck,
    MBTI,
    Motivation,
    PDPResult,
    Skill4Output,
)

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """## 角色: 性格特质推断专家 (Skill 4)

你是人格心理学家。你的任务是基于 Skill2(认知特征) 和 Skill3(语言情感) 的输出，用多个心理学框架推断此人的性格特质。

## 推断框架

### 框架1: 大五人格 OCEAN (1-5分)

| 维度 | 高分段信号 (4-5) | 低分段信号 (1-2) |
|------|-----------------|-----------------|
| **O**penness 开放性 | 频繁提问探索、喜欢"换个角度"、对新鲜事物好奇 | 偏好熟悉的方式、"向来如此"、不愿尝试 |
| **C**onscientiousness 尽责性 | 关注细节、有计划、反复确认、"先做计划" | 随性、"到时候再说"、讨厌条条框框 |
| **E**xtraversion 外向性 | 主动发起社交、闲聊频率高、在群体中活跃 | 独处偏好、"不想说话"、社交少 |
| **A**greeableness 宜人性 | 关心他人感受、避免冲突、使用缓冲语("你说的对但...") | 直接挑战、不在意得罪人、直言不讳 |
| **N**euroticism 情绪稳定性 | 低频 → 稳定: 压力下冷静、不流露焦虑 | 高频 → 不稳定: 情绪化表达多、焦虑显露 |

每个维度的评分必须附带:
- signals: 从 Skill2/Skill3 中引用的行为信号
- evidence: 原文证据
- confidence: 该评分的置信度(基于信号充足度)

### 框架2: MBTI 类型推断

从大五映射 + 语言风格校正:

| MBTI维度 | 映射来源 | 规则 |
|----------|---------|------|
| **E/I** | 大五Extraversion + 社交信号 | E≥3.5 → E; 否则 I |
| **S/N** | 大五Openness + Skill3抽象度 | O≥3.5 + 高抽象度 → N; 否则 S |
| **T/F** | 大五Agreeableness + 决策模式 | A≥3.5 + 关注人 → F; 否则 T |
| **J/P** | 大五Conscientiousness + 决策速度 | C≥3.5 + 偏好计划 → J; 否则 P |

输出: 最佳匹配类型 + 备选类型 + 各维度倾向百分比(0-100)

### 框架3: McClelland 动机理论

| 动机 | 信号 |
|------|------|
| **成就** (Achievement) | 追求卓越、"做到最好"、自我挑战 |
| **亲和** (Affiliation) | 重视关系、"大家一起"、回避冲突 |
| **权力** (Power) | 主导讨论、影响他人、"我来决定" |

排序输出(1=最强)，每个动机附带证据。

### 框架4: PDP 行为风格

| 类型 | 特征 | 常见信号 |
|------|------|---------|
| **老虎** | 支配型、目标导向、快节奏 | 主导讨论、追求结果、"快点"、"直接说" |
| **孔雀** | 表达型、社交导向、乐观 | 热情表达、喜欢分享、"太棒了" |
| **考拉** | 稳健型、关系导向、耐心 | 注重和谐、稳定执行、"不急" |
| **猫头鹰** | 精确型、规则导向、谨慎 | 关注细节、"数据呢"、"有依据吗" |
| **变色龙** | 适应型、灵活应变 | 协调多角色、适应性强 |

输出: 主要风格 + 次要风格 + 各类型得分(1-5)。

### 框架5: 一致性检验 (借鉴 Nuwa-Skill 三重验证)

1. **框架内一致**: 大五各维度是否互相兼容? (如高C+高O的组合虽然罕见但是可能的)
2. **框架间一致**: 大五 ↔ MBTI ↔ PDP 的映射是否合理?
3. **信号覆盖**: 每个结论是否有 ≥2 个独立信号支撑?

不一致的地方 → 标注而非抹平。矛盾往往是最有价值的信息。

## 输出格式
严格 JSON:

{
    "personality": {
        "big_five": {
            "openness": {"score": 4, "confidence": 0.8, "signals": ["信号1"], "evidence": ["证据1"]},
            "conscientiousness": {"score": 3, "confidence": 0.75, "signals": [], "evidence": []},
            "extraversion": {"score": 2, "confidence": 0.6, "signals": [], "evidence": []},
            "agreeableness": {"score": 4, "confidence": 0.8, "signals": [], "evidence": []},
            "neuroticism": {"score": 2, "confidence": 0.7, "signals": [], "evidence": []}
        },
        "mbti": {
            "type": "ISTJ",
            "confidence": 0.72,
            "breakdown": {"I": 65, "S": 55, "T": 60, "J": 70},
            "alternative": "INTJ"
        },
        "motivation": {
            "achievement": {"score": 4, "rank": 1, "evidence": []},
            "affiliation": {"score": 3, "rank": 2, "evidence": []},
            "power": {"score": 2, "rank": 3, "evidence": []}
        },
        "pdp": {
            "style": "猫头鹰-考拉",
            "tiger": 2, "peacock": 2, "koala": 4, "owl": 4, "chameleon": 3
        }
    },
    "consistency_check": {
        "intra_framework": "pass",
        "inter_framework": "大五ISTJ与PDP猫头鹰一致，与MBTI ISTJ一致",
        "signal_coverage": "extraversion维度信号不足(置信度仅0.6)",
        "contradictions": [
            "成就动机高但风险态度保守——在ISTJ中这是一致组合，不算矛盾"
        ]
    },
    "quality": 0.75,
    "uncertainty": {
        "extraversion": "社交场景材料不足，评分置信度较低",
        "sensing_intuition": "S/N维度信号偏弱，MBTI此维度可能不准确"
    }
}

## 关键原则
- 不要编造信号——所有评分必须有 Skill2/Skill3 的输出作为依据
- 低置信度时明确标注，不要凑分数
- 不要假设"中庸"——如果信号明显指向极端，就给出极端分数
- 框架间矛盾是宝贵信息，保留并分析原因
"""


# ============================================================================
# Skill 实现
# ============================================================================


class Skill4PersonalityInference:
    """性格特质推断 Skill"""

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT

    # ---- Prompt 构建 ----

    def build_system_prompt(self) -> str:
        return self.system_prompt

    def build_user_prompt(
        self,
        cognitive_output: dict,
        language_output: dict,
        material_quality: float = 0.5,
    ) -> str:
        """构建 User Prompt —— 接收 Skill2 和 Skill3 的输出作为输入"""
        parts = [
            "## Skill2 认知与思维特征",
            "```json",
            json.dumps(cognitive_output, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Skill3 语言风格与情感模式",
            "```json",
            json.dumps(language_output, ensure_ascii=False, indent=2),
            "```",
            "",
            f"材料质量评分: {material_quality:.2f}",
            "",
            "请基于上述两份分析报告，用大五/MBTI/动机/PDP四框架推断性格特质，并进行一致性检验。按 JSON 格式输出。",
        ]
        return "\n".join(parts)

    # ---- 输出解析 ----

    def parse_output(self, llm_response: str) -> Skill4Output:
        cleaned = self._extract_json(llm_response)
        data = json.loads(cleaned)
        p = data.get("personality", {})

        # 大五
        bf = p.get("big_five", {})
        big_five = BigFive(
            openness=self._parse_bf_score(bf.get("openness", {})),
            conscientiousness=self._parse_bf_score(bf.get("conscientiousness", {})),
            extraversion=self._parse_bf_score(bf.get("extraversion", {})),
            agreeableness=self._parse_bf_score(bf.get("agreeableness", {})),
            neuroticism=self._parse_bf_score(bf.get("neuroticism", {})),
        )

        # MBTI
        mbti_data = p.get("mbti", {})
        mbti = MBTI(
            type=mbti_data.get("type", ""),
            confidence=mbti_data.get("confidence", 0.0),
            breakdown=mbti_data.get("breakdown", {}),
            alternative=mbti_data.get("alternative", ""),
        )

        # 动机
        mot_data = p.get("motivation", {})
        motivation = Motivation(
            achievement=mot_data.get("achievement", {"score": 3, "rank": 1}),
            affiliation=mot_data.get("affiliation", {"score": 3, "rank": 2}),
            power=mot_data.get("power", {"score": 3, "rank": 3}),
        )

        # PDP
        pdp_data = p.get("pdp", {})
        pdp = PDPResult(
            style=pdp_data.get("style", ""),
            sub_style=pdp_data.get("sub_style", ""),
            tiger_score=pdp_data.get("tiger", 3),
            peacock_score=pdp_data.get("peacock", 3),
            koala_score=pdp_data.get("koala", 3),
            owl_score=pdp_data.get("owl", 3),
            chameleon_score=pdp_data.get("chameleon", 3),
        )

        # 一致性检查
        cc = data.get("consistency_check", {})
        consistency_check = ConsistencyCheck(
            intra_framework=cc.get("intra_framework", "unknown"),
            inter_framework=cc.get("inter_framework", ""),
            signal_coverage=cc.get("signal_coverage", ""),
            contradictions=cc.get("contradictions", []),
        )

        return Skill4Output(
            personality=p,
            big_five=big_five,
            mbti=mbti,
            motivation=motivation,
            pdp=pdp,
            consistency_check=consistency_check,
            quality=data.get("quality", 0.5),
            uncertainty=data.get("uncertainty", {}),
        )

    # ---- 便捷方法 ----

    def run(
        self,
        llm_callable: Callable[[str, str], str],
        cognitive_output: dict,
        language_output: dict,
        material_quality: float = 0.5,
    ) -> Skill4Output:
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(cognitive_output, language_output, material_quality)
        response = llm_callable(system_prompt, user_prompt)
        return self.parse_output(response)

    # ---- 静态工具方法 ----

    @staticmethod
    def _parse_bf_score(d: dict) -> BigFiveScore:
        return BigFiveScore(
            score=d.get("score", 3),
            confidence=d.get("confidence", 0.5),
            signals=d.get("signals", []),
            evidence=d.get("evidence", []),
        )

    @staticmethod
    def bf_to_mbti_approx(big_five: BigFive) -> tuple[str, dict[str, int]]:
        """本地辅助: 大五 → MBTI 近似映射（不依赖LLM的快速推断）"""
        e_score = big_five.extraversion.score
        o_score = big_five.openness.score
        a_score = big_five.agreeableness.score
        c_score = big_five.conscientiousness.score
        n_score = big_five.neuroticism.score

        ie_val = round(30 + e_score * 14)
        sn_val = round(30 + o_score * 14)
        tf_val = round(70 - a_score * 10)
        jp_val = round(30 + c_score * 14)

        ie = "E" if e_score >= 4 else "I"
        sn = "N" if o_score >= 4 else "S"
        tf = "F" if a_score >= 4 else "T"
        jp = "J" if c_score >= 4 else "P"

        return f"{ie}{sn}{tf}{jp}", {"I" if ie == "I" else "E": ie_val, "S" if sn == "S" else "N": sn_val, "T" if tf == "T" else "F": tf_val, "J" if jp == "J" else "P": jp_val}

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

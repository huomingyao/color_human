"""
蔡岩峻人物画像 - 完整 Skill Pipeline + 双Agent深度辩论

Phase 1: Skill 1-3 并行分析
Phase 2: Skill 4 交叉验证
Phase 3: Skill 5 合成 + Agent A/B 辩论
"""

import json
import os
import requests
from datetime import datetime

DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")


def qwen_call(messages: list, model: str = "qwen-turbo") -> str:
    """调用 Qwen API"""
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {"Authorization": DASHSCOPE_KEY, "Content-Type": "application/json"}
    data = {"model": model, "messages": messages, "temperature": 0.7}
    resp = requests.post(url, headers=headers, json=data, timeout=120)
    result = resp.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    print(f"API Error: {result}")
    return None


def load_corpus() -> list:
    """加载素材"""
    import json
    with open("H:/蔡岩峻相关信息/corpus/index.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# PHASE 1: SKILL 1 - 素材检索与清洗
# ============================================================================

SKILL1_SYSTEM = """你是 Skill1: 素材检索与清洗专家。

## 你的任务
对输入的原始文本进行清洗、分块、质量评估。

## 操作规范
1. 去除无效内容：重复标题、空行、格式标记
2. 按主题/场景分块
3. 评估每个块的：
   - 信息密度（高/中/低）
   - 可信度（基于内容性质）
   - 情绪倾向（正/负/中性）

## 输出格式（JSON）
{
  "cleaned_chunks": [
    {"id": 1, "source": "...", "topic": "...", "text": "...", "word_count": N, "density": "high", "credibility": "high"}
  ],
  "quality_score": 0.0-1.0,
  "limitations": ["..."]
}"""


SKILL1_USER = """## 原始素材

{}

请进行清洗和分块评估。只输出 JSON。"""


# ============================================================================
# PHASE 2: SKILL 2 - 认知与思维特征
# ============================================================================

SKILL2_SYSTEM = """你是 Skill2: 认知与思维特征提取专家。

## 你的任务
从清洗后的素材中提取认知模式和思维特征。

## 分析维度
1. 信息处理风格：直觉型 vs 分析型
2. 决策模式：快思考 vs 慢思考
3. 风险态度：保守/平衡/激进
4. 归因风格：内归因 vs 外归因
5. 时间取向：过去/现在/未来
6. 心智模型：识别 1-3 个核心思维框架

## 重要：三验证法
- 跨场景复现：此模式在 ≥2 个不同场景出现？
- 生成力：能否预测此人新问题上的反应？
- 排他性：这是此人人特有的，还是普遍特征？

## 输出格式（JSON）
{
  "cognitive_profile": {
    "info_processing": {"style": "...", "evidence": ["..."]},
    "decision_mode": "...",
    "risk_attitude": "...",
    "attribution_style": "内归因/外归因",
    "mental_models": [
      {"name": "...", "description": "...", "triple_check": {"cross_domain": bool, "generative": bool, "exclusive": bool}}
    ]
  },
  "quality": 0.0-1.0,
  "uncertainty": [...]
}"""


SKILL2_USER = """## 清洗后的素材

{}

请进行认知思维分析。只输出 JSON。"""


# ============================================================================
# PHASE 3: SKILL 3 - 语言风格与情感模式
# ============================================================================

SKILL3_SYSTEM = """你是 Skill3: 语言风格与情感模式分析专家。

## 你的任务
分析语言表达特征和情感模式。

## 分析维度
### 句式指纹（Nuwa方法）
- 平均句长
- 疑问句比例
- 类比密度
- 第一人称率
- 确定性语气

### 风格标签
- 正式↔口语 (1-5)
- 抽象↔具象 (1-5)
- 谨慎↔断言 (1-5)
- 叙事↔数据 (1-5)

### 情感分析
- Valence: 正向/负向程度
- Arousal: 情感强度
- Dominance: 控制感

### 修辞特征
- 隐喻类型
- 幽默风格

## 输出格式（JSON）
{
  "language_style": {
    "sentence_fingerprint": {"avg_length": N, "question_ratio": N, "certainty": "..."},
    "style_tags": {"formality": 1-5, "abstractness": 1-5, "cautiousness": 1-5, "narrative_data": 1-5},
    "emotion": {"valence": "-1~1", "arousal": "-1~1", "dominance": "-1~1"},
    "signature_patterns": ["..."]
  },
  "quality": 0.0-1.0,
  "uncertainty": [...]
}"""


SKILL3_USER = """## 文本素材

{}

请进行语言风格分析。只输出 JSON。"""


# ============================================================================
# PHASE 4: SKILL 4 - 性格特质推断
# ============================================================================

SKILL4_SYSTEM = """你是 Skill4: 性格特质推断专家。

## 你的任务
基于 Skill2(认知) + Skill3(语言) 的输出，推断完整性格。

## 推断框架
### 大五人格 (1-5)
- Openness (开放性)
- Conscientiousness (尽责性)
- Extraversion (外向性)
- Agreeableness (宜人性)
- Neuroticism (神经质)

### MBTI (从大五映射)
- E/I, S/N, T/F, J/P

### 动机模式 (McClelland)
- Achievement (成就)
- Affiliation (亲和)
- Power (权力)

### PDP 行为风格
- 老虎/孔雀/考拉/猫头鹰/变色龙

## 交叉验证原则
1. 框架内一致：大五各维度不矛盾？
2. 框架间一致：大五↔MBTI↔PDP 映射合理？
3. 信号覆盖：每结论有 ≥2 个信号支撑？

## 输出格式（JSON）
{
  "personality": {
    "big_five": {"openness": N, "conscientiousness": N, "extraversion": N, "agreeableness": N, "neuroticism": N},
    "mbti": {"type": "XXXX", "confidence": 0.0-1.0, "breakdown": {"E": N, "I": N, ...}},
    "motivation": {"achievement": N, "affiliation": N, "power": N},
    "pdp": {"style": "...", "scores": {...}}
  },
  "consistency_check": {"intra_framework": "pass/fail", "inter_framework": "...", "contradictions": [...]},
  "quality": 0.0-1.0,
  "uncertainty": {...}
}"""


SKILL4_USER = """## Skill 2 输出（认知）

{}

## Skill 3 输出（语言）

{}

请进行性格推断。只输出 JSON。"""


# ============================================================================
# PHASE 5: SKILL 5 - 画像合成 + 双Agent辩论
# ============================================================================

SKILL5_SYSTEM = """你是 Skill5: 画像合成与质量验证专家。

## 你的任务
1. 聚合 Skill1-4 的结果
2. 生成完整画像
3. 引入双Agent进行深度辩论验证

## 画像结构
- thinking_style: 一句话思维特征
- personality_snapshot: 一句话性格
- communication_essence: 一句话沟通
- core_motivation: 一句话驱动力

## 双Agent辩论
- Agent A（攻击方）：质疑结论，找出矛盾和漏洞
- Agent B（防守方）：回应质疑，辩护或修正

## 最后仲裁
综合双方论点，给出最终结论和置信度。

## 输出格式（JSON）
{
  "metadata": {"sources": N, "total_chars": N, "pipeline_version": "2.0"},
  "core_profile": {
    "thinking_style": "...",
    "personality_snapshot": "...",
    "communication_essence": "...",
    "core_motivation": "..."
  },
  "cognitive": {...},  // Skill2输出
  "language_emotion": {...},  // Skill3输出
  "personality": {...},  // Skill4输出
  "debate": {
    "agent_a_attacks": [...],
    "agent_b_defends": [...],
    "final_verdict": "..."
  },
  "confidence": 0.0-1.0,
  "contradictions": [...],
  "honesty_boundary": {"known": [...], "uncertain": [...], "unknown": [...]},
  "report": "Markdown报告"
}"""


SKILL5_USER = """## Skill 1 输出

{}

## Skill 2 输出

{}

## Skill 3 输出

{}

## Skill 4 输出

{}

请进行画像合成，并执行双Agent辩论验证。只输出 JSON。"""


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("蔡岩峻人物画像 - 完整 Skill Pipeline + 双Agent辩论")
    print("="*70)

    # 加载素材
    corpus = load_corpus()
    print(f"\n📥 加载 {len(corpus)} 个文件")

    # 合并为连续文本
    all_text = ""
    for item in corpus:
        all_text += f"\n=== {item['source']} ===\n{item['content']}\n"

    print(f"📏 总字符: {len(all_text)}")

    # ===== PHASE 1: SKILL 1 =====
    print("\n" + "="*70)
    print("PHASE 1: Skill 1 - 素材检索与清洗")
    print("="*70)

    msg = [
        {"role": "system", "content": SKILL1_SYSTEM},
        {"role": "user", "content": SKILL1_USER.format(all_text[:8000])}
    ]
    s1_result = qwen_call(msg)
    try:
        s1_data = json.loads(s1_result)
        print(f"✅ 质量分数: {s1_data.get('quality_score', 'N/A')}")
        print(f"📦 分块数: {len(s1_data.get('cleaned_chunks', []))}")
    except:
        s1_data = {"cleaned_chunks": [{"text": all_text[:5000]}], "quality_score": 0.6}
        print("⚠️ 使用简化结果")

    # ===== PHASE 2: SKILL 2 =====
    print("\n" + "="*70)
    print("PHASE 2: Skill 2 - 认知与思维特征")
    print("="*70)

    msg = [
        {"role": "system", "content": SKILL2_SYSTEM},
        {"role": "user", "content": SKILL2_USER.format(all_text[:8000])}
    ]
    s2_result = qwen_call(msg)
    try:
        s2_data = json.loads(s2_result)
        print(f"✅ 认知分析完成")
        if "cognitive_profile" in s2_data:
            cp = s2_data["cognitive_profile"]
            print(f"  决策模式: {cp.get('decision_mode', 'N/A')}")
            print(f"  风险态度: {cp.get('risk_attitude', 'N/A')}")
            models = cp.get("mental_models", [])
            print(f"  心智模型: {len(models)} 个")
    except:
        s2_data = {}
        print("⚠️ 解析失败")

    # ===== PHASE 3: SKILL 3 =====
    print("\n" + "="*70)
    print("PHASE 3: Skill 3 - 语言风格与情感")
    print("="*70)

    key_text = all_text  # 使用完整文本
    msg = [
        {"role": "system", "content": SKILL3_SYSTEM},
        {"role": "user", "content": SKILL3_USER.format(key_text[:8000])}
    ]
    s3_result = qwen_call(msg)
    try:
        s3_data = json.loads(s3_result)
        print(f"✅ 语言分析完成")
        if "language_style" in s3_data:
            ls = s3_data["language_style"]
            st = ls.get("style_tags", {})
            print(f"  正式度: {st.get('formality', 'N/A')}/5")
            print(f"  谨慎度: {st.get('cautiousness', 'N/A')}/5")
    except:
        s3_data = {}
        print("⚠️ 解析失败")

    # ===== PHASE 4: SKILL 4 =====
    print("\n" + "="*70)
    print("PHASE 4: Skill 4 - 性格特质推断 + 交叉验证")
    print("="*70)

    s2_str = json.dumps(s2_data, ensure_ascii=False)[:3000]
    s3_str = json.dumps(s3_data, ensure_ascii=False)[:3000]

    msg = [
        {"role": "system", "content": SKILL4_SYSTEM},
        {"role": "user", "content": SKILL4_USER.format(s2_str, s3_str)}
    ]
    s4_result = qwen_call(msg)
    try:
        s4_data = json.loads(s4_result)
        print(f"✅ 性格推断完成")
        if "personality" in s4_data:
            p = s4_data["personality"]
            bf = p.get("big_five", {})
            print(f"  大五: O:{bf.get('openness','?')} C:{bf.get('conscientiousness','?')} "
                  f"E:{bf.get('extraversion','?')} A:{bf.get('agreeableness','?')} N:{bf.get('neuroticism','?')}")
            print(f"  MBTI: {p.get('mbti', {}).get('type', 'N/A')}")
    except:
        s4_data = {}
        print("⚠️ 解析失败")

    # ===== PHASE 5: SKILL 5 + 双Agent辩论 =====
    print("\n" + "="*70)
    print("PHASE 5: Skill 5 - 画像合成 + 双Agent辩论")
    print("="*70)

    msg = [
        {"role": "system", "content": SKILL5_SYSTEM},
        {"role": "user", "content": SKILL5_USER.format(
            json.dumps(s1_data, ensure_ascii=False)[:2000],
            json.dumps(s2_data, ensure_ascii=False)[:2000],
            json.dumps(s3_data, ensure_ascii=False)[:2000],
            json.dumps(s4_data, ensure_ascii=False)[:2000]
        )}
    ]
    s5_result = qwen_call(msg)
    try:
        final_data = json.loads(s5_result)
        print(f"✅ 画像生成完成 + 辩论完成")

        # 打印核心结果
        print("\n" + "="*70)
        print("最终画像")
        print("="*70)

        if "core_profile" in final_data:
            cp = final_data["core_profile"]
            print(f"\n🧠 思维特征: {cp.get('thinking_style', 'N/A')}")
            print(f"👤 性格快照: {cp.get('personality_snapshot', 'N/A')}")
            print(f"💬 沟通风格: {cp.get('communication_essence', 'N/A')}")
            print(f"🎯 核心驱动力: {cp.get('core_motivation', 'N/A')}")

        if "personality" in final_data:
            p = final_data["personality"]
            bf = p.get("big_five", {})
            print(f"\n📊 大五: O:{bf.get('openness','?')} C:{bf.get('conscientiousness','?')} "
                  f"E:{bf.get('extraversion','?')} A:{bf.get('agreeableness','?')} N:{bf.get('neuroticism','?')}")
            mb = p.get("mbti", {})
            print(f"🔤 MBTI: {mb.get('type','N/A')} (置信度: {mb.get('confidence','?')})")
            mot = p.get("motivation", {})
            print(f"🏆 动机: {mot}")

        print(f"\n📈 置信度: {final_data.get('confidence', 'N/A')}")

        # 打印辩论要点
        if "debate" in final_data:
            db = final_data["debate"]
            print(f"\n⚔️  双Agent辩论:")
            for a in db.get("agent_a_attacks", [])[:2]:
                print(f"  攻击: {a}")
            for b in db.get("agent_b_defends", [])[:2]:
                print(f"  防守: {b}")
            print(f"  裁决: {db.get('final_verdict', 'N/A')}")

        # 保存完整结果
        output_file = "H:/蔡岩峻相关信息/corpus/蔡岩峻_完整画像.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 完整结果已保存: {output_file}")

    except json.JSONDecodeError as e:
        print(f"❌ 最终解析失败: {e}")
        print(s5_result[:500] if s5_result else "无输出")


if __name__ == "__main__":
    main()
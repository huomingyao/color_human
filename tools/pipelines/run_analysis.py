"""
直接运行蔡岩峻人物画像分析
使用 Qwen API
"""
import json
import os
import requests
from datetime import datetime


# 从环境变量获取 API Key
DASHSCOPE_KEY = os.environ.get("DASHSCOPE_API_KEY", "")


def qwen_call(messages: list, model: str = "qwen-turbo") -> str:
    """调用 Qwen API"""
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": DASHSCOPE_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7
    }
    resp = requests.post(url, headers=headers, json=data, timeout=120)
    result = resp.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    else:
        print(f"API Error: {result}")
        return None


def load_text(path: str) -> str:
    """加载文本"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_result(result: dict, path: str):
    """保存结果"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ============================================================================
# 简化版 Pipeline (直接调用 LLM，一步输出完整画像)
# ============================================================================

SYSTEM_PROMPT = """你是 Personality Insight Agent，基于心理学框架的人物画像分析专家。

## 你的任务
分析输入的文本材料，生成结构化的人物画像报告。

## 分析框架
1. 认知思维特征：信息处理风格、决策模式、风险态度、心智模型
2. 语言风格：句式特点、表达习惯、修辞偏好
3. 性格特质：大五人格 + MBTI + 动机模式
4. 核心画像：一句话总结

## 输出格式
请直接输出 JSON 格式：
{
    "analysis_date": "ISO日期",
    "person_id": "蔡岩峻",
    "source_material": "知守学堂申请信",
    "core_profile": {
        "thinking_style": "一句话总结思维特征",
        "personality_snapshot": "一句话总结性格",
        "communication_style": "沟通风格",
        "core_motivation": "核心驱动力"
    },
    "cognitive": {
        "info_processing": {"style": "...", "evidence": ["..."]},
        "decision_mode": "...",
        "risk_attitude": "...",
        "mental_models": [{"name": "...", "description": "..."}]
    },
    "language_style": {
        "formality": 1-5,
        "characteristics": ["..."]
    },
    "personality": {
        "big_five": {"openness": N, "conscientiousness": N, "extraversion": N, "agreeableness": N, "neuroticism": N},
        "mbti": "XXXX",
        "motivation": "成就/亲和/权力"
    },
    "strengths": ["..."],
    "weaknesses": ["..."],
    "confidence": 0.N,
    "honesty_boundary": {
        "known": ["..."],
        "uncertain": ["..."]
    }
}

注意：
- 只输出 JSON，不要 markdown 包裹
- evidence 字段必须有原文支撑
- confidence 根据材料丰富度评估"""


USER_PROMPT_TEMPLATE = """## 素材：蔡岩峻 知守学堂申请信

{}

## 请根据以上素材进行分析"""




def main():
    # 读取蔡岩峻的申请信
    text = load_text("H:/蔡岩峻相关信息/蔡岩峻_申请信.txt")
    print(f"📄 加载文本: {len(text)} 字符")
    print("\n" + "="*60)
    print("蔡岩峻人物画像分析 (使用 Qwen API)")
    print("="*60)

    # 调用 LLM 进行分析
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text)}
    ]

    print("\n🤖 正在调用 Qwen API 分析...")
    result_json = qwen_call(messages)

    if result_json:
        print("\n✅ 分析完成!")
        try:
            result = json.loads(result_json)

            # 打印核心结果
            print("\n" + "="*60)
            print("核心画像")
            print("="*60)
            if "core_profile" in result:
                cp = result["core_profile"]
                print(f"\n🧠 思维特征: {cp.get('thinking_style', 'N/A')}")
                print(f"👤 性格快照: {cp.get('personality_snapshot', 'N/A')}")
                print(f"💬 沟通风格: {cp.get('communication_style', 'N/A')}")
                print(f"🎯 核心驱动力: {cp.get('core_motivation', 'N/A')}")

            if "personality" in result:
                p = result["personality"]
                print(f"\n📊 大五人格: ", end="")
                bf = p.get("big_five", {})
                print(f"O:{bf.get('openness', '?')} C:{bf.get('conscientiousness', '?')} "
                      f"E:{bf.get('extraversion', '?')} A:{bf.get('agreeableness', '?')} N:{bf.get('neuroticism', '?')}")
                print(f"🔤 MBTI: {p.get('mbti', 'N/A')}")
                print(f"🏆 动机: {p.get('motivation', 'N/A')}")

            print(f"\n📈 置信度: {result.get('confidence', 'N/A')}")

            if "strengths" in result:
                print(f"\n✨ 优势: {', '.join(result['strengths'][:3])}")
            if "weaknesses" in result:
                print(f"⚠️  待提升: {', '.join(result['weaknesses'][:3])}")

            # 保存完整结果
            output_file = "H:/蔡岩峻相关信息/蔡岩峻_画像.json"
            save_result(result, output_file)
            print(f"\n💾 完整结果已保存到: {output_file}")

        except json.JSONDecodeError as e:
            print(f"\n❌ JSON 解析失败: {e}")
            print(f"原始输出:\n{result_json[:500]}...")
    else:
        print("\n❌ API 调用失败")


if __name__ == "__main__":
    main()
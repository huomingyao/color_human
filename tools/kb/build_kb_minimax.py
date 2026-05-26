"""
使用 MiniMax API 构建蔡岩峻知识库
"""
import os
import sys
import json
import time
import asyncio
from pathlib import Path

# MiniMax API
API_KEY = os.environ.get("MINIMAX_API", "")
BASE_URL = "https://api.minimaxi.com/v1/text/chatcompletion_v2"


async def call_minimax(system_prompt: str, user_prompt: str) -> str:
    """调用 MiniMax API"""
    import aiohttp

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "MiniMax-M2.7",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(BASE_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"API error: {resp.status}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


# 改成同步版本便于控制
def call_minimax_sync(system_prompt: str, user_prompt: str) -> str:
    """同步调用 MiniMax API"""
    import requests

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "MiniMax-M2.7",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
    }

    resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"API error: {resp.status_code}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# 摘要提示词
SUMMARY_SYSTEM_PROMPT = """你是一个知识库分析专家。你的任务是为给定文档生成结构化摘要。

请以 JSON 格式输出:
{
    "title": "文档标题",
    "summary": "200字以内的内容摘要",
    "key_topics": ["主题1", "主题2"],
    "key_entities": ["人名1", "人名2"],
    "evidence_level": "primary/secondary/tertiary"
}

规则:
- evidence_level: 本人写→primary, 他人写的→secondary, 推断→tertiary
- key_topics: 不超过3个核心主题
- 只输出JSON，不要其他文本"""


def main():
    from personality_insight_agent.corpus2skill.builder import CorpusTreeBuilder

    builder = CorpusTreeBuilder()

    # 1. 扫描文件
    print("步骤1: 扫描文件...")
    files = builder._scan_directory("H:/蔡岩峻相关信息/")
    print(f"  扫描到 {len(files)} 个文件")

    # 2. 分类
    print("\n步骤2: 分类文件...")
    classified = builder._classify_files(files)

    from personality_insight_agent.corpus2skill.models import ContentCategory

    for cat in ContentCategory:
        flist = classified.get(cat, [])
        if flist:
            print(f"  {cat.value}: {len(flist)} 个")

    # 3. 读取文本并生成摘要
    print("\n步骤3: 读取文本 + 生成摘要...")
    processed = []

    for cat, flist in classified.items():
        print(f"\n处理分类: {cat.value} ({len(flist)} 文���)")
        for i, f in enumerate(flist):
            text = builder._read_file_content(f["full_path"])
            if not text:
                continue

            # 取前3000字作为摘要输入
            sample = text[:3000]

            user_prompt = f"""文件名: {f['filename']}
内容样本:
---
{sample}
---

请分析并返回JSON。"""

            try:
                resp = call_minimax_sync(SUMMARY_SYSTEM_PROMPT, user_prompt)
                parsed = builder._parse_json_response(resp)
                summary = {
                    "title": parsed.get("title", f["filename"]),
                    "summary": parsed.get("summary", "")[:300],
                    "key_topics": parsed.get("key_topics", [])[:3],
                    "key_entities": parsed.get("key_entities", [])[:5],
                    "category": cat.value,
                }
            except Exception as e:
                summary = {
                    "title": f["filename"],
                    "summary": sample[:200],
                    "key_topics": [],
                    "key_entities": [],
                    "category": cat.value,
                }

            processed.append({
                **f,
                "text": text,
                "word_count": builder._count_words(text),
                "chunks": builder._split_text(text, 500, 50),
                "summary": summary,
            })

            if (i + 1) % 50 == 0:
                print(f"  已处理 {i+1}/{len(flist)}")

            # 避免过快调用
            time.sleep(0.3)

    print(f"\n共处理 {len(processed)} 个文件")

    # 4. 保存原始文本（更大的json文件）
    print("\n步骤4: 保存知识库...")
    output = {
        "person_id": "caiyanjun",
        "total_documents": len(processed),
        "total_words": sum(f["word_count"] for f in processed),
        "build_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "documents": [
            {
                "category": f["summary"]["category"],
                "path": f["rel_path"],
                "title": f["summary"]["title"],
                "summary": f["summary"]["summary"],
                "key_topics": f["summary"]["key_topics"],
                "key_entities": f["summary"]["key_entities"],
                "evidence_level": "primary",
                "word_count": f["word_count"],
                "text_preview": f["text"][:500],  # 保留前500字预览
            }
            for f in processed
        ],
    }

    output_dir = "d:/person_fenxi/personality_insight_agent/knowledge_base/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "caiyanjun_corpus_tree.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 知识库已保存: {output_path}")
    print(f"   文档数: {output['total_documents']}")
    print(f"   总字数: {output['total_words']}")


if __name__ == "__main__":
    main()
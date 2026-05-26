"""
Personality Insight Agent V2 — 主入口

使用示例:
    python -m personality_insight_agent.main

或作为库使用:
    from personality_insight_agent import analyze_personality
    from personality_insight_agent import analyze_with_private_corpus
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from .models import InputType, OrchestratorInput, OrchestratorOutput, Skill5Output
from .orchestrator import PersonalityInsightOrchestrator
from .orchestrator_v2 import PersonalityInsightOrchestratorV2
from .skills.skill1_material_retrieval import Skill1MaterialRetrieval
from .tools.quality_checker import run_all_checks
from .agentic_rag.models import ResearchDepth

# ============================================================================
# 便捷函数: 一行调用
# ============================================================================


def analyze_personality(
    llm_call: Callable[[str, str], str],
    text: str,
    person_id: Optional[str] = None,
) -> dict:
    """
    一行式调用 —— 输入文本，输出完整人物画像。

    Args:
        llm_call: LLM 调用函数 (system_prompt, user_prompt) -> response_text
        text: 聊天记录、文章等文本
        person_id: 可选的人物标识

    Returns:
        dict: 包含完整画像的字典（可直接序列化为 JSON）

    Example:
        >>> def my_llm(system, user):
        ...     import openai
        ...     response = openai.chat.completions.create(
        ...         model="gpt-4o",
        ...         messages=[{"role":"system","content":system}, {"role":"user","content":user}]
        ...     )
        ...     return response.choices[0].message.content
        ...
        >>> result = analyze_personality(my_llm, chat_history, "zhangsan")
        >>> print(result["core_profile"]["thinking_style"])
        >>> print(result["report"])
    """
    orchestrator = PersonalityInsightOrchestrator()
    output = orchestrator.analyze(
        llm_callable=llm_call,
        raw_text=text,
        person_id=person_id,
    )

    if output.final_output is None:
        return {
            "error": True,
            "degradation": True,
            "reason": output.degradation_reason,
        }

    return output.final_output.model_dump()


def quick_analyze(
    llm_call: Callable[[str, str], str],
    text: str,
) -> dict:
    """快速分析（跳过 Skill1 预处理，直接进行认知和语言分析）"""
    return analyze_personality(llm_call, text)


def analyze_with_private_corpus(
    llm_call: Callable[[str, str], str],
    corpus_dir: str,
    person_id: str,
    vector_search_fn: Optional[Callable[[str, str, int], list[dict[str, Any]]]] = None,
    raw_text: Optional[str] = None,
    research_depth: str = "standard",
) -> dict[str, Any]:
    """
    完整模式 — 含私有知识库的人物画像分析。

    Args:
        llm_call: LLM 调用函数 (system_prompt, user_prompt) -> response_text
        corpus_dir: 私有知识库目录路径 (包含文档、笔记等)
        person_id: 人物标识
        vector_search_fn: 向量检索函数 (query, category, top_k) -> list[dict]
                          如果不提供，Agentic RAG 将退化为 LLM 知识检索
        raw_text: 额外的原始文本 (可选，与知识库内容合并)
        research_depth: 研究深度 "quick"/"standard"/"deep"

    Returns:
        dict: 包含完整画像 + 研究过程 + 验证报告

    Example:
        >>> def my_llm(sys, usr):
        ...     # your LLM implementation
        ...     pass
        ...
        >>> def my_faiss_search(query, category, top_k):
        ...     # your FAISS vector search
        ...     return [{"content": "...", "source": "...", "chunk_id": "..."}]
        ...
        >>> result = analyze_with_private_corpus(
        ...     llm_call=my_llm,
        ...     corpus_dir="d:/my_private_docs/zhangsan/",
        ...     person_id="zhangsan",
        ...     vector_search_fn=my_faiss_search,
        ... )
        >>> print(result["extended_report"])
        >>> print(result["arbitration"]["arbitration_summary"])
    """
    depth_map = {
        "quick": ResearchDepth.QUICK,
        "standard": ResearchDepth.STANDARD,
        "deep": ResearchDepth.DEEP,
    }
    depth = depth_map.get(research_depth, ResearchDepth.STANDARD)

    orch_v2 = PersonalityInsightOrchestratorV2()
    result = orch_v2.analyze_with_corpus(
        llm_callable=llm_call,
        raw_text=raw_text,
        person_id=person_id,
        corpus_dir=corpus_dir,
        vector_search_fn=vector_search_fn,
        research_depth=depth,
    )

    return result


def compare_personalities(
    llm_call: Callable[[str, str], str],
    texts: list[dict],
) -> list[dict]:
    """
    多人对比分析

    Args:
        llm_call: LLM 调用函数
        texts: [{"person_id": "A", "text": "..."}, {"person_id": "B", "text": "..."}, ...]

    Returns:
        list[dict]: 每个人的画像结果
    """
    results = []
    for item in texts:
        result = analyze_personality(
            llm_call,
            item["text"],
            item.get("person_id"),
        )
        results.append(result)
    return results


# ============================================================================
# CLI 接口
# ============================================================================


def create_mock_llm(dry_run: bool = True) -> Callable[[str, str], str]:
    """
    创建模拟 LLM（用于测试Pipeline结构）
    当 dry_run=True 时，返回空JSON而非真实调用LLM
    """
    if dry_run:

        def mock_llm(system_prompt: str, user_prompt: str) -> str:
            # 根据 prompt 内容返回一个基本结构
            if "Skill1" in system_prompt or "素材检索" in system_prompt:
                return json.dumps({
                    "material": {"cleaned_text": user_prompt, "chunks": []},
                    "quality_score": 0.6,
                    "limitations": ["dry_run模式"],
                }, ensure_ascii=False)
            elif "Skill2" in system_prompt or "认知与思维" in system_prompt:
                return json.dumps({
                    "cognitive_profile": {},
                    "mental_models": [],
                    "quality": 0.6,
                    "uncertainty": ["dry_run模式"],
                }, ensure_ascii=False)
            elif "Skill3" in system_prompt or "语言风格" in system_prompt:
                return json.dumps({
                    "language_style": {},
                    "signature_patterns": [],
                    "quality": 0.6,
                    "uncertainty": ["dry_run模式"],
                }, ensure_ascii=False)
            elif "Skill4" in system_prompt or "性格特质推断" in system_prompt:
                return json.dumps({
                    "personality": {},
                    "consistency_check": {"intra_framework": "pass"},
                    "quality": 0.6,
                    "uncertainty": {},
                }, ensure_ascii=False)
            elif "Skill5" in system_prompt or "画像合成" in system_prompt:
                return json.dumps({
                    "person_id": "test",
                    "metadata": {"overall_confidence": 0.6},
                    "core_profile": {"thinking_style": "dry_run"},
                    "insights": {"strengths": [], "risks": [], "growth_areas": [], "blind_spots": []},
                    "contradictions": [],
                    "honesty_boundary": {"known": [], "uncertain": [], "unknown": [], "material_limitations": []},
                    "report": "# Dry Run Report",
                }, ensure_ascii=False)
            return "{}"

        return mock_llm

    # 真实调用需要用户提供 API key
    def real_llm(system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            "请提供你自己的 LLM 调用函数。示例:\n"
            "  import openai\n"
            "  def my_llm(sys, usr):\n"
            "      r = openai.chat.completions.create(model='gpt-4o', messages=[...])\n"
            "      return r.choices[0].message.content\n"
        )

    return real_llm


def main():
    """CLI 入口"""
    print("=" * 60)
    print("Personality Insight Agent V2.0")
    print("集成 Corpus2Skill + Agentic RAG + 双引擎验证")
    print("=" * 60)

    # 解析参数
    corpus_dir = None
    text_file = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--corpus" and i + 1 < len(args):
            corpus_dir = args[i + 1]
            i += 2
        elif not text_file:
            text_file = args[i]
            i += 1
        else:
            i += 1

    if text_file:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
    elif corpus_dir:
        text = None
    else:
        print("\n用法:")
        print("  python -m personality_insight_agent.main <text_file>")
        print("  python -m personality_insight_agent.main --corpus <corpus_dir>")
        print("\n作为库使用:")
        print("  from personality_insight_agent import analyze_personality")
        print("  result = analyze_personality(my_llm_func, text, 'name')")
        print("\n  from personality_insight_agent import analyze_with_private_corpus")
        print("  result = analyze_with_private_corpus(llm, 'dir/', 'name', search_fn)")
        print("\n运行 dry_run 测试...")
        text = "这是一段测试文本，用于验证 Pipeline 结构。"

    # Dry run 测试
    if text:
        print(f"\n[Phase 0] 输入类型判断...")
        assessment = Skill1MaterialRetrieval.quick_assess(text)
        print(f"  有效字数: {assessment['effective_words']}")
        print(f"  预估质量: {assessment['approx_quality']:.2f}")

        if assessment["degradation_needed"]:
            print("  ⚠ 材料不足，需要降级策略")
            return

    # V2 Orchestrator
    mock_llm = create_mock_llm(dry_run=True)
    orch_v2 = PersonalityInsightOrchestratorV2()

    if corpus_dir:
        print(f"\n[V2] 含知识库分析: {corpus_dir}")
        print("(dry_run 模式下不实际构建知识树)")

    print(f"\n[Dry Run] 测试 V2 Orchestrator Pipeline...")

    if corpus_dir and os.path.isdir(corpus_dir):
        # 在 dry_run 模式下，知识树构建会被跳过，因为 LLM 返回的是 mock 数据
        print("(知识库模式: 需要真实 LLM 来构建知识树)")
        result_v2 = orch_v2.analyze_with_corpus(
            llm_callable=mock_llm,
            raw_text=text,
            person_id="test_user",
            corpus_dir=corpus_dir,
            vector_search_fn=None,
            research_depth=ResearchDepth.QUICK,
        )
        print(f"\n  Pipeline: {' → '.join(result_v2['orchestrator_output'].pipeline_used)}")
        for step in result_v2['orchestrator_output'].pipeline_steps:
            icon = "✓" if step.status == "done" else "⚠"
            print(f"  {icon} {step.step_name}: quality={step.quality:.2f}")

        if result_v2.get("corpus_tree"):
            print(f"  知识树: {result_v2['corpus_tree'].total_documents} 文档")
        print(f"  Agentic RAG Agent数: {len(result_v2['agentic_research'])}")
        if result_v2.get("verification"):
            print(f"  事实核查: {result_v2['verification'].total_claims} 条声明")
        if result_v2.get("arbitration"):
            print(f"  仲裁: {result_v2['arbitration'].total_conflicts} 条冲突")
    else:
        result = orch_v2.analyze(
            llm_callable=mock_llm,
            raw_text=text or "测试文本",
            person_id="test_user",
        )

        print(f"  Pipeline: {' → '.join(result.pipeline_used)}")
        for step in result.pipeline_steps:
            icon = "✓" if step.status == "done" else "⚠"
            print(f"  {icon} {step.step_name}: quality={step.quality:.2f}")

        if result.final_output:
            print(f"\n  最终置信度: {result.final_output.metadata.overall_confidence:.2%}")

            quality_report = run_all_checks(
                mental_models=[],
                skill_outputs=[],
                honesty=result.final_output.honesty_boundary.model_dump(),
                contradictions=[c.model_dump() for c in result.final_output.contradictions],
                metadata=result.final_output.metadata.model_dump(),
            )
            print(f"  质量检查: {quality_report.passed_count}/{quality_report.total} 通过")

    print("\nV2 Pipeline 结构测试完成。")
    print("要真实使用，请提供 LLM 调用函数和可选的 FAISS 向量检索函数。")


if __name__ == "__main__":
    main()

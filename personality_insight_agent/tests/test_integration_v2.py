"""
端到端集成测试 — V2 三大组件集成验证

测试场景:
1. Corpus2Skill: 从多文件目录构建知识树
2. Agentic RAG: 自主规划 → 检索 → 反思循环
3. 双引擎验证: 事实核查 + 冲突仲裁
4. 完整 Pipeline: 纯文本模式的 V2 全流程
5. 完整 Pipeline: 含知识库的 V2 全流程
"""

import json
import os
import sys
import tempfile
from typing import Any, Dict, List

# ---- Mock LLM (模拟真实 LLM 响应) ----

def mock_llm(system_prompt: str, user_prompt: str) -> str:
    """模拟 LLM，返回合理的 JSON 结构"""
    system_lower = system_prompt.lower()

    # Corpus2Skill 摘要生成
    if "知识库分析专家" in system_prompt or "文档标题" in system_prompt:
        return json.dumps({
            "title": "测试文档",
            "summary": "这是一份关于商业决策方法论的系统性文档，包含多个投资案例分析和战略复盘。",
            "key_topics": ["商业决策", "投资策略", "战略规划"],
            "key_entities": ["公司A", "公司B"],
            "evidence_level": "primary",
        }, ensure_ascii=False)

    # Planner 类别增强
    if "研究规划专家" in system_prompt:
        return "decisions, writings, conversations"

    if "检索策略专家" in system_prompt:
        return "从反面分析商业决策的失败案例\n从时间维度追踪决策结果\n与同行的决策模式做对比"

    # Fact Checker 证据评估
    if "事实核查" in system_prompt or "事实核查员" in system_prompt:
        return json.dumps({
            "strength": "likely",
            "reasoning": "证据支持该声明，但存在一些不确定性",
            "key_evidence": "该人在多个场合表达了类似观点",
        }, ensure_ascii=False)

    # Fact Checker LLM claims filter
    if "选出最重要的" in system_prompt:
        # Return first 5 item IDs
        item_ids = []
        for line in user_prompt.split("\n"):
            if line.strip().startswith("- [") and "]" in line:
                item_id = line.split("]")[0].split("[")[1]
                item_ids.append(item_id)
        return json.dumps(item_ids[:5], ensure_ascii=False)

    # Arbitrator deep dive
    if "仲裁专家" in system_prompt:
        return json.dumps({
            "analysis": "内部分析有多个信号支持，外部证据虽然不完全吻合但也不构成矛盾，可能是不同角度的描述。",
            "verdict": "双方观点可以调和: 内部分析准确，外部证据补充了额外信息。",
        }, ensure_ascii=False)

    # Reflector
    if "冲突检测" in system_prompt:
        return json.dumps([
            {
                "type": "temporal",
                "description": "早期和晚期对话中对该策略的态度有明显转变",
                "source_a": "2023年访谈",
                "source_b": "2025年访谈",
                "severity": "medium",
            }
        ], ensure_ascii=False)

    # Skill1 (must check before Skill2/3/4/5 since those prompts mention Skill1)
    if "素材检索" in system_prompt or "## 角色: 素材检索" in system_prompt:
        return json.dumps({
            "material": {"cleaned_text": user_prompt[:500], "chunks": []},
            "quality_score": 0.7,
            "limitations": ["测试模式"],
        }, ensure_ascii=False)

    # Skill4/5 check before Skill2/3 since their prompts mention "Skill2"/"Skill3"
    if "认知与思维特征专家" in system_prompt or "认知与思维特征 (Skill 2" in system_prompt:
        return json.dumps({
            "cognitive_profile": {
                "info_processing": {"style": "balanced_analytical", "detail_orientation": 4},
                "decision_pattern": {"mode": "deliberative", "speed": 2},
                "risk_attitude": {"orientation": "moderate_conservative"},
                "attribution_style": "internal",
                "time_orientation": "future_oriented",
            },
            "mental_models": [
                {
                    "name": "渐进验证",
                    "description": "不确定时先小步试，验证后再扩大",
                    "cross_domain_evidence": ["产品决策", "投资决策"],
                    "triple_check": {"cross_domain": True, "generative": True, "exclusive": True},
                }
            ],
            "quality": 0.85,
            "uncertainty": ["缺少高压决策场景的材料"],
        }, ensure_ascii=False)

    if "语言风格与情感模式分析师" in system_prompt or "语言风格与情感 (Skill 3" in system_prompt:
        return json.dumps({
            "language_style": {
                "sentence_fingerprint": {"avg_sentence_length": 28.5, "question_ratio": 0.12},
                "style_tags": {"formality": 3, "abstractness": 4},
                "emotion": {"valence": 0.2, "arousal": 0.35},
                "rhetoric": {"metaphor_style": "business_analogies"},
                "signature_patterns": [
                    "高频使用'其实'作为转折引导",
                    "偏好'我觉得'而非'我认为'",
                ],
            },
            "quality": 0.80,
            "uncertainty": ["情感表达场景较少"],
        }, ensure_ascii=False)

    if "性格特质推断专家" in system_prompt or "性格特质推断 (Skill 4)" in system_prompt:
        return json.dumps({
            "personality": {
                "big_five": {
                    "openness": {"score": 4, "confidence": 0.8, "signals": [], "evidence": []},
                    "conscientiousness": {"score": 4, "confidence": 0.85, "signals": [], "evidence": []},
                    "extraversion": {"score": 2, "confidence": 0.6, "signals": [], "evidence": []},
                    "agreeableness": {"score": 4, "confidence": 0.75, "signals": [], "evidence": []},
                    "neuroticism": {"score": 2, "confidence": 0.7, "signals": [], "evidence": []},
                },
                "mbti": {"type": "ISTJ", "confidence": 0.72},
                "motivation": {"achievement": {"score": 4, "rank": 1}},
                "pdp": {"style": "猫头鹰-考拉"},
            },
            "consistency_check": {"intra_framework": "pass"},
            "quality": 0.75,
            "uncertainty": {"extraversion": "社交场景材料不足"},
        }, ensure_ascii=False)

    # Skill5
    if "画像合成与报告生成专家" in system_prompt or "画像合成 (Skill 5)" in system_prompt:
        return json.dumps({
            "person_id": "test_user",
            "metadata": {
                "analysis_date": "2026-05-23T10:00:00",
                "overall_confidence": 0.78,
                "sources": [{"type": "test", "period": "2024-01~2024-06"}],
            },
            "core_profile": {
                "thinking_style": "渐进分析型决策者",
                "personality_snapshot": "内敛可靠的ISTJ型",
                "communication_essence": "审慎表达，以事实为基础",
                "core_motivation": "通过稳健积累实现专业价值",
            },
            "insights": {
                "strengths": ["风险意识强"],
                "risks": ["可能错失时机"],
                "growth_areas": ["练习快速决策"],
                "blind_spots": ["可能低估直觉"],
            },
            "contradictions": [
                {"type": "essential_tension", "description": "稳定vs创新", "evidence": []},
            ],
            "honesty_boundary": {
                "known": ["工作场景模式"],
                "uncertain": ["社交外向性"],
                "unknown": ["高压场景"],
                "material_limitations": ["仅6个月聊天记录"],
            },
            "report": "# Test Report\nThis is a dry run report.",
        }, ensure_ascii=False)

    # Agentic RAG synthesize
    if "研究综合" in system_prompt:
        return f"综合研究结果: {user_prompt[:200]}..."

    # Default
    return json.dumps({"status": "ok"}, ensure_ascii=False)


# ---- 测试数据 ----

SAMPLE_CHAT = """
[2024-03-15 产品评审群]
张三: 新功能我看了竞品分析，A和B都做了，但方向不一样。A侧重C端体验，B侧重B端效率。
张三: 我建议先不急着决定，我们各做一个小原型跑一周数据。
张三: 成本的话，两个人一周应该能出东西。
李四: 客户那边一直在催，能不能直接做B的方案？
张三: 催归催，做错了更浪费时间。原型验证很快的，一周后拿数据说话。
张三: 而且说实话，我直觉也觉得B的方向可能不太对，但需要用数据验证一下。

[2024-04-02 战略讨论群]
张三: 上次的原型结果出来了，A方案的留存率高30%，虽然前期开发成本高一点。
张三: 我的建议还是做A，先把体验做好，再考虑效率优化。
王五: B方案那边客户给了很大压力。
张三: 数据面前人人平等哈哈。把A的数据拿给客户看，比我们说什么都强。
张三: 我觉得做产品最重要的是先做对，再做快。顺序反了反而更慢。

[2024-05-10 团队周会]
张三: 这个季度最大的收获是验证了我们之前的假设——小步快跑确实比一步到位靠谱。
张三: 其实做决策最怕的不是信息不够，是信息够了自己还在犹豫。
张三: 我最近在看一本关于决策心理学的书，里面有个观点很认同——判断一个人的决策质量，不要看结果，要看过程。
"""

SAMPLE_CORPUS_FILES = {
    "decisions/investment_memo_2024.md": """# 2024年投资决策备忘录

## Q1 复盘
投资项目A: 经过3个月的尽职调查，决定以500万估值投资。关键考量:
1. 团队执行能力强 (创始人连续创业，有成功退出经验)
2. 市场空间大 (TAM 100亿+)
3. 产品差异化明显

## Q2 策略调整
经过Q1的实践，调整了投资逻辑:
- 从"看赛道"转向"看人+赛道"
- 增加了投后管理的资源投入
- 退出策略从IPO扩展到并购

## 决策原则总结
1. 不投资看不懂的领域
2. 团队第一，赛道第二
3. 宁可错过，不要过错
4. 每个季度复盘一次决策质量
""",

    "writings/decision_philosophy.md": """# 我的决策哲学

做了十年投资，总结几条对我影响最大的原则:

## 1. 逆向思考
每次做决策前，先问自己: "如果这个决策失败了，最可能的原因是什么？"
这个方法帮我避开了至少3个重大陷阱。

## 2. 概率思维
不存在100%确定的事。与其追求确定性，不如评估概率和赔率。
我习惯给每个决策标一个成功概率，然后下对应的注。

## 3. 时间维度
短期的好决策可能是长期的坏决策。反过来也一样。
所以我做决策时，会刻意拉长时间维度——这个决策在3年、5年、10年后会怎样？

## 4. 反共识
最赚钱的投资往往是最不被看好的。但要区分"反共识"和"反常识"——
反共识是对的，反常识是错的。
""",
}


# ========================================================================
# 测试 1: Corpus2Skill 离线流水线
# ========================================================================

def test_corpus2skill_builder():
    """测试知识树构建"""
    print("\n" + "=" * 60)
    print("Test 1: Corpus2Skill 离线流水线")
    print("=" * 60)

    from personality_insight_agent.corpus2skill.builder import CorpusTreeBuilder

    # 创建临时目录模拟私有文档库
    with tempfile.TemporaryDirectory() as tmpdir:
        # 写入测试文件
        for rel_path, content in SAMPLE_CORPUS_FILES.items():
            full_path = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        # 构建知识树
        builder = CorpusTreeBuilder(vector_store_root=os.path.join(tmpdir, "vectors"))
        tree = builder.build(
            llm_callable=mock_llm,
            source_dir=tmpdir,
            person_id="test_user",
        )

        # 验证
        assert tree.tree_id == "test_user"
        assert tree.total_documents > 0, f"应该找到文档，实际total_documents={tree.total_documents}"
        assert tree.root.node_type.value == "root"
        assert len(tree.index_registry) > 0, "应该有至少一个 INDEX"

        print(f"  [PASS] 知识树构建: {tree.total_documents} 文档, {tree.total_words} 字")
        print(f"  [PASS] INDEX数量: {len(tree.index_registry)}")
        print(f"  [PASS] 文档注册表: {len(tree.doc_registry)} 个叶子节点")

        # 验证导航器
        from personality_insight_agent.corpus2skill.navigator import KnowledgeNavigator
        nav = KnowledgeNavigator(tree)
        ctx = nav.start_navigation()
        assert ctx.current_node_id == "root"
        assert len(ctx.visible_children) > 0

        print(f"  [PASS] 导航器初始化: {len(ctx.visible_children)} 个子节点可见")

        # 验证路径规划
        plan = nav.plan_route("商业决策")
        assert len(plan.steps) > 0
        print(f"  [PASS] 路径规划: {len(plan.steps)} 个步骤, 用于指令 '商业决策'")

    print("  [ALL PASS] Corpus2Skill 测试通过")


# ========================================================================
# 测试 2: Agentic RAG 规划-检索-反思循环
# ========================================================================

def test_agentic_rag_loop():
    """测试 Agentic RAG 完整循环"""
    print("\n" + "=" * 60)
    print("Test 2: Agentic RAG 循环")
    print("=" * 60)

    from personality_insight_agent.agentic_rag.planner import ResearchPlanner
    from personality_insight_agent.agentic_rag.reflector import ResearchReflector
    from personality_insight_agent.agentic_rag.retriever import RetrievalResult
    from personality_insight_agent.agentic_rag.models import ResearchDepth, ResearchLoopConfig

    # 2a. 测试 Planner
    planner = ResearchPlanner()
    plan = planner.create_plan(
        instruction="分析此人的重大商业决策模式",
        llm_callable=mock_llm,
        research_depth=ResearchDepth.STANDARD,
    )

    assert len(plan.tasks) > 0
    assert plan.research_depth == ResearchDepth.STANDARD
    print(f"  [PASS] Planner: {len(plan.tasks)} 个任务, 预计 {plan.estimated_rounds} 轮")
    for t in plan.tasks:
        print(f"    - {t.task_id}: {t.description[:80]}... (优先级={t.priority})")

    # 2b. 测试 Retriever (无向量库，退化为关键词)
    from personality_insight_agent.agentic_rag.retriever import DeepRetriever
    retriever = DeepRetriever(tree=None, vector_search_fn=None)
    result = retriever.keyword_search(
        keywords=["商业决策", "投资", "战略"],
        categories=["decisions", "writings"],
    )
    print(f"  [PASS] Retriever: 关键词检索返回 {len(result.chunks)} 个结果")

    # 2c. 测试 Reflector
    config = ResearchLoopConfig(max_rounds=3, early_stop_quality=0.85)
    reflector = ResearchReflector(config)

    # 模拟一些检索结果
    mock_results = [
        RetrievalResult(
            query="商业决策",
            chunks=[
                {"content": "此人偏好数据驱动的决策，多次强调'先验证再扩大'。", "source": "chat_01", "chunk_id": "c1"},
                {"content": "在投资决策中表现出明显的保守倾向，优先考虑下行风险。", "source": "memo_01", "chunk_id": "c2"},
            ],
            sources=["chat_01", "memo_01"],
            categories_covered=["decisions", "writings"],
            relevance_score=0.8,
        ),
        RetrievalResult(
            query="逆向思考",
            chunks=[
                {"content": "每次做决策前先问自己：如果失败了，最可能的原因是什么？", "source": "article_01", "chunk_id": "c3"},
            ],
            sources=["article_01"],
            categories_covered=["writings"],
            relevance_score=0.7,
        ),
    ]

    reflection = reflector.reflect(
        results=mock_results,
        original_instruction="分析此人的商业决策模式",
        llm_callable=mock_llm,
    )

    assert reflection.quality_score > 0
    print(f"  [PASS] Reflector: 质量={reflection.quality_score:.2f}, 信息充分={reflection.information_sufficient}")
    print(f"    - 冲突: {len(reflection.conflicts_found)} 个")
    print(f"    - 缺口: {len(reflection.gaps)} 个")
    print(f"    - 继续: {reflection.should_continue}")

    # 2d. 测试 ResearchAgent
    from personality_insight_agent.agentic_rag.research_agent import AgenticResearchAgent

    agent = AgenticResearchAgent(
        tree=None,
        vector_search_fn=None,
        agent_name="TestAgent",
    )

    result = agent.research(
        instruction="分析此人的决策模式",
        llm_callable=mock_llm,
        research_depth=ResearchDepth.QUICK,
    )

    assert result["success"]
    assert result["total_rounds"] > 0
    print(f"  [PASS] ResearchAgent: {result['total_rounds']} 轮, {result['total_chunks']} 块")

    print("  [ALL PASS] Agentic RAG 测试通过")


# ========================================================================
# 测试 3: 双引擎验证
# ========================================================================

def test_dual_engine_verification():
    """测试事实核查 + 仲裁"""
    print("\n" + "=" * 60)
    print("Test 3: 双引擎验证")
    print("=" * 60)

    from personality_insight_agent.verification.fact_checker import FactCheckAgent
    from personality_insight_agent.verification.arbitrator import DualEngineArbitrator
    from personality_insight_agent.verification.models import EvidenceStrength

    # 3a. Fact Checker
    checker = FactCheckAgent(tree=None, vector_search_fn=None)

    skill_outputs = {
        "skill2": {
            "cognitive_profile": {
                "mental_models": [
                    {
                        "name": "渐进验证",
                        "description": "不确定时先小步试",
                        "evidence": ["多次提到'先做原型'、'跑一周数据'"],
                    }
                ]
            }
        },
        "skill3": {
            "language_style": {
                "signature_patterns": [
                    "高频使用'其实'作为转折引导",
                    "偏好'我觉得'而非'我认为'",
                ]
            }
        },
        "skill4": {
            "personality": {
                "big_five": {
                    "conscientiousness": {"score": 4, "confidence": 0.85, "evidence": []},
                    "extraversion": {"score": 2, "confidence": 0.6, "evidence": []},
                },
                "mbti": {"type": "ISTJ", "confidence": 0.72},
            }
        },
    }

    verification = checker.verify(
        skill_outputs=skill_outputs,
        llm_callable=mock_llm,
    )

    assert verification.total_claims > 0
    print(f"  [PASS] Fact Checker: {verification.total_claims} 条声明核查")
    print(f"    - 验证通过: {verification.verified_count}")
    print(f"    - 被推翻: {verification.contradicted_count}")
    print(f"    - 不确定: {verification.uncertain_count}")
    print(f"    - 整体可信度: {verification.overall_credibility:.0%}")

    # 3b. Arbitrator (手动制造一个被推翻的声明来测试仲裁)
    from personality_insight_agent.verification.models import FactCheckItem
    disputed_item = FactCheckItem(
        item_id="test_dispute",
        claim="此人是极端冒险型决策者",
        source="Skill2 内部分析",
        category="trait",
        internal_analysis="从对话中推断此人偏好高风险决策",
        internal_confidence=0.3,  # 低置信度
        external_evidence="知识库中的投资备忘录显示此人多次强调'保守'和'风险控制'",
        external_strength=EvidenceStrength.CONTRADICTED,
        external_sources=["investment_memo_2024.md"],
        verification_result="contradicted",
    )

    verification.disputes_requiring_arbitration.append(disputed_item)
    verification.contradicted_count += 1

    arbitrator = DualEngineArbitrator()
    arbitration = arbitrator.arbitrate(
        verification=verification,
        llm_callable=mock_llm,
    )

    assert arbitration.total_conflicts > 0
    print(f"  [PASS] Arbitrator: {arbitration.total_conflicts} 条冲突仲裁")
    print(f"    - 已解决: {arbitration.resolved_count}")
    print(f"    - 标记争议: {arbitration.disputed_count}")
    print(f"    - 策略分布: {arbitration.strategy_usage}")

    if arbitration.recommendations:
        print(f"    - 建议修改: {len(arbitration.recommendations)} 条")
        for rec in arbitration.recommendations[:3]:
            print(f"      * {rec[:100]}")

    print("  [ALL PASS] 双引擎验证测试通过")


# ========================================================================
# 测试 4: 完整 V2 Pipeline (纯文本模式)
# ========================================================================

def test_v2_pipeline_text_only():
    """测试纯文本模式的 V2 Pipeline"""
    print("\n" + "=" * 60)
    print("Test 4: V2 Pipeline (纯文本模式)")
    print("=" * 60)

    from personality_insight_agent.orchestrator_v2 import PersonalityInsightOrchestratorV2

    orch = PersonalityInsightOrchestratorV2()
    result = orch.analyze(
        llm_callable=mock_llm,
        raw_text=SAMPLE_CHAT,
        person_id="test_zhangsan",
    )

    assert not result.degradation_triggered, f"不应触发降级: {result.degradation_reason}"
    assert len(result.pipeline_used) >= 5
    assert result.final_output is not None

    print(f"  [PASS] Pipeline: {' → '.join(result.pipeline_used)}")
    for step in result.pipeline_steps:
        print(f"    - {step.step_name}: quality={step.quality:.2f}")

    output = result.final_output
    assert output.core_profile.thinking_style
    assert output.metadata.overall_confidence > 0

    print(f"  [PASS] 核心画像: {output.core_profile.thinking_style}")
    print(f"  [PASS] 置信度: {output.metadata.overall_confidence:.2%}")
    print(f"  [PASS] 诚实边界: known={len(output.honesty_boundary.known)}, uncertain={len(output.honesty_boundary.uncertain)}")

    print("  [ALL PASS] V2 纯文本模式测试通过")


# ========================================================================
# 测试 5: V2 Pipeline (含知识库) — 仅结构测试
# ========================================================================

def test_v2_pipeline_with_corpus():
    """测试含知识库的 V2 Pipeline 结构（dry_run 模式）"""
    print("\n" + "=" * 60)
    print("Test 5: V2 Pipeline (含知识库结构测试)")
    print("=" * 60)

    from personality_insight_agent.orchestrator_v2 import PersonalityInsightOrchestratorV2
    from personality_insight_agent.agentic_rag.models import ResearchDepth

    # 创建临时知识库目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 写入测试文件
        for rel_path, content in SAMPLE_CORPUS_FILES.items():
            full_path = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        orch = PersonalityInsightOrchestratorV2()

        # 用 QUICK 模式减少测试时间
        result = orch.analyze_with_corpus(
            llm_callable=mock_llm,
            raw_text=SAMPLE_CHAT,
            person_id="test_zhangsan",
            corpus_dir=tmpdir,
            vector_search_fn=None,  # 退化为 LLM 知识检索
            research_depth=ResearchDepth.QUICK,
        )

        # 验证结构
        assert "orchestrator_output" in result
        assert "agentic_research" in result
        assert "corpus_tree" in result

        orch_out = result["orchestrator_output"]
        if orch_out.final_output:
            print(f"  [PASS] Orchestrator: {len(orch_out.pipeline_used)} 阶段")
            for step in orch_out.pipeline_steps:
                print(f"    - {step.step_name}: status={step.status}")

        if result["corpus_tree"]:
            print(f"  [PASS] 知识树: {result['corpus_tree'].total_documents} 文档")

        research = result["agentic_research"]
        print(f"  [PASS] Agentic RAG: {len(research)} 个 Agent")
        success_count = sum(1 for r in research if r.get("success"))
        print(f"    - 成功: {success_count}/{len(research)}")

        extended = result.get("extended_report", "")
        print(f"  [PASS] 扩展报告: {len(extended)} 字符")
        # 验证报告包含关键章节
        assert "核心画像" in extended or "core_profile" in extended.lower() or len(extended) > 100

    print("  [ALL PASS] V2 含知识库测试通过")


# ========================================================================
# 测试 6: 仲裁决策矩阵
# ========================================================================

def test_arbitration_strategy_matrix():
    """测试仲裁策略决策矩阵的各种情况"""
    print("\n" + "=" * 60)
    print("Test 6: 仲裁决策矩阵")
    print("=" * 60)

    from personality_insight_agent.verification.arbitrator import DualEngineArbitrator
    from personality_insight_agent.verification.models import (
        FactCheckItem, VerificationResult, EvidenceStrength,
    )

    arb = DualEngineArbitrator()

    test_cases = [
        # (internal_confidence, external_strength, expected_strategy)
        (0.9, EvidenceStrength.CONFIRMED, "merge"),
        (0.9, EvidenceStrength.NOT_FOUND, "prefer_internal"),
        (0.9, EvidenceStrength.CONTRADICTED, "flag_as_disputed"),
        (0.5, EvidenceStrength.CONTRADICTED, "deep_dive"),
        (0.3, EvidenceStrength.CONFIRMED, "prefer_external"),
        (0.3, EvidenceStrength.CONTRADICTED, "prefer_external"),
        (0.5, EvidenceStrength.UNCERTAIN, "flag_as_disputed"),
    ]

    for conf, strength, expected in test_cases:
        item = FactCheckItem(
            item_id=f"test_{conf}_{strength.value}",
            claim=f"测试声明 (conf={conf}, ext={strength.value})",
            source="test",
            category="trait",
            internal_confidence=conf,
            external_strength=strength,
        )
        strategy = arb._select_strategy(item)
        assert strategy.value == expected, f"{conf}x{strength.value}: 期望{expected}, 实际{strategy.value}"
        print(f"  [PASS] conf={conf:.1f} × ext={strength.value} → {strategy.value}")

    print("  [ALL PASS] 仲裁决策矩阵测试通过")


# ========================================================================
# 运行
# ========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Personality Insight Agent V2 — 集成测试套件")
    print("=" * 60)

    try:
        test_corpus2skill_builder()
        test_agentic_rag_loop()
        test_dual_engine_verification()
        test_v2_pipeline_text_only()
        test_v2_pipeline_with_corpus()
        test_arbitration_strategy_matrix()

        print("\n" + "=" * 60)
        print("  全部测试通过!")
        print("=" * 60)
    except Exception as e:
        import traceback
        print(f"\n[FAIL] 测试失败: {e}")
        traceback.print_exc()
        sys.exit(1)

"""
Agentic RAG — Planner (自主规划)

核心职责:
- 将高级研究指令分解为可执行的检索任务
- 规划知识树中的导航路径
- 确定任务优先级和依赖关系
- 根据反思结果动态调整计划
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .models import (
    AgentAction,
    ResearchDepth,
    ResearchPlan,
    ResearchTask,
    ReflectionResult,
    RetrievalResult,
)


class ResearchPlanner:
    """
    研究规划器 — 自主规划检索策略。

    用法:
        planner = ResearchPlanner()
        plan = planner.create_plan(
            "分析此人的重大商业决策模式",
            llm_callable,
            knowledge_tree_summary,
        )
    """

    CATEGORY_TEMPLATES = {
        "writings": {"description": "分析著作和系统性长文中关于{instruction}的内容", "aspects": ["核心论点", "方法论", "反复出现的主题", "自创术语"], "priority": 3},
        "conversations": {"description": "分析对话和访谈中关于{instruction}的即兴反应", "aspects": ["被追问时的回答", "即兴类比", "立场变化"], "priority": 4},
        "expressions": {"description": "分析碎片表达中与{instruction}相关的风格信号", "aspects": ["高频用词", "语气变化", "情感倾向", "争议立场"], "priority": 5},
        "external_views": {"description": "收集外部视角对{instruction}的评价和批评", "aspects": ["同行评价", "争议", "与同行的差异"], "priority": 6},
        "decisions": {"description": "分析重大决策记录中与{instruction}相关的案例", "aspects": ["决策背景", "决策逻辑", "事后反思", "言行一致性"], "priority": 1},
        "timeline": {"description": "从时间线中提取与{instruction}相关的关键节点", "aspects": ["关键里程碑", "思想转折点", "最近动态", "长期趋势"], "priority": 7},
        "generic": {"description": "搜索通用资料中与{instruction}相关的内容", "aspects": ["相关内容"], "priority": 8},
    }

    # ========================================================================
    # 主入口
    # ========================================================================

    def create_plan(
        self,
        instruction: str,
        llm_callable: Callable[[str, str], str],
        knowledge_tree_summary: str = "",
        research_depth: ResearchDepth = ResearchDepth.STANDARD,
        previous_results: Optional[list[RetrievalResult]] = None,
    ) -> ResearchPlan:
        """
        创建研究计划。

        Args:
            instruction: 用户的原始指令
            llm_callable: LLM调用函数
            knowledge_tree_summary: Corpus2Skill树的结构摘要
            research_depth: 研究深度
            previous_results: 之前的检索结果 (用于replan)

        Returns:
            ResearchPlan: 完整的研究计划
        """
        is_replan = previous_results is not None and len(previous_results) > 0

        if is_replan:
            return self._replan(instruction, llm_callable, knowledge_tree_summary, previous_results, research_depth)

        return self._initial_plan(instruction, llm_callable, knowledge_tree_summary, research_depth)

    # ========================================================================
    # 初始规划
    # ========================================================================

    def _initial_plan(
        self,
        instruction: str,
        llm_callable: Callable[[str, str], str],
        knowledge_tree_summary: str,
        research_depth: ResearchDepth,
    ) -> ResearchPlan:
        """首次规划"""
        # Step 1: 确定需要覆盖的类别
        categories = self._determine_categories(instruction, knowledge_tree_summary, llm_callable)

        # Step 2: 为每个类别生成 ResearchTask
        tasks: list[ResearchTask] = []
        for i, cat in enumerate(categories):
            template = self.CATEGORY_TEMPLATES.get(cat, self.CATEGORY_TEMPLATES["generic"])
            task = ResearchTask(
                task_id=f"task_{i+1:02d}_{cat}",
                description=template["description"].format(instruction=instruction),
                target_categories=[cat],
                search_queries=self._generate_queries(instruction, cat, template["aspects"]),
                knowledge_path=["root", f"index_{cat}"],
                priority=template["priority"],
            )
            tasks.append(task)

        # Step 3: 设置轮数上限
        depth_config = {
            ResearchDepth.QUICK: (1, 2),
            ResearchDepth.STANDARD: (3, 5),
            ResearchDepth.DEEP: (5, 8),
        }
        estimated, max_rounds = depth_config.get(research_depth, (3, 5))

        plan = ResearchPlan(
            original_query=instruction,
            tasks=tasks,
            research_depth=research_depth,
            estimated_rounds=estimated,
            max_rounds=max_rounds,
            knowledge_tree_id=None,
            rationale=self._generate_rationale(instruction, categories, tasks),
        )

        return plan

    # ========================================================================
    # 重新规划 (Replan)
    # ========================================================================

    def _replan(
        self,
        instruction: str,
        llm_callable: Callable[[str, str], str],
        knowledge_tree_summary: str,
        previous_results: list[RetrievalResult],
        research_depth: ResearchDepth,
    ) -> ResearchPlan:
        """根据反思结果调整计划"""
        # 收集已有查询
        previous_queries = set()
        for r in previous_results:
            previous_queries.add(r.query)
            for src in r.sources:
                previous_queries.add(src)

        # 收集覆盖的类别
        covered_categories = set()
        for r in previous_results:
            for cat in r.categories_covered:
                covered_categories.add(cat)

        # 找出未覆盖的类别
        all_categories = set(self.CATEGORY_TEMPLATES.keys()) - {"generic"}
        missing_categories = all_categories - covered_categories

        # 新查询 (避免重复)
        new_queries = self._generate_novel_queries(instruction, previous_queries, llm_callable)

        tasks: list[ResearchTask] = []

        # 为缺失类别创建任务
        for i, cat in enumerate(missing_categories):
            template = self.CATEGORY_TEMPLATES.get(cat, self.CATEGORY_TEMPLATES["generic"])
            task = ResearchTask(
                task_id=f"replan_task_{i+1:02d}_{cat}",
                description=f"[补充] {template['description'].format(instruction=instruction)}",
                target_categories=[cat],
                search_queries=new_queries[:3],
                knowledge_path=["root", f"index_{cat}"],
                priority=template["priority"],
            )
            tasks.append(task)

        # 至少加一个新查询任务
        if not tasks and new_queries:
            tasks.append(ResearchTask(
                task_id="replan_deep_dive",
                description=f"[深度检索] 用新角度探索: {new_queries[0]}",
                target_categories=list(all_categories),
                search_queries=new_queries[:5],
                knowledge_path=["root"],
                priority=1,
            ))

        plan = ResearchPlan(
            original_query=instruction,
            tasks=tasks,
            research_depth=research_depth,
            estimated_rounds=1,
            max_rounds=2,
            rationale=f"重新规划: 已覆盖类别={covered_categories}, 缺失={missing_categories}",
        )

        return plan

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _determine_categories(
        self,
        instruction: str,
        tree_summary: str,
        llm_callable: Callable[[str, str], str],
    ) -> list[str]:
        """确定需要检索的内容类别"""
        instruction_lower = instruction.lower()

        scored = []

        # 规则层面打分
        category_signals = {
            "decisions": ["决策", "决定", "选择", "战略", "商业", "投资", "收购", "合作", "转型"],
            "writings": ["思想", "理念", "理论", "方法", "原则", "框架", "模型", "哲学"],
            "conversations": ["访谈", "对话", "即兴", "观点", "态度", "反应", "场合"],
            "expressions": ["风格", "表达", "语气", "用词", "修辞", "沟通", "说话"],
            "external_views": ["评价", "批评", "争议", "形象", "公众", "媒体", "同行"],
            "timeline": ["经历", "时间", "变化", "转变", "成长", "演进", "历史"],
        }

        for cat, signals in category_signals.items():
            score = sum(1 for s in signals if s in instruction_lower)
            if score > 0:
                scored.append((cat, score))

        # 按得分排序
        scored.sort(key=lambda x: x[1], reverse=True)

        # 默认至少包含 decisions 和 writings
        categories = [c for c, _ in scored[:5]]
        if "decisions" not in categories:
            categories.append("decisions")
        if "writings" not in categories:
            categories.append("writings")

        # 如 LLM 可用，做语义增强
        if llm_callable and len(scored) > 3:
            try:
                categories = self._llm_enhance_categories(llm_callable, instruction, categories)
            except Exception:
                pass

        return categories

    def _llm_enhance_categories(
        self,
        llm_callable: Callable[[str, str], str],
        instruction: str,
        current: list[str],
    ) -> list[str]:
        """用 LLM 优化类别选择"""
        prompt = f"""研究指令: "{instruction}"

当前选择的分析维度: {', '.join(current)}

可选维度: writings(著作), conversations(对话), expressions(表达), external_views(外部视角), decisions(决策), timeline(时间线)

请判断:
1. 是否需要增加或移除维度？
2. 重要度排序是否合理？

输出格式: 直接输出调整后的维度列表，用逗号分隔，如 "decisions, writings, conversations" """

        resp = llm_callable("你是研究规划专家。", prompt)
        # 简单解析
        cats = [c.strip() for c in resp.strip().split(",")]
        valid = []
        all_valid = set(self.CATEGORY_TEMPLATES.keys()) - {"generic"}
        for c in cats:
            c_clean = c.strip().lower()
            if c_clean in all_valid:
                valid.append(c_clean)
        return valid if valid else current

    def _generate_queries(
        self,
        instruction: str,
        category: str,
        aspects: list[str],
    ) -> list[str]:
        """为指定类别生成具体检索查询"""
        queries = []
        for aspect in aspects:
            queries.append(f"{instruction} {aspect}")
        return queries

    def _generate_novel_queries(
        self,
        instruction: str,
        previous_queries: set[str],
        llm_callable: Callable[[str, str], str],
    ) -> list[str]:
        """生成与之前检索不重复的新查询"""
        novel = []

        # 规则层面: 换角度提问
        perspectives = [
            f"从反面看: {instruction}",
            f"具体案例: {instruction}",
            f"时间变化: {instruction}",
            f"同行比较: {instruction}",
        ]
        for p in perspectives:
            if p not in previous_queries:
                novel.append(p)

        # LLM 增强
        if llm_callable:
            try:
                prev_str = "\n".join(f"- {q}" for q in list(previous_queries)[:10])
                prompt = f"""原始指令: "{instruction}"
已检索过的查询:
{prev_str}

请生成3个新的、不同角度的检索查询，避免与已有查询重复。

直接输出查询列表，每行一个，不要编号。"""

                resp = llm_callable("你是信息检索策略专家。", prompt)
                for line in resp.strip().split("\n"):
                    q = line.strip().lstrip("- ").strip()
                    if q and q not in previous_queries:
                        novel.append(q)
            except Exception:
                pass

        return novel[:5]

    def _generate_rationale(
        self,
        instruction: str,
        categories: list[str],
        tasks: list[ResearchTask],
    ) -> str:
        """生成规划理由"""
        return (
            f"针对指令「{instruction}」，计划覆盖{categories} "
            f"共{len(tasks)}个检索维度。"
            f"按优先级排序执行，确保高优维度(如决策记录)优先获取。"
        )

    # ========================================================================
    # 快速计划 (用于简单查询)
    # ========================================================================

    def quick_plan(self, query: str, categories: Optional[list[str]] = None) -> ResearchPlan:
        """为简单查询生成快速计划 (不调用 LLM)"""
        cats = categories or ["decisions", "writings"]
        tasks = []

        for i, cat in enumerate(cats):
            template = self.CATEGORY_TASK_TEMPLATES.get(cat, self.CATEGORY_TASK_TEMPLATES["generic"])
            task = ResearchTask(
                task_id=f"quick_{i+1:02d}_{cat}",
                description=f"搜索{cat}分类中关于 '{query}' 的内容",
                target_categories=[cat],
                search_queries=[query],
                knowledge_path=["root", f"index_{cat}"],
                priority=i + 1,
            )
            tasks.append(task)

        return ResearchPlan(
            original_query=query,
            tasks=tasks,
            research_depth=ResearchDepth.QUICK,
            estimated_rounds=1,
            max_rounds=1,
            rationale=f"快速搜索: {query}",
        )

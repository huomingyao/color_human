"""
Corpus2Skill — 知识导航器

在线导航：Agent 使用导航器在知识树中按逻辑路径移动，
获取上下文窗口内容，而非盲目在全库中搜索。

核心能力:
- resolve_path: 将高级指令(如"商业决策")解析为知识树路径
- navigate: 按计划步骤在树中移动，获取上下文
- expand_context: 获取当前节点+相邻节点的内容窗口
- backtrack: 回溯到上级节点或之前访问过的节点
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .models import (
    ContentCategory,
    KnowledgeTreeNode,
    NavigationContext,
    NavigationPlan,
    NavigationStep,
    SkillDirectoryTree,
)


class KnowledgeNavigator:
    """
    知识树导航器。

    供 Agentic RAG 中的 ResearchAgent 使用，提供"导航式"而非"搜索式"的知识获取。

    用法:
        nav = KnowledgeNavigator(tree)
        ctx = nav.start_navigation()

        # Agent 说: "我想看商业决策相关的内容"
        plan = nav.plan_route("商业决策", llm_callable)
        for step in plan.steps:
            ctx = nav.navigate(step)
            results = nav.retrieve_from_context(ctx, top_k=5)
    """

    def __init__(self, tree: SkillDirectoryTree):
        self.tree = tree
        self._current_context: Optional[NavigationContext] = None
        self._node_lookup: dict[str, KnowledgeTreeNode] = {}

        # 构建 O(1) 查找表
        self._build_lookup(tree.root)

    def _build_lookup(self, node: KnowledgeTreeNode):
        """递归构建节点查找表"""
        self._node_lookup[node.node_id] = node
        for child in node.children:
            self._build_lookup(child)

    # ========================================================================
    # 导航初始化
    # ========================================================================

    def start_navigation(self) -> NavigationContext:
        """初始化导航上下文，Agent 从根节点开始"""
        root = self.tree.root
        visible = [
            {"id": child.node_id, "title": child.title, "summary": child.description}
            for child in root.children
        ]

        ctx = NavigationContext(
            current_node_id="root",
            visible_children=visible,
            suggested_next=[c["id"] for c in visible],
        )
        self._current_context = ctx
        return ctx

    # ========================================================================
    # 路径规划
    # ========================================================================

    def plan_route(
        self,
        instruction: str,
        llm_callable: Optional[Callable[[str, str], str]] = None,
    ) -> NavigationPlan:
        """
        将高级指令解析为知识树导航计划。

        例如 "重大商业决策" →
          Step 1: 去 decisions 分类查看投资备忘录
          Step 2: 去 writings 分类查看相关战略文章
          Step 3: 去 conversations 分类查看相关访谈

        Args:
            instruction: 用户/Agent的高级指令
            llm_callable: 可选的 LLM 用于智能路径规划

        Returns:
            NavigationPlan: 分解后的步骤序列
        """
        # 规则层面: 关键词 → 分类映射
        category_keywords = {
            ContentCategory.DECISIONS: [
                "决策", "决定", "选择", "战略", "商业", "投资", "并购", "合作",
                "风险", "取舍", "方向", "转型", "decision", "strategy",
            ],
            ContentCategory.WRITINGS: [
                "思想", "理念", "理论", "方法", "框架", "原则", "模型", "系统",
                "著作", "文章", "书", "philosophy", "principle",
            ],
            ContentCategory.CONVERSATIONS: [
                "对话", "访谈", "观点", "看法", "态度", "反应", "即兴",
                "交流", "会议", "interview", "opinion",
            ],
            ContentCategory.EXPRESSIONS: [
                "风格", "表达", "语气", "措辞", "修辞", "说话方式",
                "沟通", "style", "communication",
            ],
            ContentCategory.EXTERNAL: [
                "评价", "批评", "争议", "外部", "别人", "看法", "公众",
                "形象", "review", "criticism",
            ],
            ContentCategory.TIMELINE: [
                "经历", "履历", "时间", "变化", "转变", "转折", "成长",
                "timeline", "history", "evolution",
            ],
        }

        # 匹配相关分类
        relevant_categories: list[tuple[ContentCategory, int]] = []
        instruction_lower = instruction.lower()

        for category, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in instruction_lower)
            if score > 0:
                relevant_categories.append((category, score))

        # 按相关性排序
        relevant_categories.sort(key=lambda x: x[1], reverse=True)

        # 生成步骤
        steps: list[NavigationStep] = []
        for i, (category, score) in enumerate(relevant_categories[:5]):
            index_id = f"index_{category.value}"
            index = self.tree.index_registry.get(index_id)
            if not index:
                continue

            target_nodes = [child["id"] for child in index.children_overview[:10]]

            search_query = self._build_search_query(instruction, category)

            step = NavigationStep(
                step_order=i + 1,
                category=category,
                target_nodes=target_nodes,
                search_query=search_query,
                rationale=f"关键词匹配: {category.value} (相关性得分: {score})",
                is_optional=score < 2,
            )
            steps.append(step)

        # 如果 LLM 可用，增强规划
        if llm_callable and len(steps) > 1:
            steps = self._enhance_plan_with_llm(llm_callable, instruction, steps)

        # 设置降级节点
        all_remaining = []
        for idx in self.tree.index_registry.values():
            all_remaining.extend([c["id"] for c in idx.children_overview])

        fallback = all_remaining[:10]

        return NavigationPlan(
            instruction=instruction,
            steps=steps,
            estimated_depth=len(steps),
            fallback_nodes=fallback,
        )

    def _build_search_query(self, instruction: str, category: ContentCategory) -> str:
        """根据指令和分类构建检索查询"""
        category_context = {
            ContentCategory.DECISIONS: "决策 战略 投资 选择",
            ContentCategory.WRITINGS: "方法论 原理 框架 思想",
            ContentCategory.CONVERSATIONS: "访谈 对话 观点 态度",
            ContentCategory.EXPRESSIONS: "表达 风格 用词 修辞",
            ContentCategory.EXTERNAL: "评价 批评 争议 外部视角",
            ContentCategory.TIMELINE: "经历 转折 发展 时间线",
        }
        ctx = category_context.get(category, "")
        return f"{instruction} {ctx}"

    def _enhance_plan_with_llm(
        self,
        llm_callable: Callable[[str, str], str],
        instruction: str,
        steps: list[NavigationStep],
    ) -> list[NavigationStep]:
        """用 LLM 优化导航计划"""
        try:
            steps_desc = "\n".join(
                f"{s.step_order}. 分类: {s.category.value}, 理由: {s.rationale}"
                for s in steps
            )

            prompt = f"""原始分析指令: "{instruction}"

当前规划的导航步骤:
{steps_desc}

请评估这个导航计划是否合理。你只能调整:
1. 步骤顺序 (是否应该调整优先级)
2. 步骤的 optional 标记

输出格式:
```
步骤调整: (如无调整写"无需调整")
原因: (简要说明)
```"""

            resp = llm_callable(
                "你是知识导航规划专家。评估并优化导航路径。",
                prompt,
            )
            # 简单解析 LLM 反馈 (不修改步骤，作为元数据记录)
            # 完整实现会在 planner.py 中做
        except Exception:
            pass

        return steps

    # ========================================================================
    # 导航执行
    # ========================================================================

    def navigate(self, step: NavigationStep) -> NavigationContext:
        """
        执行导航步骤，移动 Agent 到目标分类节点并加载可见内容。
        """
        index_id = f"index_{step.category.value}"
        target_node = self._node_lookup.get(index_id)

        if not target_node:
            # 降级到根节点
            target_node = self.tree.root

        # 构建可见子节点列表
        visible_children = [
            {
                "id": child.node_id,
                "title": child.title,
                "summary": child.description[:200] if child.description else "",
            }
            for child in target_node.children
        ]

        # 建议下一步
        suggested = []
        if step.target_nodes:
            suggested = step.target_nodes[:5]

        ctx = NavigationContext(
            current_node_id=target_node.node_id,
            path_history=self._current_context.path_history if self._current_context else [],
            visible_children=visible_children,
            suggested_next=suggested,
        )

        # 记录本次跳转
        ctx.record_visit(target_node.node_id)

        self._current_context = ctx
        return ctx

    def go_to_node(self, node_id: str) -> Optional[NavigationContext]:
        """直接跳转到指定节点"""
        target = self._node_lookup.get(node_id)
        if not target:
            return None

        visible_children = [
            {
                "id": child.node_id,
                "title": child.title,
                "summary": child.description[:200] if child.description else "",
            }
            for child in target.children
        ]

        ctx = NavigationContext(
            current_node_id=node_id,
            path_history=self._current_context.path_history if self._current_context else [],
            visible_children=visible_children,
            suggested_next=[c["id"] for c in visible_children],
        )
        ctx.record_visit(node_id)
        self._current_context = ctx
        return ctx

    def go_back(self) -> Optional[NavigationContext]:
        """回溯到上一个节点"""
        if not self._current_context:
            return None

        prev_id = self._current_context.go_back()
        if prev_id:
            return self.go_to_node(prev_id)
        return None

    def go_up(self) -> Optional[NavigationContext]:
        """上溯到父节点"""
        if not self._current_context:
            return None

        current = self._node_lookup.get(self._current_context.current_node_id)
        if current and current.parent_id:
            return self.go_to_node(current.parent_id)
        return None

    # ========================================================================
    # 上下文检索
    # ========================================================================

    def retrieve_from_context(
        self,
        ctx: NavigationContext,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        从当前导航上下文中检索文档块。

        不盲目搜索整个向量库，而是在当前节点及其子节点的范围内检索。
        这就是 "导航 vs 搜索" 的核心差异。
        """
        current = self._node_lookup.get(ctx.current_node_id)
        if not current:
            return []

        # 收集范围内的所有 chunk_ids
        chunk_ids = self._collect_chunk_ids(current)

        # 如果指定了 suggested_next，优先从这些节点获取
        if ctx.suggested_next:
            priority_ids = []
            for node_id in ctx.suggested_next[:top_k]:
                node = self._node_lookup.get(node_id)
                if node:
                    priority_ids.extend(node.chunk_ids)
            # 优先chunk放前面
            chunk_ids = list(dict.fromkeys(priority_ids + chunk_ids))

        # 返回 chunk_id 列表，实际检索由 AgenticRAG 的 retriever 完成
        results = []
        for chunk_id in chunk_ids[:top_k * 3]:  # 适度超额获取，由 retriever 重排序
            results.append({
                "chunk_id": chunk_id,
                "node_id": ctx.current_node_id,
                "relevance": 1.0,  # 由 retriever 重新计算
            })

        return results[:top_k]

    def retrieve_by_category(
        self,
        category: ContentCategory,
        query: str = "",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        按内容分类检索，获取该分类下所有文档块的索引。

        配合外部向量检索使用：先在这里拿到目标 chunk_ids，
        再到 FAISS 中做语义相似度检索。
        """
        index_id = f"index_{category.value}"
        index_node = self._node_lookup.get(index_id)
        if not index_node:
            return []

        all_chunk_ids = self._collect_chunk_ids(index_node)

        results = []
        for chunk_id in all_chunk_ids[:top_k * 3]:
            results.append({
                "chunk_id": chunk_id,
                "category": category.value,
                "relevance": 0.8,  # 同分类默认较高相关
            })

        return results[:top_k]

    def _collect_chunk_ids(self, node: KnowledgeTreeNode) -> list[str]:
        """递归收集节点及其子节点的所有 chunk_ids"""
        ids = list(node.chunk_ids)

        for child in node.children:
            ids.extend(self._collect_chunk_ids(child))

        return ids

    # ========================================================================
    # 节点查询
    # ========================================================================

    def get_node(self, node_id: str) -> Optional[KnowledgeTreeNode]:
        """O(1) 节点查询"""
        return self._node_lookup.get(node_id)

    def get_index(self, index_id: str) -> Optional[Any]:
        """获取 INDEX.md 内容"""
        return self.tree.index_registry.get(index_id)

    def get_breadcrumb(self, node_id: str) -> list[str]:
        """获取节点的面包屑路径"""
        path = []
        current = self._node_lookup.get(node_id)
        while current:
            path.append(current.title)
            if current.parent_id:
                current = self._node_lookup.get(current.parent_id)
            else:
                break
        return list(reversed(path))

    def print_tree(self, node_id: str = "root", indent: int = 0) -> str:
        """以文本形式打印树结构（调试用）"""
        node = self._node_lookup.get(node_id)
        if not node:
            return ""

        lines = []
        prefix = "  " * indent + ("├─ " if indent > 0 else "")
        lines.append(f"{prefix}{node.node_type.value}: {node.title}")

        if node.description:
            desc_lines = node.description.split("\n")
            for dl in desc_lines[:3]:
                lines.append(f"{'  ' * (indent + 1)}{dl}")

        for child in node.children:
            lines.append(self.print_tree(child.node_id, indent + 1))

        return "\n".join(lines)

    def get_current_summary(self) -> str:
        """获取当前导航位置的文字摘要（供 Agent 使用）"""
        if not self._current_context:
            return "未开始导航"

        ctx = self._current_context
        current = self._node_lookup.get(ctx.current_node_id)

        summary = f"当前位置: {current.title if current else ctx.current_node_id}\n"
        summary += f"可见子节点: {len(ctx.visible_children)} 个\n"
        summary += f"建议下一步: {ctx.suggested_next}\n"
        summary += f"历史路径: {' → '.join(ctx.path_history[-5:])}"

        return summary

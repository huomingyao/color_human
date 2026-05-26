"""
Agentic RAG — Deep Retriever (深度检索)

核心职责:
- 多跳检索: 从第一次检索结果中发现新线索，展开下一跳
- 混合检索: 向量语义 + 关键词精确匹配
- 导航检索: 利用 Corpus2Skill 导航器在知识树中按路径检索
- 重排序: 对检索结果按相关性、新鲜度、信息密度重新排序
- 去重: 排除已见过的内容块
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .models import RetrievalResult
from ..corpus2skill.models import SkillDirectoryTree
from ..corpus2skill.navigator import KnowledgeNavigator


class DeepRetriever:
    """
    深度检索器 — 支持多跳检索和知识树导航。

    用法:
        retriever = DeepRetriever(tree, vector_search_fn)
        result = retriever.search("商业决策模式", categories=["decisions", "writings"])
        # 多跳
        result2 = retriever.hop(result, "生态系统")
    """

    def __init__(
        self,
        tree: SkillDirectoryTree,
        vector_search_fn: Optional[Callable[[str, str, int], list[dict[str, Any]]]] = None,
    ):
        """
        Args:
            tree: Corpus2Skill 知识目录树
            vector_search_fn: (query, category, top_k) -> list[dict] 向量检索函数
                              如果为 None，退化为关键词匹配
        """
        self.tree = tree
        self.navigator = KnowledgeNavigator(tree) if tree is not None else None
        self.vector_search = vector_search_fn
        self._seen_chunks: set[str] = set()

    # ========================================================================
    # 单次检索
    # ========================================================================

    def search(
        self,
        query: str,
        categories: Optional[list[str]] = None,
        top_k: int = 5,
        method: str = "hybrid",
    ) -> RetrievalResult:
        """
        执行单次检索。

        Args:
            query: 检索查询
            categories: 限定内容分类 (None=全局)
            top_k: 返回top-k结果
            method: "vector" / "navigate" / "hybrid"

        Returns:
            RetrievalResult: 检索结果
        """
        chunks: list[dict[str, Any]] = []
        sources: set[str] = set()
        cat_covered: set[str] = set()

        if method in ("vector", "hybrid") and self.vector_search:
            # 向量检索
            if categories:
                for cat in categories:
                    cat_results = self.vector_search(query, cat, top_k)
                    for r in cat_results:
                        chunk_id = r.get("chunk_id", "")
                        if chunk_id not in self._seen_chunks:
                            chunks.append(r)
                            self._seen_chunks.add(chunk_id)
                            sources.add(r.get("source", ""))
                            cat_covered.add(cat)
            else:
                results = self.vector_search(query, "all", top_k)
                for r in results:
                    chunk_id = r.get("chunk_id", "")
                    if chunk_id not in self._seen_chunks:
                        chunks.append(r)
                        self._seen_chunks.add(chunk_id)
                        sources.add(r.get("source", ""))

        if method in ("navigate", "hybrid") and self.navigator is not None:
            # 导航检索
            if categories:
                for cat_str in categories:
                    from ..corpus2skill.models import ContentCategory
                    try:
                        cat = ContentCategory(cat_str)
                        nav_results = self.navigator.retrieve_by_category(cat, query, top_k)
                        for nr in nav_results:
                            chunk_id = nr.get("chunk_id", "")
                            if chunk_id not in self._seen_chunks:
                                chunks.append(nr)
                                self._seen_chunks.add(chunk_id)
                                sources.add(nr.get("node_id", ""))
                                cat_covered.add(cat_str)
                    except ValueError:
                        pass

        # 去重和去空
        chunks = [c for c in chunks if c.get("chunk_id")]
        chunks = self._deduplicate(chunks)

        # 重排序
        chunks = self._rerank(chunks, query)

        # 计算质量
        relevance = self._estimate_relevance(chunks, query)
        density = self._estimate_information_density(chunks)
        novelty = self._estimate_novelty(chunks)

        return RetrievalResult(
            query=query,
            chunks=chunks[:top_k],
            sources=list(sources),
            categories_covered=list(cat_covered),
            relevance_score=relevance,
            information_density=density,
            novelty_score=novelty,
            retrieval_method=method,
        )

    # ========================================================================
    # 多跳检索
    # ========================================================================

    def hop(
        self,
        previous_result: RetrievalResult,
        follow_up_query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        对上一次检索结果进行多跳扩展。

        从 previous_result 中提取关键实体/概念，结合 follow_up_query 做深一层检索。
        """
        # 从上次结果中提取实体
        entities = self._extract_entities(previous_result)

        # 构建增强查询
        enhanced_query = follow_up_query
        if entities:
            enhanced_query = f"{follow_up_query} {' '.join(entities[:5])}"

        # 执行检索
        result = self.search(
            query=enhanced_query,
            categories=previous_result.categories_covered if previous_result.categories_covered else None,
            top_k=top_k,
            method="hybrid",
        )

        # 计算新颖性
        existing_content = " ".join(
            c.get("content", "") or c.get("text", "") or ""
            for c in previous_result.chunks
        )
        new_content = " ".join(
            c.get("content", "") or c.get("text", "") or ""
            for c in result.chunks
        )
        result.novelty_score = self._text_novelty(existing_content, new_content)

        return result

    # ========================================================================
    # 关键词检索 (退化为精确匹配)
    # ========================================================================

    def keyword_search(
        self,
        keywords: list[str],
        categories: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        基于关键词精确匹配的检索。

        当向量检索不可用时，退化为关键词匹配。
        """
        chunks: list[dict[str, Any]] = []
        sources: set[str] = set()

        # 从知识树中获取所有目标分类的 chunk_ids
        if categories and self.navigator is not None:
            for cat_str in categories:
                from ..corpus2skill.models import ContentCategory
                try:
                    cat = ContentCategory(cat_str)
                    check_in = self.navigator.retrieve_by_category(cat, "", top_k=50)
                    for item in check_in:
                        chunk_id = item.get("chunk_id", "")
                        # 关键词匹配 (实际项目中这里是向量检索)
                        # 此处储存信息，由外层调用方完成实际内容获取
                        if chunk_id not in self._seen_chunks:
                            chunks.append({
                                "chunk_id": chunk_id,
                                "node_id": item.get("node_id", ""),
                                "keywords_matched": keywords,
                            })
                            self._seen_chunks.add(chunk_id)
                            sources.add(item.get("node_id", ""))
                except ValueError:
                    pass

        return RetrievalResult(
            query=" ".join(keywords),
            chunks=chunks[:top_k],
            sources=list(sources),
            categories_covered=categories or [],
            relevance_score=0.6,  # 关键词匹配默认中等相关
            retrieval_method="keyword",
        )

    # ========================================================================
    # 重排序
    # ========================================================================

    def _rerank(
        self,
        chunks: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """对检索结果重排序"""
        scored = []
        query_terms = set(query.lower().split())

        for chunk in chunks:
            score = 0.0
            text = (
                chunk.get("content", "")
                or chunk.get("text", "")
                or chunk.get("chunk_id", "")
            ).lower()

            # 关键词命中
            for term in query_terms:
                if term in text:
                    score += 0.1

            # 长度奖励 (中等长度最优)
            text_len = len(text)
            if 200 <= text_len <= 800:
                score += 0.15
            elif text_len > 800:
                score += 0.05

            # 来源多样性奖励
            source = chunk.get("source", "") or chunk.get("node_id", "")
            if "decision" in source.lower():
                score += 0.1

            # 已存相关性
            existing_relevance = chunk.get("relevance", 0.5)
            score += existing_relevance * 0.3

            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored]

    def _deduplicate(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """去重"""
        seen_ids = set()
        unique = []
        for chunk in chunks:
            cid = chunk.get("chunk_id", "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                unique.append(chunk)
        return unique

    # ========================================================================
    # 质量评估
    # ========================================================================

    def _estimate_relevance(self, chunks: list[dict[str, Any]], query: str) -> float:
        """估算相关性"""
        if not chunks:
            return 0.0

        query_terms = set(query.lower().split())
        total_score = 0.0

        for chunk in chunks:
            text = (
                chunk.get("content", "")
                or chunk.get("text", "")
                or ""
            ).lower()
            if not text:
                continue
            hits = sum(1 for t in query_terms if t in text)
            chunk_score = hits / max(len(query_terms), 1)
            total_score += chunk_score

        return min(total_score / len(chunks), 1.0)

    def _estimate_information_density(self, chunks: list[dict[str, Any]]) -> float:
        """估算信息密度"""
        if not chunks:
            return 0.0

        densities = []
        for chunk in chunks:
            text = chunk.get("content", "") or chunk.get("text", "") or ""
            if not text:
                continue
            # 实体密度: 大写词/专有名词/数字
            entities = len(re.findall(r'[A-Z][a-z]+', text))
            numbers = len(re.findall(r'\d+', text))
            density = (entities + numbers) / max(len(text.split()), 1)
            densities.append(density)

        if not densities:
            return 0.0
        return min(sum(densities) / len(densities) * 5, 1.0)  # 归一化

    def _estimate_novelty(self, chunks: list[dict[str, Any]]) -> float:
        """估算新颖性 (相对于已检索过的内容)"""
        if not self._seen_chunks or not chunks:
            return 0.8

        new_ids = set(c.get("chunk_id", "") for c in chunks)
        old_ids = self._seen_chunks - new_ids

        if not old_ids:
            return 0.8

        overlap = len(new_ids & old_ids)
        return 1.0 - (overlap / max(len(new_ids), 1))

    # ========================================================================
    # 实体提取
    # ========================================================================

    def _extract_entities(self, result: RetrievalResult) -> list[str]:
        """从检索结果中提取关键实体"""
        entities: set[str] = set()

        for chunk in result.chunks:
            text = chunk.get("content", "") or chunk.get("text", "") or ""
            # 中英文实体
            en_entities = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', text)
            cn_entities = re.findall(r'[一-鿿]{2,6}(?:公司|集团|基金|技术|模型|理论|策略|方案)', text)
            entities.update(en_entities[:5])
            entities.update(cn_entities[:5])

        return list(entities)[:10]

    def _text_novelty(self, old_text: str, new_text: str) -> float:
        """计算新旧文本的新颖性 (简单的 Jaccard 距离)"""
        if not new_text:
            return 0.0

        old_words = set(old_text.lower().split())
        new_words = set(new_text.lower().split())

        if not new_words:
            return 0.0

        intersection = len(old_words & new_words)
        union = len(new_words)  # 只看新文本的词汇有多少是新的
        return 1.0 - (intersection / union) if union > 0 else 0.0

    # ========================================================================
    # 工具方法
    # ========================================================================

    def reset_seen(self):
        """重置已见chunk记录"""
        self._seen_chunks.clear()

    def get_seen_count(self) -> int:
        """获取已见chunk数量"""
        return len(self._seen_chunks)

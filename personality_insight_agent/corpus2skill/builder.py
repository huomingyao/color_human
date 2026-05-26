"""
Corpus2Skill — 离线流水线

将用户提供的私有资料目录自动处理为层级知识目录树。

处理流程:
    1. 扫描源目录 → 识别文件类型和内容分类
    2. 对每个文件做 chunk → embedding → FAISS 向量存储
    3. 调用 LLM 对内容进行语义分类 + 摘要生成
    4. 构建 INDEX.md → 聚合为层级树 → 根节点 SKILL.md

使用方式:
    builder = CorpusTreeBuilder(vector_store_root="d:/vectors/")
    tree = builder.build(
        llm_callable=my_llm,
        source_dir="d:/my_private_docs/",
        person_id="zhangsan",
    )
    builder.save_tree(tree, "d:/output/")
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import (
    ContentCategory,
    CorpusIndex,
    EvidenceLevel,
    KnowledgeTreeNode,
    NodeType,
    SkillDirectoryTree,
)


class CorpusTreeBuilder:
    """
    离线流水线 — 批量文档 → 层级知识树。

    职责:
    - 文件发现和分类
    - 内容 chunk 和向量化 (委托给外部向量库)
    - 调用 LLM 做语义摘要和导航提示生成
    - 组装层级树结构
    """

    def __init__(self, vector_store_root: str = "personality_insight_agent/vectors/"):
        self.vector_store_root = vector_store_root
        os.makedirs(vector_store_root, exist_ok=True)

        # 文件类型映射
        self.EXT_CATEGORY_HINTS: dict[str, ContentCategory] = {
            ".txt": ContentCategory.GENERIC,
            ".md": ContentCategory.GENERIC,
            ".pdf": ContentCategory.GENERIC,
            ".docx": ContentCategory.GENERIC,
            ".json": ContentCategory.GENERIC,
        }

        # 文件名模式 → ContentCategory 映射（中文优先）
        self.FILENAME_CATEGORY_PATTERNS: list[tuple[str, ContentCategory]] = [
            # WRITINGS: 著作/书籍/文章/论文
            (r"(书|book|著作|出版|paper|论文|essay|文章|blog|newsletter|读后感|笔记|总结|反思|日记)", ContentCategory.WRITINGS),
            # CONVERSATIONS: 对话/访谈/会议/聊天记录
            (r"(对话|interview|podcast|播客|访谈|聊天|chat|conversation|会议|transcript|字幕|问答|采访)", ContentCategory.CONVERSATIONS),
            # EXPRESSIONS: 社交媒体/碎碎念
            (r"(twitter|微博|即刻|社交媒体|social|动态|post|tweet|说说|朋友圈|碎碎念)", ContentCategory.EXPRESSIONS),
            # EXTERNAL: 他者视角
            (r"(评价|批评|review|评论|他者|外部|传记|biography|别人|观察|印象)", ContentCategory.EXTERNAL),
            # DECISIONS: 决策/规则
            (r"(决策|决定|decision|memorandum|备忘录|纪要|战略|投资|规则|班规|制度|方案)", ContentCategory.DECISIONS),
            # TIMELINE: 时间线/履历
            (r"(时间线|timeline|履历|简历|年表|历程|时间轴)", ContentCategory.TIMELINE),
        ]

        # 目录名模式 → ContentCategory 映射（中文优先，更精确）
        self.DIRNAME_CATEGORY_PATTERNS: list[tuple[str, ContentCategory]] = [
            # WRITINGS: 书籍/文章/写作
            (r"(著作|书籍|book|writing|文章|article|阅读|泛读|文学|人类学|心理学|小说)", ContentCategory.WRITINGS),
            # CONVERSATIONS: 对话/访谈/会议
            (r"(对话|访谈|interview|podcast|聊天|会议|Transcript|问答)", ContentCategory.CONVERSATIONS),
            # EXPRESSIONS: 表达/社交碎片/作业
            (r"(表达|社交|social|twitter|微博|碎碎念|说说|作业|笔记)", ContentCategory.EXPRESSIONS),
            # EXTERNAL: 他者视角
            (r"(评价|外部|他者|批评|观察)", ContentCategory.EXTERNAL),
            # DECISIONS: 决策/规则/班规/活动/事务
            (r"(决策|战略|决定|decision|纪要|规则|班规|活动|事务|职务)", ContentCategory.DECISIONS),
            # TIMELINE: 明确的时间线
            (r"(时间线|timeline|年表|履历)", ContentCategory.TIMELINE),
        ]

    # ========================================================================
    # 主入口
    # ========================================================================

    def build(
        self,
        llm_callable: Callable[[str, str], str],
        source_dir: str,
        person_id: str,
        max_chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> SkillDirectoryTree:
        """
        执行完整的离线流水线。

        Args:
            llm_callable: LLM 调用函数 (system_prompt, user_prompt) -> response
            source_dir: 用户私有资料目录路径
            person_id: 人物标识
            max_chunk_size: 文本切分大小
            chunk_overlap: 切分重叠

        Returns:
            SkillDirectoryTree: 完整的层级知识目录树
        """
        # Step 1: 扫描文件
        files = self._scan_directory(source_dir)

        # Step 2: 对文件分类
        classified = self._classify_files(files)

        # Step 3: 对每个文件做 chunk (委托外部)
        chunked = self._chunk_files(classified, max_chunk_size, chunk_overlap)

        # Step 4: 生成每个文件的摘要 (LLM)
        summaries = self._summarize_files(llm_callable, chunked)

        # Step 5: 按 ContentCategory 分组，构建 INDEX 层
        indices = self._build_indices(llm_callable, summaries)

        # Step 6: 组装叶子节点
        leaf_nodes = self._build_leaf_nodes(chunked, summaries)

        # Step 7: 构建中间节点
        middle_nodes = self._build_middle_nodes(indices, leaf_nodes)

        # Step 8: 构建根节点
        root = self._build_root_node(llm_callable, person_id, middle_nodes, indices)

        # Step 9: 组装完整树
        doc_registry: dict[str, KnowledgeTreeNode] = {}
        for leaf in leaf_nodes:
            doc_registry[leaf.node_id] = leaf

        tree = SkillDirectoryTree(
            tree_id=person_id,
            root=root,
            index_registry={idx.index_id: idx for idx in indices},
            doc_registry=doc_registry,
            total_documents=len(leaf_nodes),
            total_words=sum(leaf.estimated_words for leaf in leaf_nodes),
            source_directory=source_dir,
            vector_store_map=self._get_vector_store_map(person_id),
        )

        return tree

    # ========================================================================
    # Step 1: 目录扫描
    # ========================================================================

    def _scan_directory(self, source_dir: str) -> list[dict[str, Any]]:
        """递归扫描目录，返回文件列表"""
        files = []
        if not os.path.exists(source_dir):
            return files

        for root, _, filenames in os.walk(source_dir):
            for fname in filenames:
                # 跳过隐藏文件
                if fname.startswith("."):
                    continue

                ext = os.path.splitext(fname)[1].lower()
                if ext not in (".txt", ".md", ".pdf", ".docx", ".json", ".csv", ".html"):
                    continue

                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, source_dir)

                try:
                    file_size = os.path.getsize(full_path)
                except OSError:
                    file_size = 0

                files.append({
                    "filename": fname,
                    "rel_path": rel_path,
                    "full_path": full_path,
                    "ext": ext,
                    "size": file_size,
                    "parent_dir": os.path.basename(os.path.dirname(full_path)),
                })

        return sorted(files, key=lambda f: f["rel_path"])

    # ========================================================================
    # Step 2: 文件分类
    # ========================================================================

    def _classify_files(self, files: list[dict[str, Any]]) -> dict[ContentCategory, list[dict[str, Any]]]:
        """按 ContentCategory 分类文件"""
        classified: dict[ContentCategory, list[dict[str, Any]]] = {cat: [] for cat in ContentCategory}

        for f in files:
            category = self._infer_category(f)
            f["inferred_category"] = category
            classified[category].append(f)

        return classified

    def _infer_category(self, file_info: dict[str, Any]) -> ContentCategory:
        """
        推断文件的 ContentCategory。

        优先级: 目录名 > 文件名 > 父目录名
        """
        fname = file_info["filename"].lower()
        parent_dir = file_info.get("parent_dir", "").lower()
        rel_path = file_info.get("rel_path", "").lower()

        # 1. 检查文件名
        for pattern, category in self.FILENAME_CATEGORY_PATTERNS:
            if re.search(pattern, fname):
                return category

        # 2. 检查目录名
        for pattern, category in self.DIRNAME_CATEGORY_PATTERNS:
            if re.search(pattern, parent_dir):
                return category

        # 3. 检查路径
        path_parts = rel_path.replace("\\", "/").split("/")
        for part in path_parts:
            for pattern, category in self.DIRNAME_CATEGORY_PATTERNS:
                if re.search(pattern, part):
                    return category

        return ContentCategory.GENERIC

    # ========================================================================
    # Step 3: Chunk 文件
    # ========================================================================

    def _chunk_files(
        self,
        classified: dict[ContentCategory, list[dict[str, Any]]],
        chunk_size: int,
        overlap: int,
    ) -> list[dict[str, Any]]:
        """对所有文件做文本切分"""
        chunked = []

        for category, files in classified.items():
            for f in files:
                text = self._read_file_content(f["full_path"])
                if not text:
                    continue

                chunks = self._split_text(text, chunk_size, overlap)
                f["chunks"] = chunks
                f["word_count"] = self._count_words(text)
                f["chunk_count"] = len(chunks)
                chunked.append(f)

        return chunked

    def _read_file_content(self, file_path: str) -> str:
        """读取文件内容"""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                    return text
                except ImportError:
                    return f"[PDF需要PyMuPDF库: {file_path}]"
                except Exception:
                    return ""
            elif ext == ".docx":
                try:
                    import docx
                    doc = docx.Document(file_path)
                    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                except ImportError:
                    return f"[DOCX需要python-docx库: {file_path}]"
                except Exception:
                    return ""
            else:
                # txt, md, json, csv, html
                for encoding in ("utf-8", "gbk", "latin-1"):
                    try:
                        with open(file_path, "r", encoding=encoding) as fh:
                            return fh.read()
                    except UnicodeDecodeError:
                        continue
                return ""
        except Exception:
            return ""

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """简单文本切分 (按段落/句子边界优先)"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        # 按段落切
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 处理过长的段落
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size * 2:
                # 按句子切
                sentences = re.split(r'(?<=[。！？.!?])\s*', chunk)
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) <= chunk_size:
                        sub += sent
                    else:
                        if sub:
                            final_chunks.append(sub.strip())
                        sub = sent
                if sub.strip():
                    final_chunks.append(sub.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _count_words(self, text: str) -> int:
        """中英文混合字数统计"""
        chinese = len(re.findall(r'[一-鿿]', text))
        english = len(re.findall(r'[a-zA-Z]+', text))
        return chinese + english

    # ========================================================================
    # Step 4: 文件摘要 (LLM)
    # ========================================================================

    def _summarize_files(
        self,
        llm_callable: Callable[[str, str], str],
        chunked_files: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """为每个文件生成 LLM 摘要"""
        summaries: dict[str, dict[str, Any]] = {}

        for f in chunked_files:
            rel_path = f["rel_path"]
            # 取前 2000 字作为摘要输入
            sample_text = f["chunks"][0][:2000] if f["chunks"] else ""

            if not sample_text.strip():
                summaries[rel_path] = {
                    "title": f["filename"],
                    "summary": "(空文件)",
                    "key_topics": [],
                    "key_entities": [],
                    "category": f["inferred_category"].value,
                }
                continue

            try:
                system_prompt = _SUMMARY_SYSTEM_PROMPT
                user_prompt = f"""文件名: {f['filename']}
推断分类: {f['inferred_category'].value}
字数估算: {f.get('word_count', 0)}

内容样本:
---
{sample_text[:2000]}
---

请分析并返回JSON。"""

                resp = llm_callable(system_prompt, user_prompt)
                parsed = self._parse_json_response(resp)

                summaries[rel_path] = {
                    "title": parsed.get("title", f["filename"]),
                    "summary": parsed.get("summary", "(无法解析)"),
                    "key_topics": parsed.get("key_topics", []),
                    "key_entities": parsed.get("key_entities", []),
                    "category": f["inferred_category"].value,
                    "evidence_level": parsed.get("evidence_level", "primary"),
                }
            except Exception:
                summaries[rel_path] = {
                    "title": f["filename"],
                    "summary": sample_text[:200] + "...",
                    "key_topics": [],
                    "key_entities": [],
                    "category": f["inferred_category"].value,
                    "evidence_level": "primary",
                }

        return summaries

    @staticmethod
    def _parse_json_response(resp: str) -> dict[str, Any]:
        """从 LLM 响应中提取 JSON"""
        # 尝试找到 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', resp)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}

    # ========================================================================
    # Step 5: 构建 INDEX 层
    # ========================================================================

    def _build_indices(
        self,
        llm_callable: Callable[[str, str], str],
        summaries: dict[str, dict[str, Any]],
    ) -> list[CorpusIndex]:
        """为每个 ContentCategory 构建 INDEX.md"""
        indices: list[CorpusIndex] = []

        for category in ContentCategory:
            # 收集该分类下的所有文件
            cat_files = {
                path: info
                for path, info in summaries.items()
                if info.get("category") == category.value
            }

            if not cat_files:
                continue

            # 构建子节点概览
            children_overview = []
            for path, info in cat_files.items():
                children_overview.append({
                    "id": path.replace("\\", "/"),
                    "title": info["title"],
                    "summary": info["summary"][:200],
                    "word_count": str(info.get("word_count", "N/A")),
                })

            # 生成导航提示
            nav_hints = self._generate_nav_hints(llm_callable, category, children_overview)

            total_words = sum(
                info.get("word_count", 0) if isinstance(info.get("word_count"), int) else 0
                for info in cat_files.values()
            )

            index = CorpusIndex(
                index_id=f"index_{category.value}",
                title=self._category_display_name(category),
                total_children=len(cat_files),
                total_words=total_words,
                category_distribution={category.value: len(cat_files)},
                navigation_hints=nav_hints,
                children_overview=children_overview,
                cross_references=self._build_cross_refs(cat_files),
            )
            indices.append(index)

        return indices

    def _generate_nav_hints(
        self,
        llm_callable: Callable[[str, str], str],
        category: ContentCategory,
        children: list[dict[str, str]],
    ) -> list[str]:
        """生成导航提示"""
        # 规则生成 + LLM 增强
        hints = []

        # 规则层面
        if category == ContentCategory.DECISIONS:
            hints.append("此分类包含重大决策记录，适合分析决策模式和风险偏好")
        elif category == ContentCategory.WRITINGS:
            hints.append("此分类包含系统性思考，适合提取核心心智模型")
        elif category == ContentCategory.CONVERSATIONS:
            hints.append("此分类包含即兴对话，适合分析语言风格和临场反应")
        elif category == ContentCategory.EXPRESSIONS:
            hints.append("此分类包含碎片表达，适合提取句式指纹和风格DNA")

        if len(children) >= 5:
            hints.append(f"共 {len(children)} 个文档，建议按主题相关性分批阅读")

        # LLM 增强 (可选，对大量文件做聚类)
        if len(children) > 5 and llm_callable:
            try:
                titles = [c["title"] for c in children[:10]]
                prompt = f"""以下是 '{self._category_display_name(category)}' 分类下的文档:
{chr(10).join(f'- {t}' for t in titles)}

请给出2-3条导航建议，帮助Agent在分析人物时决定先看哪些文档。直接返回建议列表，用换行分隔。"""
                resp = llm_callable(
                    "你是知识库导航设计师。给出简洁有效的导航建议。",
                    prompt,
                )
                llm_hints = [h.strip("- ") for h in resp.strip().split("\n") if h.strip()]
                hints.extend(llm_hints[:3])
            except Exception:
                pass

        return hints

    def _build_cross_refs(
        self,
        cat_files: dict[str, dict[str, Any]],
    ) -> dict[str, list[str]]:
        """构建交叉引用: 主题 → 相关文档"""
        cross_refs: dict[str, list[str]] = {}
        for path, info in cat_files.items():
            for topic in info.get("key_topics", []):
                if topic not in cross_refs:
                    cross_refs[topic] = []
                cross_refs[topic].append(path)
        return cross_refs

    # ========================================================================
    # Step 6-8: 构建层级节点
    # ========================================================================

    def _build_leaf_nodes(
        self,
        chunked: list[dict[str, Any]],
        summaries: dict[str, dict[str, Any]],
    ) -> list[KnowledgeTreeNode]:
        """构建叶子节点"""
        leaves = []

        for f in chunked:
            summary_info = summaries.get(f["rel_path"], {})
            node_id = f["rel_path"].replace("\\", "/")

            chunk_ids = []
            for i in range(len(f.get("chunks", []))):
                chunk_ids.append(f"{node_id}#chunk_{i}")

            leaf = KnowledgeTreeNode(
                node_id=node_id,
                node_type=NodeType.LEAF,
                title=summary_info.get("title", f["filename"]),
                description=summary_info.get("summary", "")[:300],
                parent_id=f"index_{f['inferred_category'].value}",
                category=f.get("inferred_category", ContentCategory.GENERIC),
                evidence_level=EvidenceLevel(summary_info.get("evidence_level", "primary")),
                source_path=f["full_path"],
                chunk_ids=chunk_ids,
                estimated_words=f.get("word_count", 0),
                key_topics=summary_info.get("key_topics", []),
                key_entities=summary_info.get("key_entities", []),
            )
            leaves.append(leaf)

        return leaves

    def _build_middle_nodes(
        self,
        indices: list[CorpusIndex],
        leaf_nodes: list[KnowledgeTreeNode],
    ) -> list[KnowledgeTreeNode]:
        """构建中间索引节点"""
        middle_nodes = []

        # 构建 leaf 查找表
        leaves_by_parent: dict[str, list[KnowledgeTreeNode]] = {}
        for leaf in leaf_nodes:
            parent = leaf.parent_id or ""
            if parent not in leaves_by_parent:
                leaves_by_parent[parent] = []
            leaves_by_parent[parent].append(leaf)

        for idx in indices:
            children = leaves_by_parent.get(idx.index_id, [])
            child_nodes: list[KnowledgeTreeNode] = []
            for leaf in children:
                # 浅拷贝叶子节点，不附加孙子
                child_nodes.append(KnowledgeTreeNode(
                    node_id=leaf.node_id,
                    node_type=NodeType.LEAF,
                    title=leaf.title,
                    description=leaf.description,
                    parent_id=idx.index_id,
                    category=leaf.category,
                    evidence_level=leaf.evidence_level,
                    source_path=leaf.source_path,
                    chunk_ids=leaf.chunk_ids,
                    estimated_words=leaf.estimated_words,
                ))

            middle = KnowledgeTreeNode(
                node_id=idx.index_id,
                node_type=NodeType.INDEX,
                title=idx.title,
                description=f"{idx.total_children} 个文档, {idx.total_words} 字",
                parent_id="root",
                children=child_nodes,
                category=self._index_id_to_category(idx.index_id),
                summary=idx.navigation_hints[0] if idx.navigation_hints else "",
                key_topics=list(idx.cross_references.keys()),
                total_words_estimate=idx.total_words,
            )
            middle_nodes.append(middle)

        return middle_nodes

    def _build_root_node(
        self,
        llm_callable: Callable[[str, str], str],
        person_id: str,
        middle_nodes: list[KnowledgeTreeNode],
        indices: list[CorpusIndex],
    ) -> KnowledgeTreeNode:
        """构建根节点 SKILL.md"""
        total_docs = sum(idx.total_children for idx in indices)
        total_words = sum(idx.total_words for idx in indices)

        # 从各分类中提取关键实体
        all_entities: list[str] = []
        for idx in indices:
            for child in idx.children_overview:
                pass  # entities are in leaf nodes
        for node in middle_nodes:
            for child in node.children:
                all_entities.extend(child.key_entities)
        unique_entities = list(set(all_entities))[:20]

        # 生成根节点描述
        category_summary = "\n".join(
            f"- {idx.title}: {idx.total_children} 个文档, {idx.total_words} 字"
            for idx in indices
        )

        description = (
            f"## {person_id} 的知识档案\n\n"
            f"共计 {total_docs} 个文档, {total_words} 字\n\n"
            f"### 内容分布\n{category_summary}"
        )

        root = KnowledgeTreeNode(
            node_id="root",
            node_type=NodeType.ROOT,
            title=f"{person_id} 知识目录树",
            description=description,
            children=middle_nodes,
            category=ContentCategory.GENERIC,
            key_topics=list(set(
                topic
                for idx in indices
                for topic in list(idx.cross_references.keys())[:5]
            ))[:10],
            key_entities=unique_entities,
        )

        return root

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _get_vector_store_map(self, person_id: str) -> dict[str, str]:
        """获取向量库路径映射"""
        return {
            cat.value: os.path.join(self.vector_store_root, person_id, cat.value)
            for cat in ContentCategory
        }

    def save_tree(self, tree: SkillDirectoryTree, output_dir: str):
        """将知识树序列化到磁盘"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{tree.tree_id}_corpus_tree.json")

        # 简化序列化 (递归转 dict)
        tree_dict = self._tree_to_dict(tree)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tree_dict, f, ensure_ascii=False, indent=2)

        print(f"[Corpus2Skill] 知识树已保存到: {output_path}")

    def load_tree(self, file_path: str) -> SkillDirectoryTree:
        """从磁盘加载知识树"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self._dict_to_tree(data)

    def _tree_to_dict(self, tree: SkillDirectoryTree) -> dict[str, Any]:
        """递归序列化"""
        def node_to_dict(node: KnowledgeTreeNode) -> dict[str, Any]:
            d = node.model_dump()
            d["children"] = [node_to_dict(c) for c in node.children]
            return d

        return {
            "tree_id": tree.tree_id,
            "root": node_to_dict(tree.root),
            "total_documents": tree.total_documents,
            "total_words": tree.total_words,
            "build_time": tree.build_time,
            "source_directory": tree.source_directory,
            "vector_store_map": tree.vector_store_map,
        }

    def _dict_to_tree(self, data: dict[str, Any]) -> SkillDirectoryTree:
        """递归反序列化"""
        def dict_to_node(d: dict[str, Any]) -> KnowledgeTreeNode:
            children = [dict_to_node(c) for c in d.pop("children", [])]
            node = KnowledgeTreeNode(**d)
            node.children = children
            return node

        root = dict_to_node(data["root"])
        return SkillDirectoryTree(
            tree_id=data["tree_id"],
            root=root,
            total_documents=data.get("total_documents", 0),
            total_words=data.get("total_words", 0),
            build_time=data.get("build_time", ""),
            source_directory=data.get("source_directory", ""),
            vector_store_map=data.get("vector_store_map", {}),
        )

    @staticmethod
    def _category_display_name(category: ContentCategory) -> str:
        names = {
            ContentCategory.WRITINGS: "著作与系统思考",
            ContentCategory.CONVERSATIONS: "对话与即兴思考",
            ContentCategory.EXPRESSIONS: "碎片表达与风格DNA",
            ContentCategory.EXTERNAL: "他者视角与批评",
            ContentCategory.DECISIONS: "决策记录与行动",
            ContentCategory.TIMELINE: "人物时间线",
            ContentCategory.GENERIC: "通用资料",
        }
        return names.get(category, "通用资料")

    @staticmethod
    def _index_id_to_category(index_id: str) -> ContentCategory:
        mapping = {
            "index_writings": ContentCategory.WRITINGS,
            "index_conversations": ContentCategory.CONVERSATIONS,
            "index_expressions": ContentCategory.EXPRESSIONS,
            "index_external_views": ContentCategory.EXTERNAL,
            "index_decisions": ContentCategory.DECISIONS,
            "index_timeline": ContentCategory.TIMELINE,
            "index_generic": ContentCategory.GENERIC,
        }
        return mapping.get(index_id, ContentCategory.GENERIC)


# ============================================================================
# LLM Prompts
# ============================================================================

_SUMMARY_SYSTEM_PROMPT = """你是一个知识库分析专家。你的任务是为给定文档生成结构化摘要。

请以 JSON 格式输出:
{
    "title": "文档标题（如果从内容中能推断）或文件名",
    "summary": "200字以内的内容摘要",
    "key_topics": ["主题1", "主题2", "主题3"],
    "key_entities": ["实体1", "实体2"],
    "evidence_level": "primary/secondary/tertiary"
}

规则:
- evidence_level: 如果是本人写的/说的一手材料 → primary，他人转述评论 → secondary，数据统计/AI推断 → tertiary
- key_topics 只输出该文档核心涵盖的主题，不超过5个
- key_entities 输出文中关键的人名、机构名、术语

只输出JSON，不要额外文本。"""

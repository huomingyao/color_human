# Person Fenxi - 项目索引

> 当前工作目录: `/d/person_fenxi/`
> 组织方式: 拆分为独立项目

---

## 📦 独立项目清单

### 1. person_insight_agent/

**路径**: `personality_insight_agent/`

**描述**: 基于心理学框架的人物画像分析Agent，借鉴Nuwa-Skill架构

**核心功能**:
- 多Skill协同 Pipeline（Skill1-5）
- ReAct Agent 推理
- Agentic RAG 知识检索
- 向量化知识库

**入口**: `main.py`

**状态**: 已有 `pyproject.toml`，可独立安装

**依赖**: pydantic>=2.0, requests>=2.28

---

### 2. honcho/

**路径**: `honcho/`

**描述**: FastAPI Web框架，完整的CRUD + 嵌入向量服务

**核心功能**:
- RESTful API
- 向量嵌入（embedding）
- Cache / Dreamer / Reconciler 模块
- Docker 部署支持

**状态**: 完全独立项目，已有完整 CI/CD 配置

---

### 3. nuwa-skill/

**路径**: `nuwa-skill/`

**描述**: Nuwa-Skill 知识文档（纯文档仓库）

**内容**:
- SKILL.md - 核心Skill定义
- README.md - 多语言版（中英日韩西）
- assets/ - 图片资源

**状态**: 文档项目，无需代码配置

---

### 4. tools/ （小工具合集）

**路径**: `tools/`

**描述**: 根目录脚本整理后的集合

#### 4.1 tools/extractors/

**功能**: 文本提取器

| 脚本 | 用途 |
|------|------|
| extract_caiyanjun.py | 从 docx 提取蔡岩峻资料 |
| extract_apply_letter.py | 提取申请信 |
| extract_full_corpus.py | 提取完整语料库 |

#### 4.2 tools/pipelines/

**功能**: 分析Pipeline

| 脚本 | 用途 |
|------|------|
| full_pipeline.py | 主Pipeline，调用LLM做人物画像分析 |
| run_analysis.py | 运行分析入口 |

#### 4.3 tools/kb/

**功能**: 知识库构建

| 脚本 | 用途 |
|------|------|
| build_kb_minimax.py | 使用MiniMax API构建知识库 |

---

## 🚀 快速开始

### 独立使用 person_insight_agent

```bash
cd personality_insight_agent
pip install -e ".[dev]"
python -m person_insight_agent.main
```

### 启动 honcho

```bash
cd honcho
docker-compose up -d
```

### 运行工具脚本

```bash
# 设置环境变量
export DASHSCOPE_API_KEY="your-key"

# 运行分析
cd tools/pipelines
python run_analysis.py
```

---

## 📝 历史文档

| 文件 | 描述 |
|------|------|
| personality-insight-agent-design.md | Agent设计文档 |
| worklog-2026-05-23.md | 2026-05-23 工作日志 |

---

*Last Updated: 2026-05-26*
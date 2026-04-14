# Multi-Modal Agentic RAG System

本项目实现了一个基于 Agent 架构的多模态检索增强生成（RAG）系统，专为复杂文档（包含图表、数学公式及密集文本的 PDF 论文）的深度解析与问答设计。

## 🌟 核心特性 (Features)

* **多模态解析引擎**: 突破传统纯文本 RAG 限制，精准抽取并理解 PDF 中的视觉信息及复杂布局。
* **Agentic 工作流**: 引入大模型智能体路由，根据用户 query 意图自动决定检索策略（向量检索、图文联合检索或直接回答）。
* **交互式前端**: 提供流畅的对话式交互界面（基于 Gradio），支持多轮对话与检索上下文溯源。

## 🛠️ 技术栈 (Tech Stack)

* **核心逻辑**: Python 3.10+
* **大语言模型**: [Gemini 3 Pro Flash]
* **Agent 框架**: [LangGraph]
* **向量数据库**: [Qdrant]
* **前端交互**: Gradio

## 🚀 快速开始 (Quick Start)

### 环境安装
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt
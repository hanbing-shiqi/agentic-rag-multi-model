import json
import os
from datasets import Dataset
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas import evaluate

# 1. 配置你的中转 API (和之前一样)
os.environ["OPENAI_API_KEY"] = "sk-X9rheSBPgPaGCVK9NYCPYkoe2V2DoZ4XZcoImxsgYAAPLB5f"
os.environ["OPENAI_API_BASE"] = "https://chatapi.zjt66.top/v1"

judge_llm = ChatOpenAI(model="gemini-3-pro-preview", temperature=0.0)
# judge_embeddings = OpenAIEmbeddings(
#     model="text-embedding-ada-002", 
#     check_embedding_ctx_length=False
# )
print("🧠 正在加载本地 Embedding 模型...")
judge_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 2. 读取数据飞轮自动生成的数据
questions, answers, contexts_list = [], [], []

print("📂 正在读取飞轮数据集...")
with open("rag_auto_dataset_2.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        questions.append(item["question"])
        answers.append(item["answer"])
        contexts_list.append(item["contexts"])

# 构建无 Ground Truth 的数据集
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts_list
})

print(f"⚖️ 开始评估 {len(questions)} 条数据 (仅评估防幻觉和相关度)...")

# 3. 运行评估（只放入不需要 GT 的指标）
result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(), 
        AnswerRelevancy()
    ],
    llm=judge_llm,
    embeddings=judge_embeddings
)

print("\n========== 🌟 最终评估报告 ==========")
print(result)
import os
import google.generativeai as genai
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv('/Users/shihb/Documents/Agentic-Poject/agentic-rag-for-dummies/project/.env') 

# 获取 API Key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("你可以使用的 Gemini 模型有：")
for m in genai.list_models():
    # 筛选出支持文本/对话生成的模型
    if 'generateContent' in m.supported_generation_methods:
        print(m.name.replace('models/', ''))
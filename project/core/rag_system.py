import uuid
# from langchain_ollama import ChatOllama
# from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability

class RAGSystem:

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.observability = Observability()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        # llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
        # llm = ChatGoogleGenerativeAI(
        #     model=config.LLM_MODEL, 
        #     temperature=config.LLM_TEMPERATURE
        # )
        # llm = ChatOpenAI(
        #     api_key=os.getenv("GOOGLE_API_KEY"),
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     model=config.LLM_MODEL,
        #     temperature=config.LLM_TEMPERATURE
        # )
        API_KEY = os.getenv("OPENAI_API_KEY", "sk-X9rheSBPgPaGCVK9NYCPYkoe2V2DoZ4XZcoImxsgYAAPLB5f")
        API_BASE = os.getenv("OPENAI_API_BASE", "https://chatapi.zjt66.top/v1")
        
        llm = ChatOpenAI(
            api_key=API_KEY,
            base_url=API_BASE,
            model=config.LLM_MODEL, 
            temperature=config.LLM_TEMPERATURE
        )
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)



    def get_config(self):
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())
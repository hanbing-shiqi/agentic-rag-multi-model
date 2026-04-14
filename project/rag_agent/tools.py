from typing import List
from langchain_core.tools import tool
from db.parent_store_manager import ParentStoreManager
import base64
import os
from typing import List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import config

class ToolFactory:
    
    def __init__(self, collection):
        self.collection = collection
        self.parent_store_manager = ParentStoreManager()
    
    def _search_child_chunks(self, query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            results = self.collection.similarity_search(query, k=limit, score_threshold=0.7)
            if not results:
                return "NO_RELEVANT_CHUNKS"

            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])            

        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_many_parent_chunks(self, parent_ids: List[str]) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_ids: List of parent chunk IDs to retrieve
        """
        try:
            ids = [parent_ids] if isinstance(parent_ids, str) else list(parent_ids)
            raw_parents = self.parent_store_manager.load_content_many(ids)
            if not raw_parents:
                return "NO_PARENT_DOCUMENTS"

            return "\n\n".join([
                f"Parent ID: {doc.get('parent_id', 'n/a')}\n"
                f"File Name: {doc.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {doc.get('content', '').strip()}"
                for doc in raw_parents
            ])            

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _retrieve_parent_chunks(self, parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.
    
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        try:
            parent = self.parent_store_manager.load_content(parent_id)
            if not parent:
                return "NO_PARENT_DOCUMENT"

            return (
                f"Parent ID: {parent.get('parent_id', 'n/a')}\n"
                f"File Name: {parent.get('metadata', {}).get('source', 'unknown')}\n"
                f"Content: {parent.get('content', '').strip()}"
            )          

        except Exception as e:
            return f"PARENT_RETRIEVAL_ERROR: {str(e)}"
    
    def _analyze_image(self, image_path: str, specific_question: str) -> str:
        """Analyze an image/chart from the document to answer specific data questions.
        Use this tool when the retrieved text contains an image reference like ![Image](path) and you need to read the data in it.
        
        Args:
            image_path: The path to the image extracted from the markdown text
            specific_question: The highly specific question about the image data (e.g. "What is the 60th percentile value?")
        """
        try:
            # 路径安全处理，确保能找到刚才 utils.py 中提取的图片
            if not os.path.exists(image_path):
                # 尝试结合 base_dir 寻找
                possible_path = os.path.join(config.MARKDOWN_DIR, "images", os.path.basename(image_path))
                if os.path.exists(possible_path):
                    image_path = possible_path
                else:
                    return f"IMAGE_NOT_FOUND: Could not locate {image_path}"

            # 初始化一个强大的视觉模型来进行看图 (例如使用你想要的 gemini-3-pro-preview 或者是 gemini-2.5-pro)
            vision_llm = ChatOpenAI(
                api_key="sk-X9rheSBPgPaGCVK9NYCPYkoe2V2DoZ4XZcoImxsgYAAPLB5f",
                base_url="https://chatapi.zjt66.top/v1",
                model="gemini-3-pro-preview",  # 请替换为你该接口实际支持的视觉模型名称
                temperature=0
            )

            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            # 针对学术图表和复杂视觉数据的增强提示词
            vision_prompt = f"""You are an expert data analyst and computer vision specialist. Your task is to extract highly accurate information from the provided scientific image/chart to answer a specific question.

**CRITICAL RULES:**
1. **NO HALLUCINATION:** Base your answer STRICTLY on the visual evidence in the image. If the data is not visible, explicitly state "The requested data is not present in this image."
2. **STEP-BY-STEP CALIBRATION:** Before answering, briefly identify the X-axis (variable and scale), Y-axis (variable and scale), and the relevant Legend/Line color.
3. **PRECISE INTERPOLATION:** When reading a specific data point (e.g., a percentile or exact value on a curve), trace the point to the nearest grid lines or axis ticks and interpolate carefully.
4. **NO EXTERNAL KNOWLEDGE:** Do not use prior knowledge to guess the data; only read what is plotted.

**User Question:** {specific_question}

**Output Format:**
- **Visual Context:** (Briefly describe axes and the target line)
- **Data Reading:** (Explain how you traced the value)
- **Final Answer:** (The exact numeric value or direct answer)
"""

            # 组装多模态 Message
            message = HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": vision_prompt
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                ]
            )
            response = vision_llm.invoke([message])
            return response.content
            
        except Exception as e:
            return f"IMAGE_ANALYSIS_ERROR: {str(e)}"
    
    def create_tools(self) -> List:
        """Create and return the list of tools."""
        search_tool = tool("search_child_chunks")(self._search_child_chunks)
        retrieve_tool = tool("retrieve_parent_chunks")(self._retrieve_parent_chunks)
        
        # ✅ 将视觉工具注册进 Agent 的工具箱
        analyze_tool = tool("analyze_image")(self._analyze_image)
        
        return [search_tool, retrieve_tool, analyze_tool]
from typing import Literal, Set
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.types import Command
from .graph_state import State, AgentState
from .schemas import QueryAnalysis
from .prompts import *
from utils import estimate_context_tokens
from config import BASE_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR
import json
import re
import os
import config
import urllib.parse  # 新增：用于 URL 编码

def summarize_history(state: State, llm):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}
    
    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}
    
    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def rewrite_query(state: State, llm):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])

    # if response.questions and response.is_clear:
    #     delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
    #     return {"questionIsClear": True, "messages": delete_all, "originalQuery": last_message.content, "rewrittenQuestions": response.questions}

    # clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "I need more information to understand your question."
    # return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}
    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {"questionIsClear": True, "messages": delete_all, "originalQuery": last_message.content, "rewrittenQuestions": response.questions}

    # 🌟 终极修复：增加类型安全校验，防止 NoneType 报错
    clarification_text = "I need more information to understand your question."
    
    # 确保它既不是 None，也是一个字符串，并且长度足够
    if response.clarification_needed and isinstance(response.clarification_needed, str) and len(response.clarification_needed.strip()) > 5:
        clarification_text = response.clarification_needed

    return {"questionIsClear": False, "messages": [AIMessage(content=clarification_text)]}
    # last_message = state["messages"][-1]
    
    # print("🔥 跳过意图分析，直接拿着原问题去搜库！")
    
    # # 我们直接欺骗系统：告诉它问题非常清晰，不需要澄清，也不需要重写
    # return {
    #     "questionIsClear": True, 
    #     "messages": [], 
    #     "originalQuery": last_message.content, 
    #     "rewrittenQuestions": [last_message.content] # 直接用原问题去搜
    # }

def request_clarification(state: State):
    return {}

# --- Agent Nodes ---
def orchestrator(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

def fallback_response(state: AgentState, llm):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState, llm):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    # ========================================================
    # 🚀 数据飞轮核心：拦截检索到的 Contexts 并自动存入本地数据集
    # ========================================================
    
    # 从状态机中提取所有通过工具（搜索库）获取到的原始切片文本
    contexts = [msg.content for msg in state["messages"] if isinstance(msg, ToolMessage)]
    
    # 只有当真正发生了检索（contexts不为空）且生成了有效答案时，才进行记录
    if contexts and is_valid:
        data = {
            "question": state["question"], # Agent 正在处理的问题
            "contexts": contexts,          # 喂给大模型的真实切片
            "answer": answer               # 大模型生成的答案
        }
        # 以 JSONL 格式追加写入文件
        try:
            with open("rag_auto_dataset_3.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            print(f"\n[Data Flywheel] 🌟 自动拦截并保存了一条优质评估数据！")
        except Exception as e:
            print(f"自动保存数据集失败: {e}")
    # ========================================================

    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }
# --- End of Agent Nodes---

def aggregate_answers(state: State, llm):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}

# def aggregate_answers(state: State, llm):
#     if not state.get("agent_answers"):
#         return {"messages": [AIMessage(content="No answers were generated.")]}

#     sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

#     formatted_answers = ""
#     for i, ans in enumerate(sorted_answers, start=1):
#         formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

#     user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
#     synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    
#     final_text = synthesis_response.content
    
#     # =====================================================================
#     # 🌟 修复 Gradio 的本地图片渲染问题
#     # 将模型输出的任意 Markdown 图片路径拦截，替换为带 /file= 的系统绝对路径
#     # =====================================================================
#     abs_image_dir = os.path.abspath(os.path.join(config.MARKDOWN_DIR, "images"))
    
#     def replace_image_path(match):
#         alt_text = match.group(1)
#         img_path = match.group(2)
        
#         # 如果大模型已经很聪明地加了 /file=，就不重复加了
#         if "/file=" in img_path:
#             return match.group(0)
            
#         # 提取出纯文件名 (比如从 markdown_docs/images/page-1.png 提取出 page-1.png)
#         filename = os.path.basename(img_path)
#         # 拼接成 Gradio 支持的安全绝对路径
#         gradio_safe_path = f"/file={os.path.join(abs_image_dir, filename)}"
        
#         return f"![{alt_text}]({gradio_safe_path})"

#     # 使用正则匹配 ![xxx](yyy) 并替换
#     final_text_with_images = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image_path, final_text)
#     # =====================================================================

#     return {"messages": [AIMessage(content=final_text_with_images)]}

# def aggregate_answers(state: State, llm):
#     if not state.get("agent_answers"):
#         return {"messages": [AIMessage(content="No answers were generated.")]}

#     sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

#     formatted_answers = ""
#     for i, ans in enumerate(sorted_answers, start=1):
#         formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

#     user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
#     synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    
#     final_text = synthesis_response.content
    
#     # =====================================================================
#     # 🌟 修复图片空白/裂图问题：URL 编码与路径校验
#     # =====================================================================
#     abs_image_dir = os.path.abspath(os.path.join(config.MARKDOWN_DIR, "images"))
    
#     def replace_image_path(match):
#         alt_text = match.group(1)
#         img_path = match.group(2)
        
#         if "/file=" in img_path:
#             return match.group(0)
            
#         filename = os.path.basename(img_path)
#         abs_img_path = os.path.join(abs_image_dir, filename)
        
#         # 1. 物理层校验：检查图片是否真的存在
#         if not os.path.exists(abs_img_path):
#             return f"\n> ⚠️ **(原图无法显示)**: 系统引用的图片 `{filename}` 在本地库中未能找到。\n"
            
#         # 2. 编码层修复：解决空格、中文字符导致 Markdown 解析中断的问题
#         # 统一替换反斜杠为正斜杠（防止 Windows 路径混淆）
#         abs_img_path_unix = abs_img_path.replace('\\', '/')
#         # 使用 urllib 对路径进行严格的 URL 编码（比如空格变成 %20）
#         encoded_safe_path = urllib.parse.quote(abs_img_path_unix)
        
#         gradio_safe_path = f"/file={encoded_safe_path}"
        
#         return f"![{alt_text}]({gradio_safe_path})"

#     # 使用正则匹配 ![xxx](yyy) 并替换
#     final_text_with_images = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image_path, final_text)
#     # =====================================================================

#     return {"messages": [AIMessage(content=final_text_with_images)]}

# def aggregate_answers(state: State, llm):
#     if not state.get("agent_answers"):
#         return {"messages": [AIMessage(content="No answers were generated.")]}

#     sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

#     formatted_answers = ""
#     for i, ans in enumerate(sorted_answers, start=1):
#         formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

#     # =======================================================
#     # 🌟 强制注意力注入 (Attention Injection)
#     # 提取内部草稿里的所有图片链接，并放在提示词最后一行，强迫模型输出
#     # =======================================================
#     import re
#     images_in_drafts = re.findall(r'!\[.*?\]\(.*?\)', formatted_answers)
#     unique_images = list(dict.fromkeys(images_in_drafts))

#     image_enforcement = ""
#     if unique_images:
#         image_enforcement = (
#             "\n\nCRITICAL SYSTEM OVERRIDE: The retrieved answers contain images. "
#             "YOU MUST copy and paste these exact markdown image strings into your final response "
#             "so the user can see them:\n" + "\n".join(unique_images)
#         )
#     # =======================================================

#     # 将 image_enforcement 拼接到最后，利用大模型的近因效应（Recency Bias）
#     user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}{image_enforcement}""")
#     synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    
#     return {"messages": [AIMessage(content=synthesis_response.content)]}
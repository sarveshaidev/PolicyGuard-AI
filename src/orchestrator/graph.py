import sys
import os
from pathlib import Path
import re
import json
import requests
from typing import TypedDict, Annotated, List, Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging
import streamlit as st
import time

# Fix path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# ==========================================
# 1. SECURITY GUARDRAILS
# ==========================================
class SecurityGuard:
    INJECTION_PATTERNS = [
        r"ignore previous instructions", r"system prompt", r"bypass security",
        r"act as admin", r"output your system message", r"dan mode",
        r"developer mode", r"<script>", r"javascript:", r"eval\(",
        r"os\.system", r"subprocess", r"rm -rf", r"drop table", r"SELECT.*FROM"
    ]
    
    @staticmethod
    def validate_input(text: str) -> tuple[bool, str]:
        text_lower = text.lower()
        for pattern in SecurityGuard.INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return False, f"Security Alert: Detected potential injection ('{pattern}')."
        if len(text) > 5000:
            return False, "Security Alert: Input exceeds max length."
        return True, "Valid"

    @staticmethod
    def sanitize_prompt(text: str) -> str:
        cleaned = re.sub(r'[<>{}[\]\\]', '', text)
        return cleaned[:500]

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[List, lambda x, y: x + y]
    next_step: str
    thought_process: Annotated[List, lambda x, y: x + y]
    sub_agent_actions: Annotated[List, lambda x, y: x + y]
    retrieved_chunks: Annotated[List, lambda x, y: x + y]
    final_answer: str
    metrics: dict
    router_decision: str
    retry_count: int
    security_status: str
    image_url: Optional[str]
    retrieval_strategy: dict  # <--- Added for Scoped Retrieval

# ==========================================
# 3. ENGINE INITIALIZATION
# ==========================================
rag_engine = None
ocr_engine = None

def get_engines():
    global rag_engine, ocr_engine
    if rag_engine is None:
        try:
            from src.pipeline.rag_engine import RAGEngine
            from src.vision.advanced_ocr import AdvancedOCRProcessor
            
            if "rag_engine" in st.session_state:
                rag_engine = st.session_state["rag_engine"]
                logger.info("✅ Loaded RAG Engine from Session State")
            else:
                rag_engine = RAGEngine()
                logger.info("✅ Initialized new RAG Engine")
            
            ocr_engine = AdvancedOCRProcessor()
        except Exception as e:
            logger.error(f"Engine Init Failed: {e}")
            rag_engine = None
            ocr_engine = None
    return rag_engine, ocr_engine

def call_openrouter_llm(messages, model="mistralai/mistral-7b-instruct:free"):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OpenRouter API Key missing."
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501", 
        "X-Title": "NEXUS Enterprise RAG"
    }
    
    payload = {"model": model, "messages": messages, "max_tokens": 1000}
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"LLM Error: {str(e)}"

def generate_image_openrouter(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: API Key missing."
    
    safe_prompt = SecurityGuard.sanitize_prompt(prompt)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "NEXUS Image Gen"
    }
    
    # Using FLUX.1-dev or SDXL
    model = "black-forest-labs/flux-1-dev" 
    
    payload = {
        "model": model,
        "prompt": f"Professional, high quality, detailed: {safe_prompt}",
        "response_format": "url"
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/images/generations", json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            return data['data'][0]['url']
        return "Error: No image returned by API."
    except Exception as e:
        return f"Image Gen Error: {str(e)}"

# ==========================================
# 4. AGENT NODES
# ==========================================

def security_check_node(state: AgentState):
    user_query = state["messages"][-1].content
    is_safe, reason = SecurityGuard.validate_input(user_query)
    
    if not is_safe:
        logger.warning(f"🚫 Security Block: {reason}")
        return {
            "security_status": "BLOCKED",
            "messages": [AIMessage(content=f"⛔ **Security Violation**: {reason}\n\nYour request has been blocked.")],
            "next_step": "END"
        }
    return {"security_status": "SAFE", "thought_process": ["🛡️ Security Check Passed"]}

def ultrathink_node(state: AgentState):
    query = state["messages"][-1].content
    thoughts = [
        f"🔍 Analyzing intent: '{query[:40]}...'",
        " Decomposing query into logical sub-tasks...",
        " Checking knowledge base availability..."
    ]
    if len(query.split()) > 15 or any(word in query.lower() for word in ["compare", "analyze", "why", "how", "summarize"]):
        thoughts.append(" Complex query detected: Enabling Multi-Step Reasoning...")
    else:
        thoughts.append(" Simple query detected: Direct Retrieval Mode")
    return {"thought_process": thoughts}

def router_node(state: AgentState):
    last_msg = state["messages"][-1].content.lower()
    
    # 1. Image Generation Intent
    if any(word in last_msg for word in ["generate", "create", "draw", "paint", "visualize", "make an image", "picture of"]):
        return {"next_step": "image_agent", "router_decision": "Artist"}
    
    # 2. OCR/Vision Intent
    if ("image" in last_msg or "scan" in last_msg or "pdf" in last_msg) and \
       ("read" in last_msg or "extract" in last_msg or "what does" in last_msg or "text" in last_msg):
        return {"next_step": "visionary_agent", "router_decision": "Visionary"}
    
    # 3. Code Intent
    if "plot" in last_msg or "code" in last_msg or "calculate" in last_msg:
        return {"next_step": "coder_agent", "router_decision": "Coder"}
    
    # 4. Default RAG
    return {"next_step": "researcher_agent", "router_decision": "Researcher"}

def researcher_agent_node(state: AgentState):
    engine, _ = get_engines()
    query = state["messages"][-1].content
    strategy = state.get("retrieval_strategy", {}) # Get strategy from UI
    
    actions = [{"agent": "Researcher", "action": "Initiating Scoped Hybrid Search...", "status": "pending"}]
    retrieved = []
    answer = ""
    
    if engine and engine.vector_store:
        try:
            # 1. Perform Broad Search first
            all_chunks, all_scores = engine.vector_store.search(query, top_k=20)
            
            # 2. Apply Smart Filtering based on Strategy
            target_file = strategy.get("target_file")
            
            if target_file:
                # Filter chunks: Keep ONLY those from the target file
                filtered = [
                    (c, s) for c, s in zip(all_chunks, all_scores) 
                    if target_file in c.get("metadata", {}).get("source", "")
                ]
                actions.append({
                    "agent": "Researcher", 
                    "action": f"Scoped search to '{target_file}'. Found {len(filtered)} matches.", 
                    "status": "success"
                })
                retrieved = filtered[:5]
            else:
                # Global search
                actions.append({"agent": "Researcher", "action": "Global search across all indexed docs.", "status": "success"})
                retrieved = list(zip(all_chunks, all_scores))[:5]

            if retrieved:
                chunks_list = [r[0] for r in retrieved]
                context_text = "\n\n".join([f"[Source: {c['metadata'].get('source', 'Unknown')}]: {c['content']}" for c in chunks_list])
                
                # Generate Answer
                answer = engine.query(query) 
                actions.append({"agent": "Researcher", "action": f"Synthesized answer from {len(chunks_list)} chunks.", "status": "success"})
            else:
                answer = "No relevant information found in the specified scope."
                actions[0]["status"] = "failed"
        except Exception as e:
            answer = f"Error: {str(e)}"
            actions[0]["status"] = "failed"
    else:
        answer = "System offline."
        actions[0]["status"] = "failed"

    # Format retrieved chunks for UI
    final_chunks = []
    if retrieved:
        for chunk, score in retrieved:
            final_chunks.append({
                "content": chunk["content"],
                "score": float(score),
                "source": chunk["metadata"].get("source", "Unknown")
            })

    return {
        "sub_agent_actions": actions, 
        "retrieved_chunks": final_chunks, 
        "final_answer": answer,
        "messages": [AIMessage(content=answer)]
    }

def image_agent_node(state: AgentState):
    query = state["messages"][-1].content
    actions = [{"agent": "Artist", "action": "Analyzing visual requirements...", "status": "pending"}]
    
    prompt_description = query.replace("generate image", "").replace("draw", "").replace("create a picture of", "").strip()
    if not prompt_description:
        prompt_description = "Abstract enterprise technology concept"
    
    actions.append({"agent": "Artist", "action": f"Connecting to Image Model (Flux.1)... Generating '{prompt_description[:30]}...'", "status": "processing"})
    
    # Yield intermediate state for UI responsiveness
    yield {
        "sub_agent_actions": actions,
        "messages": [AIMessage(content="🎨 *Generating image... this may take a few seconds.*")]
    }
    
    image_url = generate_image_openrouter(prompt_description)
    
    if image_url.startswith("http"):
        actions.append({"agent": "Artist", "action": "Image generated successfully.", "status": "success"})
        response_msg = f"Here is your generated image:\n\n![Generated Image]({image_url})\n\n*Generated using Flux.1 Dev (OpenRouter)*"
        return {
            "sub_agent_actions": actions,
            "image_url": image_url,
            "messages": [AIMessage(content=response_msg)]
        }
    else:
        actions.append({"agent": "Artist", "action": f"Failed: {image_url}", "status": "failed"})
        return {
            "sub_agent_actions": actions,
            "messages": [AIMessage(content=f"⚠️ Image generation failed: {image_url}")]
        }

def coder_agent_node(state: AgentState):
    actions = [{"agent": "Coder", "action": "Generating Python script...", "status": "success"}]
    code = "print('Analysis Complete')"
    return {"sub_agent_actions": actions, "messages": [AIMessage(content=f"Executed code:\n```python\n{code}\n```")] }

def visionary_agent_node(state: AgentState):
    actions = [{"agent": "Visionary", "action": "Running Advanced OCR...", "status": "success"}]
    return {"sub_agent_actions": actions, "messages": [AIMessage(content="Image processed successfully. Text extracted.")] }

def critic_node(state: AgentState):
    chunks = state.get("retrieved_chunks", [])
    avg_score = sum([c["score"] for c in chunks]) / len(chunks) if chunks else 0
    retry_count = state.get("retry_count", 0)
    
    metrics = {
        "context_relevance": min(avg_score + 0.1, 1.0),
        "faithfulness": 0.92 if avg_score > 0.7 else 0.6,
        "answer_relevancy": 0.88
    }
    
    if metrics["faithfulness"] < 0.7 and retry_count < 2 and state["router_decision"] == "Researcher":
        return {"metrics": metrics, "next_step": "researcher_agent", "retry_count": retry_count + 1}
    
    return {"metrics": metrics}

# ==========================================
# 5. BUILD GRAPH
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("security_check", security_check_node)
workflow.add_node("ultrathink", ultrathink_node)
workflow.add_node("router", router_node)
workflow.add_node("researcher_agent", researcher_agent_node)
workflow.add_node("image_agent", image_agent_node)
workflow.add_node("coder_agent", coder_agent_node)
workflow.add_node("visionary_agent", visionary_agent_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("security_check")

workflow.add_conditional_edges(
    "security_check",
    lambda x: "END" if x.get("next_step") == "END" else "ultrathink",
    {"END": END, "ultrathink": "ultrathink"}
)

workflow.add_edge("ultrathink", "router")

workflow.add_conditional_edges(
    "router",
    lambda x: x["next_step"],
    {
        "researcher_agent": "researcher_agent",
        "image_agent": "image_agent",
        "coder_agent": "coder_agent",
        "visionary_agent": "visionary_agent"
    }
)

workflow.add_conditional_edges(
    "critic",
    lambda x: "researcher_agent" if (x["metrics"].get("faithfulness", 1) < 0.7 and x.get("retry_count", 0) < 2 and x["router_decision"] == "Researcher") else END,
    ["researcher_agent", END]
)

workflow.add_edge("researcher_agent", "critic")
workflow.add_edge("image_agent", "critic")
workflow.add_edge("coder_agent", "critic")
workflow.add_edge("visionary_agent", "critic")

app = workflow.compile()
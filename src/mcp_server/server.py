import sys
import os
from pathlib import Path

# ==========================================
# 1. FIX PYTHON PATH (Critical)
# ==========================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==========================================
# 2. IMPORTS (Fixed Logger & MCP v2)
# ==========================================
import logging
logger = logging.getLogger(__name__)

try:
    from mcp.server.mcpserver import MCPServer
    MCP_CLASS = MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
        MCP_CLASS = FastMCP
        logger.warning("Using MCP v1 (FastMCP). Consider upgrading to v2.")
    except ImportError:
        print("[ERROR] MCP library not found.")
        sys.exit(1)

try:
    from src.pipeline.rag_engine import RAGEngine
except ImportError as e:
    print(f"[CRITICAL ERROR] Failed to import RAGEngine: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("[INFO] Initializing EnterpriseRAG-Server...")
    
    mcp = MCP_CLASS("EnterpriseRAG-Server")
    
    engine = None
    try:
        engine = RAGEngine()
        print("[INFO] RAG Engine initialized successfully.")
    except Exception as e:
        print(f"[WARNING] Engine init failed: {e}")

    @mcp.tool()
    def query_knowledge_base(query: str) -> str:
        if engine is None:
            return "Error: Engine not initialized."
        try:
            return engine.query(query)
        except Exception as e:
            return f"Error: {str(e)}"

    print("[INFO] Starting MCP Server...")
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
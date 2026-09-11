try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # Fallback if fastmcp is not available
    print("Warning: mcp.server.fastmcp not found. Install with: pip install mcp")
    FastMCP = None

from src.pipeline.rag_engine import RAGEngine
import logging

logger = logging.getLogger(__name__)

if FastMCP:
    # Initialize MCP Server
    mcp = FastMCP("EnterpriseRAG-Server")
    engine = RAGEngine()

    @mcp.tool()
    def query_knowledge_base(query: str) -> str:
        """
        Search the enterprise knowledge base to answer questions about documents and images.
        Use this tool when the user asks about company data, PDFs, or visual assets.
        """
        logger.info(f"MCP Tool invoked: query_knowledge_base with query: {query}")
        try:
            return engine.query(query)
        except Exception as e:
            return f"Error querying knowledge base: {str(e)}"

    if __name__ == "__main__":
        # Run the MCP server
        mcp.run()
else:
    print("MCP server cannot be initialized due to missing dependencies.")
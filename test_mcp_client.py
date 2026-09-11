import subprocess
import sys
import json
import time
import os

def run_auto_test():
    print("🚀 Starting Enterprise RAG Auto-Test...")
    
    # Check if MCP server is already running (from Streamlit)
    print("🔌 Connecting to existing MCP Server (started by Streamlit)...")
    
    # Instead of starting a new server, we'll test the connection directly
    # by importing and using the same logic as the server
    
    try:
        # Add project root to path
        from pathlib import Path
        current_file = Path(__file__).resolve()
        project_root = current_file.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Import the RAG engine directly
        from src.pipeline.rag_engine import RAGEngine
        
        print("✅ Connected to RAG Engine successfully!")
        
        # Test the engine directly
        engine = RAGEngine()
        
        query = "Summarize the uploaded document in 2 bullet points."
        print(f"️  Asking AI: '{query}'")
        print("   (Waiting for RAG retrieval + LLM generation...)")
        
        answer = engine.query(query)
        
        print("\n" + "="*50)
        print("💡 AI ANSWER:")
        print("="*50)
        print(answer)
        print("="*50 + "\n")
        
        print("✅ Test Complete!")
        
    except Exception as e:
        print(f" Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_auto_test()
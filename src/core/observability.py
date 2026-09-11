import os
import json
import time
import logging
from pathlib import Path
from functools import wraps
from config.settings import TRACE_DIR

# Initialize Logger first
logger = logging.getLogger(__name__)

# ==========================================
# 1. LANGSMITH INITIALIZATION (Global Scope)
# ==========================================
traceable = None
wrapped_client = None

if os.getenv("LANGSMITH_API_KEY"):
    try:
        from langsmith.wrappers import wrap_openai
        from langsmith import traceable as ls_traceable
        import openai
        
        # Wrap the OpenAI client globally
        wrapped_client = wrap_openai(openai.Client())
        traceable = ls_traceable
        logger.info("✅ LangSmith tracing enabled!")
    except ImportError:
        logger.warning("LangSmith library not found. Install with: pip install langsmith")
        traceable = None
else:
    logger.warning("LANGSMITH_API_KEY not found. Tracing disabled.")

# Fallback decorator if LangSmith is not available
if traceable is None:
    def dummy_traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    traceable = dummy_traceable

# ==========================================
# 2. CUSTOM TRACE DECORATOR
# ==========================================
def trace_operation(operation_name: str):
    """Decorator to trace function execution and send to LangSmith + Local JSONL."""
    def decorator(func):
        @wraps(func)
        @traceable(name=operation_name, tags=["enterprise-rag"])
        def wrapper(*args, **kwargs):
            start_time = time.time()
            trace_log = {
                "operation": operation_name,
                "timestamp": time.time(),
                "input_args": str(args)[:200], # Truncate for safety
            }
            try:
                result = func(*args, **kwargs)
                trace_log["output"] = str(result)[:200]
                trace_log["status"] = "success"
                return result
            except Exception as e:
                trace_log["error"] = str(e)
                trace_log["status"] = "failure"
                raise
            finally:
                trace_log["latency_ms"] = round((time.time() - start_time) * 1000, 2)
                
                # Ensure trace directory exists
                TRACE_DIR.mkdir(parents=True, exist_ok=True)
                
                # Append to local JSONL file (Backup)
                with open(TRACE_DIR / "traces.jsonl", "a") as f:
                    f.write(json.dumps(trace_log) + "\n")
                
                logger.info(f"[TRACE] {operation_name} | {trace_log['latency_ms']}ms | {trace_log['status']}")
        
        return wrapper
    return decorator
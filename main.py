import os
import sys
import logging
from pathlib import Path  # Kept for compatibility, but logic uses os.path for safety
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
import uvicorn

# --- Project Imports ---
# Ensure the src directory is in the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from retrieval.hybrid_search import HybridVectorStore
    from orchestrator.graph import app as langgraph_app
    from core.embedder_singleton import embedder
    from ingestion.universal_parser import UniversalParser
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    print("Ensure your folder structure is: Enterprise_RAG/src/...")
    sys.exit(1)

from langchain_core.messages import HumanMessage

# ==========================================
# 1. CONFIG & LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load .env if exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env load")

# ==========================================
# 2. FASTAPI APP INIT
# ==========================================
app = FastAPI(
    title="NEXUS Enterprise API",
    description="Secure, Hybrid RAG API with RBAC and Self-Healing Agents",
    version="3.0.0"
)

# ==========================================
# 3. SECURITY CONFIG (RBAC)
# ==========================================
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey_change_in_prod_12345")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock User Database (Replace with PostgreSQL in Production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@nexus.ai",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW", # "secret"
        "role": "admin",
        "disabled": False,
    },
    "editor": {
        "username": "editor",
        "full_name": "Editor User",
        "email": "editor@nexus.ai",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW", # "secret"
        "role": "editor",
        "disabled": False,
    },
    "viewer": {
        "username": "viewer",
        "full_name": "Viewer User",
        "email": "viewer@nexus.ai",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW", # "secret"
        "role": "viewer",
        "disabled": False,
    }
}

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return user_dict
    return None

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user

def require_role(required_role: str):
    async def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] != required_role and current_user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return role_checker

# ==========================================
# 5. GLOBAL RAG INSTANCE (FIXED PATHS)
# ==========================================
# Use absolute path based on this file's location to avoid "Path not found" errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH_STR = os.path.join(BASE_DIR, "data", "vector_db")
DB_PATH_OBJ = Path(DB_PATH_STR)

# Ensure directory exists
os.makedirs(DB_PATH_STR, exist_ok=True)

logger.info(f"📂 Initializing Vector Store at: {DB_PATH_STR}")

vector_store = HybridVectorStore(db_path=DB_PATH_OBJ)

if vector_store.load():
    logger.info(f"OK: Successfully loaded Vector DB from {DB_PATH_STR}")
    logger.info(f"   Chunks loaded: {len(vector_store.chunks)}")
    if hasattr(vector_store, 'faiss_index'):
        logger.info(f"   Index size: {vector_store.faiss_index.ntotal}")
else:
    logger.warning(f"WARN: No existing vector DB found at {DB_PATH_STR}. Ready for ingestion.")

# ==========================================
# 6. PYDANTIC MODELS
# ==========================================
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    metrics: dict
    agent: str
    latency_ms: int

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    chunks_added: int

# ==========================================
# 7. API ENDPOINTS
# ==========================================

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "role": user["role"]}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_role("editor"))
):
    logger.info(f"User {current_user['username']} uploading {file.filename}")
    
    try:
        content = await file.read()
        
        # Simple parsing logic (Extend with UniversalParser for real PDFs)
        # For demo, we assume text extraction happened
        # In a real scenario: text = UniversalParser.parse_bytes(content, file.filename)
        
        # Mocking chunk creation for demonstration if parser isn't fully linked
        # Replace this block with actual parsing logic:
        text_content = f"Content of {file.filename} (Mocked for Demo)" 
        chunks = [{"content": text_content, "metadata": {"source": file.filename}}]
        
        # Generate embeddings
        texts = [c["content"] for c in chunks]
        embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
        
        # Add to Vector Store
        vector_store.add_chunks(chunks, embeddings)
        vector_store.save()
        
        logger.info(f"OK: Indexed {len(chunks)} chunks from {file.filename}")
        
        return UploadResponse(
            status="success",
            message=f"File {file.filename} processed successfully",
            filename=file.filename,
            chunks_added=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=ChatResponse)
async def query_knowledge_base(
    request: QueryRequest,
    current_user: dict = Depends(require_role("viewer"))
):
    import time
    start_time = time.time()
    
    logger.info(f"Query from {current_user['username']}: {request.query}")
    
    # 1. Hybrid Retrieval
    chunks, scores = vector_store.search(request.query, top_k=request.top_k)
    
    if not chunks:
        latency = int((time.time() - start_time) * 1000)
        return ChatResponse(
            answer="No relevant context found in the knowledge base.",
            sources=[],
            metrics={},
            agent="Researcher",
            latency_ms=latency
        )

    # 2. Prepare Context
    context_text = "\n\n".join([f"[Source {i+1}]: {c['content']}" for i, c in enumerate(chunks)])
    
    # 3. Call LangGraph Agent
    inputs = {
        "messages": [HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {request.query}")],
        "next_step": "", "thought_process": [], "sub_agent_actions": [],
        "final_answer": "", "metrics": {}, "router_decision": "", "retrieved_chunks": [], "retry_count": 0
    }
    
    final_state = None
    try:
        # Run the graph synchronously for the API response
        for event in langgraph_app.stream(inputs, stream_mode="values"):
            final_state = event
        
        if final_state:
            answer = final_state["messages"][-1].content
            agent = final_state.get("router_decision", "General")
            metrics = final_state.get("metrics", {})
        else:
            answer = "Error: Agent returned no response."
            agent = "Error"
            metrics = {}
            
    except Exception as e:
        logger.error(f"Graph Execution Error: {e}")
        answer = f"System error during processing: {str(e)}"
        agent = "Error"
        metrics = {}

    sources = [{"content": c["content"][:150] + "...", "score": float(s)} for c, s in zip(chunks, scores)]
    latency_ms = int((time.time() - start_time) * 1000)

    return ChatResponse(
        answer=answer,
        sources=sources,
        metrics=metrics,
        agent=agent,
        latency_ms=latency_ms
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "NEXUS API",
        "version": "3.0.0",
        "vector_db_path": DB_PATH_STR,
        "chunks_loaded": len(vector_store.chunks) if hasattr(vector_store, 'chunks') else 0
    }

# ==========================================
# 8. RUN SERVER
# ==========================================
if __name__ == "__main__":
    # Run with Uvicorn directly
    # --reload enables auto-restart on code change (Dev mode)
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # if __name__ == "__main__":
    # # Changed from 0.0.0.0 to 127.0.0.1 for Windows compatibility
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
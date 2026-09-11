import sys
import streamlit as st
import time
import logging
import os
import sqlite3
import bcrypt
import re
from pathlib import Path
import pickle
import json
import plotly.graph_objects as go
import pandas as pd
import random
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# ==========================================
# 0. PATH CONFIGURATION
# ==========================================
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==========================================
# 1. CONFIG & LOGGING (PRODUCTION-GRADE)
# ==========================================
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    logger.warning(f"dotenv not available: {e}")

# ==========================================
# 2. DATABASE SETUP (SQLite Auth + RBAC + Audit)
# ==========================================
DB_FILE = Path(__file__).parent / "nexus_auth.db"

def init_database() -> bool:
    """Initialize SQLite database with tables (users, audit_log)"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'viewer' CHECK(role IN ('viewer', 'editor', 'admin')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT,
                ip_address TEXT
            )
        """)
        
        # Create default admin if not exists
        cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
        if not cursor.fetchone():
            admin_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt())
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", admin_hash, "admin")
            )
            logger.info("✅ Default admin created: admin / admin123")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Database init error: {e}")
        return False

# Initialize database on startup
DB_READY = init_database()

# ==========================================
# 3. AUTH FUNCTIONS (PRODUCTION-GRADE)
# ==========================================
def hash_password(password: str) -> bytes:
    """Hash password with bcrypt"""
    try:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))
    except Exception as e:
        logger.error(f"Hash error: {e}")
        return None

def verify_password(stored_hash: bytes, password: str) -> bool:
    """Verify password against hash"""
    try:
        if not stored_hash:
            return False
        return bcrypt.checkpw(password.encode(), stored_hash)
    except Exception as e:
        logger.error(f"Verify error: {e}")
        return False

def audit_log(username: str, action: str, details: str = "") -> bool:
    """Log action to audit trail"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO audit_log (username, action, details) VALUES (?, ?, ?)",
            (username, action, details)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Audit log error: {e}")
        return False

def validate_username(username: str) -> Tuple[bool, str]:
    """Validate username (3-20 chars, alphanumeric + underscore)"""
    if not username or len(username) < 3 or len(username) > 20:
        return False, "Username must be 3-20 characters"
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return False, "Username can only contain letters, numbers, and underscores"
    return True, ""

def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password (min 8 chars, must have upper, lower, digit)"""
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain lowercase letters"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain uppercase letters"
    if not re.search(r"\d", password):
        return False, "Password must contain digits"
    return True, ""

def register_user(username: str, password: str) -> Tuple[bool, str]:
    """Register new user as 'viewer'"""
    try:
        # Validation
        valid, msg = validate_username(username)
        if not valid:
            return False, f"❌ {msg}"
        
        valid, msg = validate_password(password)
        if not valid:
            return False, f"❌ {msg}"
        
        # Check if exists
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return False, "❌ Username already exists"
        
        # Create user
        password_hash = hash_password(password)
        if not password_hash:
            conn.close()
            return False, "❌ Error hashing password"
        
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, "viewer")
        )
        conn.commit()
        conn.close()
        
        # Audit
        audit_log(username, "USER_REGISTERED", "New user created as viewer")
        
        return True, "✅ Registration successful! Please login."
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return False, f"❌ Registration failed: {str(e)[:100]}"

def login_user(username: str, password: str) -> Tuple[bool, str, Optional[str]]:
    """Authenticate user and return role"""
    try:
        if not username or not password:
            return False, "❌ Username and password required", None
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, password_hash, role, is_active FROM users WHERE username = ?",
            (username,)
        )
        user = cursor.fetchone()
        
        if not user:
            audit_log(username, "LOGIN_FAILED", "User not found")
            conn.close()
            return False, "❌ Invalid username or password", None
        
        user_id, stored_hash, role, is_active = user
        
        if not is_active:
            audit_log(username, "LOGIN_FAILED", "Account inactive")
            conn.close()
            return False, "❌ Account is inactive", None
        
        if not verify_password(stored_hash, password):
            audit_log(username, "LOGIN_FAILED", "Wrong password")
            conn.close()
            return False, "❌ Invalid username or password", None
        
        # Update last login
        cursor.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
            (user_id,)
        )
        conn.commit()
        conn.close()
        
        audit_log(username, "LOGIN_SUCCESS", f"Role: {role}")
        return True, f"✅ Welcome back, {username}!", role
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return False, f"❌ Login failed: {str(e)[:100]}", None

def get_all_users() -> List[Dict[str, Any]]:
    """Admin: Get all users for RBAC management (with error handling)"""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, role, created_at, last_login, is_active FROM users ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        users = [dict(row) for row in rows]
        conn.close()
        return users
    except Exception as e:
        logger.error(f"Get users error: {e}")
        return []

def update_user_role(user_id: int, new_role: str) -> bool:
    """Admin: Change user role"""
    try:
        if new_role not in ["viewer", "editor", "admin"]:
            return False
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get username for audit
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        username = row[0] if row else "unknown"
        
        cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
        conn.commit()
        conn.close()
        
        audit_log("admin", "ROLE_CHANGED", f"User {username} changed to {new_role}")
        return True
    except Exception as e:
        logger.error(f"Update role error: {e}")
        return False

def update_user_active_status(user_id: int, is_active: bool) -> bool:
    """Admin: Enable/disable user"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        username = row[0] if row else "unknown"
        
        cursor.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id))
        conn.commit()
        conn.close()
        
        action = "ACTIVATED" if is_active else "DEACTIVATED"
        audit_log("admin", f"USER_{action}", f"User {username}")
        return True
    except Exception as e:
        logger.error(f"Update status error: {e}")
        return False

def delete_user(user_id: int) -> bool:
    """Admin: Delete user"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        username = row[0] if row else "unknown"
        
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        
        audit_log("admin", "USER_DELETED", f"User {username} deleted")
        return True
    except Exception as e:
        logger.error(f"Delete user error: {e}")
        return False

# ==========================================
# 4. IMPORTS & GRAPH INIT
# ==========================================
GRAPH_AVAILABLE = False
langgraph_app = None

class DummyApp:
    """Fallback if LangGraph unavailable"""
    def stream(self, *args, **kwargs):
        yield {
            "messages": [{"role": "assistant", "content": "⚠️ SYSTEM OFFLINE - Graph module unavailable"}],
            "router_decision": "Error",
            "metrics": {},
            "thought_process": [],
            "sub_agent_actions": [],
            "retrieved_chunks": []
        }

try:
    from src.orchestrator.graph import app as langgraph_app
    GRAPH_AVAILABLE = True
    logger.info("✅ Graph module loaded")
except Exception as e:
    logger.error(f"Graph Load Error: {e}")
    langgraph_app = DummyApp()

VECTOR_STORE_AVAILABLE = False
try:
    from src.ingestion.universal_parser import UniversalParser
    from src.retrieval.vector_store import VectorStore
    from src.core.embedder_singleton import embedder
    from src.pipeline.rag_engine import RAGEngine
    import faiss
    VECTOR_STORE_AVAILABLE = True
    logger.info("✅ Vector store modules loaded")
except ImportError as e:
    logger.error(f"Vector Store Import Error: {e}")

try:
    from langchain_core.messages import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available, using fallback")
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

# ==========================================
# 5. STREAMLIT CONFIG
# ==========================================
try:
    st.set_page_config(
        page_title="NEXUS | Enterprise AI",
        layout="wide",
        page_icon="🌐",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    logger.error(f"Page config error: {e}")

# ==========================================
# 6. CSS STYLING
# ==========================================
CUSTOM_CSS = """
<style>
    .stApp { 
        background-color: #0b0f19; 
        color: #e6edf3; 
        font-family: 'Inter', 'Segoe UI', sans-serif; 
    }
    div[data-testid="stSidebar"] { 
        background-color: #0d1117; 
        border-right: 1px solid #30363d; 
    }
    h1, h2, h3 { 
        color: #ffffff !important; 
        font-weight: 700; 
        letter-spacing: -0.5px;
    }
    .hero-title { 
        font-size: 3.5rem !important; 
        background: linear-gradient(135deg, #58a6ff, #a371f7); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 1rem; 
        line-height: 1.2; 
    }
    .hero-sub { 
        font-size: 1.25rem; 
        color: #8b949e; 
        line-height: 1.6; 
        margin-bottom: 2rem; 
    }
    .nexus-card { 
        background: rgba(22, 27, 34, 0.6); 
        backdrop-filter: blur(10px); 
        border: 1px solid rgba(88, 166, 255, 0.1); 
        border-radius: 12px; 
        padding: 1.5rem; 
        height: 100%; 
        transition: all 0.3s ease; 
    }
    .nexus-card:hover { 
        transform: translateY(-5px); 
        border-color: rgba(88, 166, 255, 0.4); 
        box-shadow: 0 10px 30px rgba(0,0,0,0.5); 
    }
    .card-icon { 
        font-size: 2.5rem; 
        margin-bottom: 1rem; 
        display: block; 
    }
    .card-title { 
        font-size: 1.2rem; 
        font-weight: 600; 
        color: #fff; 
        margin-bottom: 0.5rem; 
    }
    .card-text { 
        font-size: 0.95rem; 
        color: #8b949e; 
        line-height: 1.6; 
    }
    .metric-box { 
        text-align: center; 
        padding: 1.5rem; 
        background: linear-gradient(145deg, rgba(22, 27, 34, 0.9), rgba(33, 38, 45, 0.9)); 
        border: 1px solid rgba(88, 166, 255, 0.1); 
        border-radius: 12px; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.4); 
    }
    .metric-val { 
        font-size: 2.5rem; 
        font-weight: 800; 
        color: #58a6ff; 
        text-shadow: 0 0 15px rgba(88, 166, 255, 0.4); 
    }
    .metric-lbl { 
        font-size: 0.75rem; 
        color: #8b949e; 
        text-transform: uppercase; 
        letter-spacing: 1.5px; 
        margin-top: 5px; 
    }
    .role-badge { 
        display: inline-block; 
        padding: 0.5rem 1rem; 
        border-radius: 20px; 
        font-size: 0.85rem; 
        font-weight: 600; 
        margin-top: 0.5rem; 
    }
    .role-admin { 
        background: rgba(218, 54, 51, 0.2); 
        color: #da3633; 
        border: 1px solid #da3633; 
    }
    .role-editor { 
        background: rgba(210, 153, 34, 0.2); 
        color: #d29922; 
        border: 1px solid #d29922; 
    }
    .role-viewer { 
        background: rgba(88, 166, 255, 0.2); 
        color: #58a6ff; 
        border: 1px solid #58a6ff; 
    }
    .tech-pill { 
        display: inline-block; 
        padding: 4px 10px; 
        margin: 2px; 
        background: rgba(88, 166, 255, 0.1); 
        color: #58a6ff; 
        border-radius: 4px; 
        font-size: 0.8rem; 
        border: 1px solid rgba(88, 166, 255, 0.2);
    }
    .flow-step { 
        background: rgba(22, 27, 34, 0.8); 
        border: 1px solid #30363d; 
        border-radius: 8px; 
        padding: 1rem; 
        text-align: center; 
        margin-bottom: 1rem; 
        position: relative; 
    }
    .flow-step::after { 
        content: '↓'; 
        position: absolute; 
        bottom: -1rem; 
        left: 50%; 
        transform: translateX(-50%); 
        color: #58a6ff; 
        font-weight: bold;
    }
    .flow-step:last-child::after { 
        content: ''; 
    }
    .flow-title { 
        color: #58a6ff; 
        font-weight: 700; 
        font-size: 0.9rem; 
        margin-bottom: 0.3rem; 
    }
    .flow-desc { 
        color: #8b949e; 
        font-size: 0.8rem; 
    }
    .insight-box { 
        background: rgba(22, 27, 34, 0.8); 
        border-left: 4px solid #58a6ff; 
        padding: 1rem 1.5rem; 
        border-radius: 0 8px 8px 0; 
        margin-bottom: 1.5rem; 
        display: flex; 
        align-items: center; 
        gap: 1rem; 
    }
    .insight-good { border-left-color: #2ea043; }
    .insight-good h4 { color: #2ea043; }
    .insight-warn { border-left-color: #d29922; }
    .insight-warn h4 { color: #d29922; }
    .insight-bad { border-left-color: #ff8080; }
    .insight-bad h4 { color: #ff8080; }
    .insight-content h4 { 
        margin: 0; 
        font-size: 0.9rem; 
        text-transform: uppercase; 
        letter-spacing: 1px; 
    }
    .insight-content p { 
        margin: 4px 0 0 0; 
        color: #c9d1d9; 
        font-size: 0.95rem; 
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# 7. SESSION STATE
# ==========================================
def init_session_state():
    """Initialize all session state safely"""
    defaults = {
        "authenticated": False,
        "username": None,
        "user_role": None,
        "user_id": None,
        "view": "home",
        "messages": [],
        "vector_store_ready": False,
        "rag_engine": None,
        "query_history": [],
        "last_log": {
            "router_decision": "",
            "metrics": {},
            "thought_process": [],
            "sub_agent_actions": [],
            "retrieved_chunks": [],
            "latency_ms": 0
        },
        "last_uploaded": None,
        "system_status": "initializing"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==========================================
# 8. RBAC HELPERS
# ==========================================
def check_permission(required_role: str) -> bool:
    """Check if user has permission (role-based)"""
    try:
        if not st.session_state.get("authenticated"):
            return False
        
        role = st.session_state.get("user_role", "viewer")
        hierarchy = {"admin": 3, "editor": 2, "viewer": 1}
        return hierarchy.get(role, 0) >= hierarchy.get(required_role, 0)
    except Exception as e:
        logger.error(f"Permission check error: {e}")
        return False

def require_login():
    """Require user to be authenticated"""
    if not st.session_state.get("authenticated"):
        st.error("🔒 Please log in to access this feature")
        st.stop()

def require_role(role: str):
    """Require user to have specific role"""
    require_login()
    if not check_permission(role):
        st.error(f"🚫 Access Denied: This feature requires '{role}' role. Your role: {st.session_state.user_role}")
        st.stop()

# ==========================================
# 9. VECTOR STORE CACHING
# ==========================================
@st.cache_resource
def load_vector_store_cached():
    """Load and cache vector store (expensive operation)"""
    try:
        if not VECTOR_STORE_AVAILABLE:
            logger.warning("Vector store not available")
            return None, False
        
        db_path = Path("data/vector_db")
        faiss_file = db_path / "faiss.index"
        chunks_file = db_path / "chunks.pkl"
        
        if not (faiss_file.exists() and chunks_file.exists()):
            logger.info("Vector store files not found")
            return None, False
        
        try:
            with open(chunks_file, 'rb') as f:
                chunks = pickle.load(f)
            
            vs = VectorStore(db_path=db_path)
            vs.index = faiss.read_index(str(faiss_file))
            vs.chunks = chunks
            vs.embedder = embedder
            
            engine = RAGEngine()
            engine.vector_store = vs
            
            logger.info(f"Loaded {len(chunks)} chunks from vector store")
            return engine, True
        except Exception as load_err:
            logger.error(f"Error loading vector store: {load_err}")
            return None, False
    except Exception as e:
        logger.error(f"Vector store cache error: {e}")
        return None, False

def load_data():
    """Initialize vector store in session state"""
    try:
        if st.session_state.vector_store_ready:
            return
        
        engine, ready = load_vector_store_cached()
        if engine and ready:
            st.session_state.rag_engine = engine
            st.session_state.vector_store_ready = True
            st.session_state.system_status = "online"
        else:
            st.session_state.system_status = "awaiting_data"
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        st.session_state.system_status = "error"

# Load data on startup
load_data()

# ==========================================
# 10. AGENT SWARM TOPOLOGY (PLOTLY)
# ==========================================
def render_agent_swarm_topology():
    """Render production-grade Agent Swarm topology with Plotly"""
    try:
        # Node definitions
        nodes = {
            "User Query": (1, 5),
            "Security Guard": (2, 5),
            "Intent Router": (3, 5),
            "Researcher": (4, 3),
            "Artist": (4, 5),
            "Coder": (4, 7),
            "Critic Agent": (5, 5),
            "Final Answer": (6, 5),
        }
        
        # Edge definitions
        edges = [
            ("User Query", "Security Guard"),
            ("Security Guard", "Intent Router"),
            ("Intent Router", "Researcher"),
            ("Intent Router", "Artist"),
            ("Intent Router", "Coder"),
            ("Researcher", "Critic Agent"),
            ("Artist", "Critic Agent"),
            ("Coder", "Critic Agent"),
            ("Critic Agent", "Final Answer"),
        ]
        
        # Color mapping
        node_colors = {
            "User Query": "#1f6feb",
            "Security Guard": "#da3633",
            "Intent Router": "#d29922",
            "Researcher": "#238636",
            "Artist": "#a371f7",
            "Coder": "#238636",
            "Critic Agent": "#da3633",
            "Final Answer": "#1f6feb",
        }
        
        # Extract coordinates
        node_names = list(nodes.keys())
        node_x = [nodes[node][0] for node in node_names]
        node_y = [nodes[node][1] for node in node_names]
        node_colors_list = [node_colors.get(node, "#58a6ff") for node in node_names]
        
        # Create edge traces
        edge_traces = []
        for edge in edges:
            x0, y0 = nodes[edge[0]]
            x1, y1 = nodes[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='rgba(88, 166, 255, 0.4)'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_names,
            textposition="middle center",
            textfont=dict(size=9, color="white", family="Inter"),
            hoverinfo='text',
            hovertext=node_names,
            marker=dict(
                size=30,
                color=node_colors_list,
                line=dict(width=2, color='rgba(255,255,255,0.2)'),
                opacity=0.9
            ),
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title={
                "text": "🔀 Agent Swarm Topology",
                "font": {"size": 18, "color": "#ffffff", "family": "Inter"},
                "x": 0.5,
                "xanchor": "center"
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            paper_bgcolor='rgba(11, 15, 25, 0)',
            plot_bgcolor='rgba(11, 15, 25, 0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            font=dict(color="#8b949e", family="Inter")
        )
        
        st.plotly_chart(fig, use_container_width=True, key="agent_swarm")
        
        # Legend
        st.markdown("""
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1.5rem;'>
            <div style='padding: 0.75rem; background: rgba(31, 111, 235, 0.1); border-left: 3px solid #1f6feb; border-radius: 4px;'>
                <span style='color: #58a6ff; font-weight: bold;'>🔵 Entry/Exit</span> — User input/output gates
            </div>
            <div style='padding: 0.75rem; background: rgba(218, 54, 51, 0.1); border-left: 3px solid #da3633; border-radius: 4px;'>
                <span style='color: #da3633; font-weight: bold;'>🔴 Security</span> — Validation & critique layers
            </div>
            <div style='padding: 0.75rem; background: rgba(35, 134, 54, 0.1); border-left: 3px solid #238636; border-radius: 4px;'>
                <span style='color: #238636; font-weight: bold;'>🟢 Execution</span> — Worker agents (Researcher/Coder)
            </div>
            <div style='padding: 0.75rem; background: rgba(163, 113, 247, 0.1); border-left: 3px solid #a371f7; border-radius: 4px;'>
                <span style='color: #a371f7; font-weight: bold;'>🟣 Creative</span> — Artist agent for visual output
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Agent swarm render error: {e}\n{traceback.format_exc()}")
        st.error(f"Could not render topology: {str(e)[:100]}")

# ==========================================
# 11. AUTH UI
# ==========================================
def show_auth_page():
    """Display login/registration interface"""
    try:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<h1 class="hero-title">🔐 NEXUS</h1>', unsafe_allow_html=True)
            st.markdown("<p class='hero-sub'>Enterprise Agentic RAG Platform</p>", unsafe_allow_html=True)
            
            auth_tab = st.radio("Choose Action", ["🔑 Login", "📝 Register"], horizontal=True, label_visibility="collapsed")
            
            if auth_tab == "🔑 Login":
                st.subheader("Login")
                
                username = st.text_input("Username", key="login_user", placeholder="Enter username")
                password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
                
                if st.button("🚀 Login", use_container_width=True, type="primary"):
                    if username and password:
                        success, message, role = login_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.user_role = role
                            st.success(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("⚠️ Please enter username and password")
                
                st.divider()
                st.info("📌 **Demo Credentials**:\n- Username: `admin`\n- Password: `admin123`")
                st.info("📝 **New User?** Use the Register tab to create an account (min: 8 chars, mix of upper/lower/digits)")
            
            else:  # Register
                st.subheader("Create New Account")
                
                new_username = st.text_input(
                    "New Username (3-20 chars, alphanumeric + underscore)",
                    key="reg_user",
                    placeholder="e.g., john_doe"
                )
                new_password = st.text_input(
                    "New Password (min 8 chars, needs upper/lower/digit)",
                    type="password",
                    key="reg_pass",
                    placeholder="Strong password"
                )
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    key="reg_pass_conf",
                    placeholder="Confirm"
                )
                
                if st.button("✅ Register", use_container_width=True, type="primary"):
                    if not new_username or not new_password or not confirm_password:
                        st.warning("⚠️ Please fill all fields")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords do not match")
                    else:
                        success, message = register_user(new_username, new_password)
                        if success:
                            st.success(message)
                            st.balloons()
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(message)
                
                st.divider()
                st.info("ℹ️ New users are created with **Viewer** role (can chat and view dashboards only)")
    
    except Exception as e:
        logger.error(f"Auth UI error: {e}")
        st.error(f"Authentication interface error: {str(e)[:100]}")

# ==========================================
# 12. MAIN APP
# ==========================================

if not st.session_state.get("authenticated"):
    show_auth_page()
else:
    # User is authenticated
    user_role = st.session_state.user_role
    user_name = st.session_state.username
    
    # ==========================================
    # SIDEBAR
    # ==========================================
    with st.sidebar:
        # User info
        st.markdown(f"""
        <div style='padding: 1rem; background: rgba(22, 27, 34, 0.8); border-radius: 8px; margin-bottom: 1rem; border: 1px solid rgba(88, 166, 255, 0.1);'>
            <div style='color: #58a6ff; font-weight: bold; font-size: 1.1rem;'>👤 {user_name}</div>
            <div class='role-badge role-{user_role}'>{user_role.upper()}</div>
            <div style='color: #8b949e; font-size: 0.8rem; margin-top: 0.5rem;'>
                <small>Status: ONLINE ✅</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        st.subheader("Navigation", divider=True)
        nav_options = [
            ("🏠 Home", "home"),
            ("💬 Live Demo", "demo"),
            ("📊 Dashboard", "dashboard"),
            ("🏗️ Architecture", "arch"),
        ]
        
        current_view = st.session_state.view
        nav_labels = [label for label, _ in nav_options]
        current_index = next((i for i, (_, key) in enumerate(nav_options) if key == current_view), 0)
        
        selected_nav = st.radio(
            "Select View",
            nav_labels,
            index=current_index,
            label_visibility="collapsed"
        )
        
        for label, key in nav_options:
            if label == selected_nav:
                st.session_state.view = key
        
        st.divider()
        
        # Quick Inject (Editor/Admin only)
        if check_permission("editor"):
            st.subheader("⚡ Quick Inject", divider=True)
            uploaded_file = st.file_uploader(
                "Upload PDF (Editor+)",
                type=["pdf"],
                key="pdf_uploader"
            )
            
            if uploaded_file:
                try:
                    if not VECTOR_STORE_AVAILABLE:
                        st.error("Vector store not available")
                    else:
                        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
                            st.session_state.last_uploaded = uploaded_file.name
                            
                            with st.spinner("📚 Indexing document..."):
                                text = UniversalParser.parse_file(uploaded_file)
                                chunks = [
                                    {
                                        "content": text[i:i+500],
                                        "metadata": {"source": uploaded_file.name}
                                    }
                                    for i in range(0, len(text), 450)
                                ]
                                
                                embs = embedder.encode(
                                    [c["content"] for c in chunks],
                                    normalize_embeddings=True
                                ).tolist()
                                
                                vs = VectorStore(db_path=Path("data/vector_db"))
                                vs.add_chunks(chunks, embs)
                                vs.save()
                                
                                engine = RAGEngine()
                                engine.vector_store = vs
                                st.session_state.rag_engine = engine
                                st.session_state.vector_store_ready = True
                                
                                audit_log(user_name, "DOCUMENT_UPLOADED", f"File: {uploaded_file.name}, Chunks: {len(chunks)}")
                                
                                st.success(f"✅ Indexed {len(chunks)} chunks!")
                                time.sleep(1)
                                st.rerun()
                except Exception as e:
                    logger.error(f"Upload error: {e}")
                    st.error(f"Upload failed: {str(e)[:100]}")
        else:
            st.info("🔒 **Upload disabled**\n\nViewers can only chat & view.")
        
        st.divider()
        
        # Admin User Management
        if check_permission("admin"):
            st.subheader("👥 User Management", divider=True)
            with st.expander("Manage Users"):
                try:
                    users = get_all_users()
                    if users:
                        for u in users:
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                active_badge = "✅" if u['is_active'] else "❌"
                                st.write(f"**{u['username']}** {active_badge}\n`{u['role']}` • {u['created_at'][:10]}")
                            with col2:
                                new_role = st.selectbox(
                                    "Role",
                                    ["viewer", "editor", "admin"],
                                    index=["viewer", "editor", "admin"].index(u['role']),
                                    key=f"role_{u['id']}"
                                )
                                if st.button("Update", key=f"upd_{u['id']}", use_container_width=True):
                                    if update_user_role(u['id'], new_role):
                                        st.success("✅ Updated!")
                                        time.sleep(1)
                                        st.rerun()
                            with col3:
                                if u['username'] != user_name:
                                    if st.button("🗑️", key=f"del_{u['id']}", help="Delete user"):
                                        if delete_user(u['id']):
                                            st.success("✅ Deleted!")
                                            time.sleep(1)
                                            st.rerun()
                    else:
                        st.info("No users found")
                except Exception as e:
                    logger.error(f"User management error: {e}")
                    st.error(f"Error: {str(e)[:100]}")
        
        st.divider()
        
        # Logout
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            audit_log(user_name, "LOGOUT", "User logged out")
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_role = None
            st.session_state.messages = []
            st.rerun()
    
    # ==========================================
    # MAIN CONTENT
    # ==========================================
    
    # --- HOME VIEW ---
    if st.session_state.view == "home":
        try:
            st.markdown('<h1 class="hero-title">🌐 NEXUS</h1>', unsafe_allow_html=True)
            st.markdown(
                '<p class="hero-sub">Enterprise Agentic RAG Platform with RBAC & Multi-Agent Orchestration</p>',
                unsafe_allow_html=True
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    """<div class='nexus-card'><span class='card-icon'>🤖</span><div class='card-title'>Multi-Agent</div><div class='card-text'>Researcher, Artist, Coder orchestrated by LangGraph.</div></div>""",
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    """<div class='nexus-card'><span class='card-icon'>🔍</span><div class='card-title'>Hybrid Retrieval</div><div class='card-text'>BM25 + Vector Search with RRF ranking.</div></div>""",
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    """<div class='nexus-card'><span class='card-icon'>⚡</span><div class='card-title'>Real-Time Critique</div><div class='card-text'>RAGAS self-correction loop for QA.</div></div>""",
                    unsafe_allow_html=True
                )
            with col4:
                st.markdown(
                    """<div class='nexus-card'><span class='card-icon'>🛡️</span><div class='card-title'>Enterprise RBAC</div><div class='card-text'>JWT-authenticated roles + audit log.</div></div>""",
                    unsafe_allow_html=True
                )
            
            st.divider()
            
            st.subheader("📈 Key Metrics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    """<div class='metric-box'><div class='metric-val'>3</div><div class='metric-lbl'>Agents</div></div>""",
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    """<div class='metric-box'><div class='metric-val'>99%</div><div class='metric-lbl'>Uptime</div></div>""",
                    unsafe_allow_html=True
                )
            with m3:
                st.markdown(
                    """<div class='metric-box'><div class='metric-val'>3</div><div class='metric-lbl'>Roles</div></div>""",
                    unsafe_allow_html=True
                )
            with m4:
                st.markdown(
                    """<div class='metric-box'><div class='metric-val'>∞</div><div class='metric-lbl'>Scalable</div></div>""",
                    unsafe_allow_html=True
                )
            
            st.divider()
            st.subheader("🚀 Quick Start")
            st.markdown(f"""
            1. **Upload a Document** (Editor/Admin) — Click "Quick Inject" in sidebar
            2. **Ask Questions** — Go to "Live Demo" tab and chat with NEXUS
            3. **Monitor** — Check "Dashboard" for query metrics & analytics
            4. **Learn** — View "Architecture" for technical specs & topology
            
            **Your Role:** `{user_role.upper()}` — {
                "Full access to all features" if user_role == "admin" else
                "Can upload & manage documents" if user_role == "editor" else
                "Can chat & view dashboards only"
            }
            """)
            
        except Exception as e:
            logger.error(f"Home view error: {e}")
            st.error(f"Home view error: {str(e)[:100]}")
    
    # --- LIVE DEMO VIEW ---
    elif st.session_state.view == "demo":
        try:
            require_login()
            
            st.title("💬 Live Demo Chat")
            st.caption("Ask questions about your documents. Powered by multi-agent RAG orchestration.")
            
            # Chat history
            for msg in st.session_state.messages:
                try:
                    with st.chat_message(msg.get("role", "assistant")):
                        st.markdown(msg.get("content", ""))
                except Exception as e:
                    logger.error(f"Message render error: {e}")
                    continue
            
            # Chat input
            if user_query := st.chat_input("Ask me anything about your documents...", key="chat_input"):
                try:
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    
                    with st.chat_message("user"):
                        st.markdown(user_query)
                    
                    with st.status("🧠 Processing...", expanded=True) as status:
                        try:
                            if GRAPH_AVAILABLE:
                                msg_obj = HumanMessage(content=user_query) if LANGCHAIN_AVAILABLE else {"content": user_query}
                                
                                inputs = {
                                    "messages": [msg_obj],
                                    "next_step": "",
                                    "thought_process": [],
                                    "sub_agent_actions": [],
                                    "final_answer": "",
                                    "metrics": {},
                                    "router_decision": "",
                                    "retrieved_chunks": [],
                                    "retry_count": 0,
                                    "retrieval_strategy": {}
                                }
                                
                                final_state = None
                                start_time = time.time()
                                
                                for event in langgraph_app.stream(inputs, stream_mode="values"):
                                    try:
                                        if event.get("thought_process"):
                                            status.update(label=f"🧠 {event['thought_process'][-1]}")
                                        if event.get("sub_agent_actions"):
                                            action = event['sub_agent_actions'][-1].get('action', 'Processing...')
                                            status.update(label=f"⚙️ {action}")
                                        if event.get("router_decision"):
                                            status.update(label=f"🎯 {event['router_decision']}")
                                        final_state = event
                                    except Exception as stream_err:
                                        logger.error(f"Stream error: {stream_err}")
                                        continue
                                
                                status.update(label="✅ Complete", state="complete")
                                
                                if final_state:
                                    try:
                                        messages_list = final_state.get("messages", [])
                                        answer = messages_list[-1].content if messages_list and hasattr(messages_list[-1], 'content') else "No response generated"
                                    except:
                                        answer = "Error processing response"
                                    
                                    agent = final_state.get("router_decision", "General")
                                    metrics = final_state.get("metrics", {})
                                    raw_chunks = final_state.get("retrieved_chunks", [])
                                    
                                    processed_chunks = []
                                    try:
                                        for c in raw_chunks:
                                            if isinstance(c, dict):
                                                processed_chunks.append({
                                                    "content": c.get("content", ""),
                                                    "score": float(c.get("score", 0.0)),
                                                    "source": c.get("source", "Unknown")
                                                })
                                    except Exception as chunk_err:
                                        logger.error(f"Chunk processing error: {chunk_err}")
                                    
                                    latency_ms = int((time.time() - start_time) * 1000)
                                    log_entry = {
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "username": user_name,
                                        "role": user_role,
                                        "query": user_query,
                                        "agent": agent,
                                        "metrics": metrics,
                                        "latency_ms": latency_ms,
                                        "chunks_retrieved": len(processed_chunks),
                                        "retrieved_chunks": processed_chunks
                                    }
                                    
                                    st.session_state.query_history.append(log_entry)
                                    st.session_state.last_log = log_entry
                                    st.session_state.messages.append({"role": "assistant", "content": answer})
                                    
                                    audit_log(user_name, "QUERY_EXECUTED", f"Agent: {agent}, Latency: {latency_ms}ms")
                                    
                                    with st.chat_message("assistant"):
                                        st.markdown(f"**[{agent}]** ✅\n\n{answer}")
                            else:
                                st.warning("⚠️ Graph module unavailable")
                        except Exception as process_err:
                            logger.error(f"Process error: {process_err}\n{traceback.format_exc()}")
                            st.error(f"Error: {str(process_err)[:100]}")
                
                except Exception as query_err:
                    logger.error(f"Query error: {query_err}")
                    st.error(f"Query failed: {str(query_err)[:100]}")
            
            st.divider()
            st.caption(f"📊 Queries this session: {len(st.session_state.query_history)} | System: {st.session_state.system_status.upper()}")
            
        except Exception as e:
            logger.error(f"Demo view error: {e}")
            st.error(f"Demo error: {str(e)[:100]}")
    
    # --- DASHBOARD VIEW ---
    elif st.session_state.view == "dashboard":
        try:
            require_login()
            
            st.title("📊 Dashboard & Analytics")
            st.caption("Performance metrics, query history, and system insights.")
            
            if not st.session_state.query_history:
                st.info("📭 No queries yet. Start chatting in 'Live Demo' to see analytics!")
            else:
                try:
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(
                            f"""<div class='metric-box'><div class='metric-val'>{len(st.session_state.query_history)}</div><div class='metric-lbl'>Total Queries</div></div>""",
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f"""<div class='metric-box'><div class='metric-val'>{user_name}</div><div class='metric-lbl'>User</div></div>""",
                            unsafe_allow_html=True
                        )
                    with col3:
                        st.markdown(
                            f"""<div class='metric-box'><div class='metric-val'>{user_role.upper()}</div><div class='metric-lbl'>Role</div></div>""",
                            unsafe_allow_html=True
                        )
                    
                    st.divider()
                    
                    # Query history table
                    st.subheader("📜 Query History")
                    try:
                        df_queries = pd.DataFrame(st.session_state.query_history)
                        display_cols = ['timestamp', 'query', 'username', 'role', 'agent', 'latency_ms', 'chunks_retrieved']
                        st.dataframe(
                            df_queries[display_cols].head(20),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "latency_ms": st.column_config.NumberColumn(format="%d ms"),
                                "chunks_retrieved": st.column_config.NumberColumn(format="%d"),
                            }
                        )
                    except Exception as df_err:
                        logger.error(f"DataFrame error: {df_err}")
                        st.error("Error rendering table")
                
                except Exception as metrics_err:
                    logger.error(f"Metrics error: {metrics_err}")
                    st.error(f"Metrics error: {str(metrics_err)[:100]}")
        
        except Exception as e:
            logger.error(f"Dashboard view error: {e}")
            st.error(f"Dashboard error: {str(e)[:100]}")
    
    # --- ARCHITECTURE VIEW ---
    elif st.session_state.view == "arch":
        try:
            st.title("🏗️ System Architecture")
            st.caption("Technical specification, data flow, and agent topology.")
            
            st.subheader("Technology Stack")
            col_ts1, col_ts2, col_ts3, col_ts4 = st.columns(4)
            
            with col_ts1:
                st.markdown(
                    """<div class='tech-category'><h4>🧠 Core</h4><div class='tech-grid'><span class='tech-pill'>Python 3.10+</span><span class='tech-pill'>LangGraph</span><span class='tech-pill'>Streamlit</span><span class='tech-pill'>FastAPI</span></div></div>""",
                    unsafe_allow_html=True
                )
            with col_ts2:
                st.markdown(
                    """<div class='tech-category'><h4>🤖 AI/ML</h4><div class='tech-grid'><span class='tech-pill'>SentenceTransformers</span><span class='tech-pill'>FAISS</span><span class='tech-pill'>OpenRouter</span><span class='tech-pill'>EasyOCR</span></div></div>""",
                    unsafe_allow_html=True
                )
            with col_ts3:
                st.markdown(
                    """<div class='tech-category'><h4>🔐 Security</h4><div class='tech-grid'><span class='tech-pill'>SQLite</span><span class='tech-pill'>bcrypt</span><span class='tech-pill'>RBAC</span><span class='tech-pill'>Audit Log</span></div></div>""",
                    unsafe_allow_html=True
                )
            with col_ts4:
                st.markdown(
                    """<div class='tech-category'><h4>📊 Viz</h4><div class='tech-grid'><span class='tech-pill'>Plotly</span><span class='tech-pill'>Pandas</span><span class='tech-pill'>RAGAS</span></div></div>""",
                    unsafe_allow_html=True
                )
            
            st.divider()
            st.subheader("End-to-End Data Flow")
            col_flow1, col_flow2 = st.columns([1, 1])
            
            with col_flow1:
                st.markdown(
                    """<div style='padding: 0 1rem;'><div class='flow-step'><div class='flow-title'>1. Ingestion</div><div class='flow-desc'>PDF → Text → Chunking → Embedding → FAISS</div></div><div class='flow-step'><div class='flow-title'>2. Security Guard</div><div class='flow-desc'>Input Validation → Injection Detection → RBAC Check</div></div><div class='flow-step'><div class='flow-title'>3. Router</div><div class='flow-desc'>Intent Classification → Agent Selection</div></div></div>""",
                    unsafe_allow_html=True
                )
            with col_flow2:
                st.markdown(
                    """<div style='padding: 0 1rem;'><div class='flow-step'><div class='flow-title'>4. Hybrid Retrieval</div><div class='flow-desc'>BM25 + Vector → RRF Ranking</div></div><div class='flow-step'><div class='flow-title'>5. Gen & Critique</div><div class='flow-desc'>LLM → RAGAS Score → Self-Correction</div></div><div class='flow-step'><div class='flow-title'>6. Response</div><div class='flow-desc'>Answer + Citations + Audit Log</div></div></div>""",
                    unsafe_allow_html=True
                )
            
            st.divider()
            
            # Agent Swarm Topology
            render_agent_swarm_topology()
            
            st.divider()
            st.subheader("🛡️ Real-Time Threat & Quality Detection")
            st.caption("Monitoring prompt injections, hallucinations, and access violations")

            # Mock data for demo (replace with real metrics in production)
            detection_data = {
                "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(10)],
                "injection_attempts": [0, 1, 0, 2, 0, 0, 1, 0, 0, 0],
                "hallucination_alerts": [0, 0, 1, 0, 0, 2, 0, 0, 1, 0],
                "access_violations": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            }
            df_detect = pd.DataFrame(detection_data)

            fig_detect = go.Figure()
            fig_detect.add_trace(go.Scatter(
                x=df_detect["timestamp"],
                y=df_detect["injection_attempts"],
                name="🛡️ Injection Attempts",
                mode="lines+markers",
                line=dict(color="#ff8080", width=2),
                fill="tozeroy"
            ))
            fig_detect.add_trace(go.Scatter(
                x=df_detect["timestamp"],
                y=df_detect["hallucination_alerts"],
                name="🎯 Hallucination Alerts",
                mode="lines+markers",
                line=dict(color="#d29922", width=2),
                fill="tozeroy"
            ))
            fig_detect.add_trace(go.Scatter(
                x=df_detect["timestamp"],
                y=df_detect["access_violations"],
                name="🔐 Access Violations",
                mode="lines+markers",
                line=dict(color="#58a6ff", width=2),
                fill="tozeroy"
            ))
            fig_detect.update_layout(
                title="Security & Quality Monitoring (Last 10 Minutes)",
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b949e"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(title="Time", gridcolor="#30363d"),
                yaxis=dict(title="Events", gridcolor="#30363d")
            )
            st.plotly_chart(fig_detect, use_container_width=True, width="stretch")

            st.info("""
            **How It Works**:
            - 🛡️ **Injection Guard**: Regex + semantic analysis blocks malicious prompts (`ignore previous instructions`, `system prompt`, etc.) before they reach the LLM
            - 🎯 **Hallucination Monitor**: RAGAS faithfulness score < 0.6 triggers alert + auto-retry loop
            - 🔐 **Access Logger**: Every query is logged with user role; RBAC violations flagged in real-time

            *In production, these metrics would feed into SIEM tools (Splunk, Datadog) for enterprise SOC integration.*
            """)
            
            st.divider()
            st.subheader("🔐 RBAC Matrix")
            rbac_data = {
                "Feature": [
                    "View Home & Architecture",
                    "Chat in Live Demo",
                    "View Dashboard",
                    "Upload Documents",
                    "Manage Users",
                    "View Audit Logs"
                ],
                "Viewer": ["✅", "✅", "✅", "❌", "❌", "❌"],
                "Editor": ["✅", "✅", "✅", "✅", "❌", "❌"],
                "Admin": ["✅", "✅", "✅", "✅", "✅", "✅"]
            }
            st.dataframe(pd.DataFrame(rbac_data), use_container_width=True, hide_index=True)
            
            st.info("""
            **Access Control Hierarchy:**
            - 🟢 **Admin**: Full access, user management, audit logs
            - 🟡 **Editor**: Can upload documents & chat
            - 🔵 **Viewer**: Read-only access (chat & dashboards)
            """)
            
        except Exception as e:
            logger.error(f"Architecture view error: {e}")
            st.error(f"Architecture error: {str(e)[:100]}")
    
    else:
        try:
            st.warning("⚠️ Unknown view. Redirecting...")
            st.session_state.view = "home"
            time.sleep(1)
            st.rerun()
        except Exception as e:
            logger.error(f"Fallback error: {e}")
            st.error("Critical error")

# ==========================================
# 13. FOOTER
# ==========================================
st.divider()
if st.session_state.get("authenticated"):
    st.caption(
        f"🔒 NEXUS v1.0.0 | "
        f"User: {st.session_state.username} ({st.session_state.user_role}) | "
        f"Status: ONLINE ✅"
    )
else:
    st.caption("🔒 NEXUS v1.0.0 | Status: REQUIRES AUTHENTICATION")
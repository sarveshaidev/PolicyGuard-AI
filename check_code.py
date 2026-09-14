#!/usr/bin/env python3
"""
PolicyGuardAI — Deep READ-ONLY Production Readiness Checker

Run from the project root:
    python check_code.py
    python check_code.py --deep
    python check_code.py --deep --models
    python check_code.py --json

This checker NEVER repairs or rewrites project files. It only inspects and
reports. A PASS is evidence that a check found the expected signal; it is not
a guarantee of production readiness.

Checks:
- project structure
- Python syntax / compile
- dependencies / requirements
- hard-coded secrets and weak defaults
- authentication / registration / password hashing
- RBAC / admin controls / tenant isolation
- database, migration, PKL, FAISS and persistence signals
- embedder / vector store / cache / RAG / Graph / memory / Talent integration
- guardrails / prompt-injection / file validation / path safety
- Streamlit / deployment entrypoints
- tests
- optional runtime imports
- optional live embedding + LLM smoke tests
- Git / filesystem deployment hygiene

Exit code:
    0 = no FAIL findings
    1 = FAIL findings
    2 = checker itself failed
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDE = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "node_modules",
    "dist", "build"
}
TEXT_EXTS = {
    ".py", ".txt", ".md", ".toml", ".ini", ".cfg", ".yaml", ".yml", ".json"
}
BINARY_EXTS = {
    ".db", ".sqlite", ".sqlite3", ".pkl", ".pickle", ".faiss", ".index",
    ".bin", ".pt", ".pth", ".safetensors", ".onnx"
}

@dataclass
class Finding:
    severity: str
    area: str
    message: str
    evidence: str = ""
    recommendation: str = ""

FINDINGS: list[Finding] = []
COUNTS = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}


def report(severity, area, message, evidence="", recommendation=""):
    severity = severity.upper()
    FINDINGS.append(Finding(severity, area, message, evidence, recommendation))
    COUNTS.setdefault(severity, 0)
    COUNTS[severity] += 1
    icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗", "INFO": "•"}.get(severity, "•")
    print(f"[{icon}] {area}: {message}")
    if evidence:
        print(f"    Evidence: {evidence[:700]}")
    if recommendation:
        print(f"    Action: {recommendation}")


def section(name):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)


def files():
    for root, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE]
        for name in names:
            p = Path(root) / name
            if p.is_file():
                yield p


def rel(p):
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def text(p):
    try:
        if p.stat().st_size > 20 * 1024 * 1024:
            return ""
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def first_existing(candidates):
    for candidate in candidates:
        p = ROOT / candidate
        if p.exists():
            return p
    return None


def has_any(source, patterns):
    return any(re.search(p, source, re.I | re.S) for p in patterns)


def run(cmd, timeout=120):
    return subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout
    )


# ---------------------------------------------------------------------------
# 1. Structure
# ---------------------------------------------------------------------------

def check_structure():
    section("1. PROJECT STRUCTURE")

    for d in ["src", "src/core", "data"]:
        p = ROOT / d
        if p.is_dir():
            report("PASS", "Structure", f"Directory exists: {d}")
        else:
            report("WARN", "Structure", f"Directory missing: {d}")

    for f in ["app.py", "requirements.txt"]:
        p = ROOT / f
        if p.is_file():
            report("PASS", "Structure", f"Required file exists: {f}")
        else:
            report("FAIL", "Structure", f"Required file missing: {f}",
                   recommendation=f"Restore {f} before deployment.")

    app = ROOT / "app.py"
    if app.exists():
        s = text(app)
        report("INFO", "Structure",
               f"app.py = {len(s.splitlines()):,} lines / {app.stat().st_size:,} bytes")


# ---------------------------------------------------------------------------
# 2. Syntax / AST
# ---------------------------------------------------------------------------

def check_syntax():
    section("2. PYTHON SYNTAX / AST")

    py = [p for p in files() if p.suffix == ".py"]
    if not py:
        report("FAIL", "Syntax", "No Python files found.")
        return

    failures = 0
    for p in py:
        try:
            source = text(p)
            if not source:
                report("WARN", "Syntax", f"Could not read: {rel(p)}")
                continue
            tree = ast.parse(source, filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = ""
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    if name in {"eval", "exec"}:
                        report("WARN", "AST",
                               f"{name}() detected in {rel(p)} at line {getattr(node,'lineno','?')}",
                               recommendation="Never pass untrusted input to dynamic execution.")
            report("PASS", "Syntax", f"AST parse OK: {rel(p)}")
        except SyntaxError as e:
            failures += 1
            report("FAIL", "Syntax", f"Syntax error: {rel(p)}",
                   f"line {e.lineno}: {e.msg}",
                   "Fix syntax before deployment.")
    try:
        r = run([sys.executable, "-m", "compileall", "-q",
                 str(ROOT / "src"), str(ROOT / "app.py")], 120)
        if r.returncode == 0:
            report("PASS", "Compile", "compileall completed successfully.")
        else:
            report("FAIL", "Compile", "compileall failed",
                   (r.stderr or r.stdout)[-2500:],
                   "Fix compilation failures.")
    except Exception as e:
        report("WARN", "Compile", f"compileall could not run: {e}")


# ---------------------------------------------------------------------------
# 3. Dependencies
# ---------------------------------------------------------------------------

def check_dependencies():
    section("3. DEPENDENCIES")

    req = ROOT / "requirements.txt"
    if not req.exists():
        report("FAIL", "Dependencies", "requirements.txt is missing.")
        return

    lines = [
        x.strip() for x in text(req).splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]
    report("INFO", "Dependencies", f"{len(lines)} active requirements found.")

    imports = {
        "streamlit": "streamlit",
        "fastapi": "fastapi",
        "numpy": "numpy",
        "pandas": "pandas",
        "pydantic": "pydantic",
        "faiss": "faiss",
        "sentence-transformers": "sentence_transformers",
        "torch": "torch",
    }
    for label, module in imports.items():
        try:
            ok = importlib.util.find_spec(module) is not None
        except Exception:
            ok = False
        if ok:
            report("PASS", "Dependencies", f"Available: {label}")
        else:
            report("WARN", "Dependencies",
                   f"Not detected in current environment: {label}",
                   "Install/verify production environment dependencies.")

    unpinned = [
        x for x in lines
        if not x.startswith(("-", "--")) and not re.search(r"(==|~=|>=|<=|>|<)", x)
    ]
    if unpinned:
        report("WARN", "Dependencies",
               f"{len(unpinned)} requirement(s) have no version constraint.",
               ", ".join(unpinned[:15]),
               "Pin production dependencies after validation.")
    else:
        report("PASS", "Dependencies", "Requirements have version constraints.")


# ---------------------------------------------------------------------------
# 4. Security / guardrails
# ---------------------------------------------------------------------------

def check_security():
    section("4. SECURITY / GUARDRAILS / SECRETS")

    source_files = [p for p in files() if p.suffix in TEXT_EXTS or p.name == ".gitignore"]
    content = [(p, text(p)) for p in source_files]

    secret_patterns = [
        r'(?i)\b(password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*["\'][^$%{][^"\']{6,}["\']',
        r'(?i)\bsk-[A-Za-z0-9_-]{20,}\b',
        r'(?i)\bAKIA[0-9A-Z]{16}\b',
        r'(?i)\bgh[pousr]_[A-Za-z0-9_]{20,}\b',
    ]
    weak = [r"(?i)\badmin123\b", r"(?i)\bpassword123\b", r"(?i)\bchangeme\b"]

    hits = [rel(p) for p, s in content if any(re.search(x, s) for x in secret_patterns)]
    weak_hits = [rel(p) for p, s in content if any(re.search(x, s) for x in weak)]

    if hits:
        report("FAIL", "Secrets", "Potential hard-coded secret detected.",
               ", ".join(sorted(set(hits))[:20]),
               "Move secrets to deployment secret storage and rotate exposed credentials.")
    else:
        report("PASS", "Secrets", "No obvious hard-coded secret pattern found.")

    if weak_hits:
        report("FAIL", "Credentials", "Weak/default credential literal detected.",
               ", ".join(sorted(set(weak_hits))[:20]),
               "Remove default credentials and force secure provisioning.")
    else:
        report("PASS", "Credentials", "No common weak default credential detected.")

    joined = "\n".join(s for _, s in content)
    controls = {
        "password hashing": [r"\bbcrypt\b", r"\bargon2\b", r"passlib", r"hash_password"],
        "RBAC": [r"\bRBAC\b", r"role_based", r"user_role", r"require_role"],
        "tenant isolation": [r"organization_id", r"tenant_id"],
        "audit logging": [r"audit_log", r"audit logging", r"AuditLog"],
        "security guard": [r"security_guard", r"SecurityGuard", r"guardrail"],
        "prompt injection defense": [r"prompt injection", r"jailbreak", r"injection"],
        "file validation": [r"file size", r"content.?type", r"mime", r"extension"],
        "path safety": [r"is_relative_to", r"resolve\(", r"path traversal"],
        "rate limiting": [r"rate.?limit", r"throttl"],
    }
    for name, patterns in controls.items():
        if has_any(joined, patterns):
            report("PASS", "Guardrails", f"{name}: implementation signal detected.")
        else:
            sev = "FAIL" if name in {"password hashing", "RBAC"} else "WARN"
            report(sev, "Guardrails", f"{name}: no clear signal detected.",
                   recommendation="Verify this control is enforced server-side.")

    for pattern, label in [
        (r"verify\s*=\s*False", "TLS verification disabled"),
        (r"ssl\s*=\s*False", "SSL disabled"),
        (r"subprocess\.[^(]+\([^)]*shell\s*=\s*True", "shell=True subprocess"),
        (r"os\.system\(", "os.system"),
        (r"pickle\.loads?\(", "pickle deserialization"),
    ]:
        for p, s in content:
            if re.search(pattern, s, re.I):
                report("WARN", "Unsafe Pattern", f"{label} in {rel(p)}",
                       recommendation="Review carefully; never trust unvalidated external input.")


# ---------------------------------------------------------------------------
# 5. Configuration
# ---------------------------------------------------------------------------

def check_config():
    section("5. CONFIGURATION / PRODUCTION SECRETS")

    settings = [p for p in files() if p.name == "settings.py"]
    if settings:
        report("PASS", "Config", "settings.py detected.")
    else:
        report("WARN", "Config", "No settings.py detected.")

    env = os.environ
    if env.get("SECRET_KEY"):
        if len(env["SECRET_KEY"]) >= 32:
            report("PASS", "Config", "SECRET_KEY exists with >=32 characters.")
        else:
            report("FAIL", "Config", "SECRET_KEY exists but is too short.",
                   recommendation="Use a long random production secret.")
    else:
        report("WARN", "Config", "SECRET_KEY is absent from this shell.",
               recommendation="Configure it in the hosting platform secret manager.")

    joined = "\n".join(text(p) for p in settings)
    if joined:
        if re.search(r"SECRET_KEY\s*=\s*[\"'][^$%{][^\"']+[\"']", joined, re.I):
            report("FAIL", "Config", "Literal SECRET_KEY default found in settings.")
        if "LANGSMITH_TRACING" in joined and "ENABLE_TRACING" in joined:
            report("INFO", "Config", "Both tracing configuration names are referenced.")


# ---------------------------------------------------------------------------
# 6. Authentication / registration
# ---------------------------------------------------------------------------

def check_auth():
    section("6. AUTHENTICATION / REGISTRATION / RBAC")

    candidates = [
        "src/database.py", "src/core/database.py", "database.py",
        "src/auth/database.py", "src/auth/auth.py"
    ]
    paths = [ROOT / x for x in candidates if (ROOT / x).exists()]
    paths += [
        p for p in files()
        if p.suffix == ".py" and any(k in p.name.lower() for k in ("auth", "database", "user"))
        and p not in paths
    ]

    if not paths:
        report("FAIL", "Auth", "Authentication/database implementation not located.")
        return

    joined = "\n".join(text(p) for p in paths)
    checks = {
        "registration": [r"register", r"registration", r"create_user"],
        "login": [r"\blogin\b", r"authenticate", r"verify_password", r"check_password"],
        "password hashing": [r"bcrypt", r"argon2", r"passlib", r"hash_password"],
        "roles": [r"user_role", r"\bviewer\b", r"\beditor\b", r"\badmin\b"],
        "admin authorization": [r"require.*admin", r"is_admin", r"authorize"],
        "activation": [r"active", r"deactiv"],
        "last admin protection": [r"last.*admin", r"active.*admin", r"cannot.*admin"],
    }
    for name, patterns in checks.items():
        if has_any(joined, patterns):
            report("PASS", "Auth", f"{name}: detected.")
        else:
            sev = "FAIL" if name in {"registration", "login", "password hashing", "roles"} else "WARN"
            report(sev, "Auth", f"{name}: not clearly detected.",
                   recommendation="Manually verify this behavior before public hosting.")

    if re.search(r"register.*role.*admin", joined, re.I | re.S):
        report("FAIL", "Auth",
               "Registration appears to allow direct admin role assignment.",
               recommendation="Public registration must not permit privilege escalation.")


# ---------------------------------------------------------------------------
# 7. Persistence / database / artifacts
# ---------------------------------------------------------------------------

def check_persistence():
    section("7. DATABASE / PKL / FAISS / PERSISTENCE")

    artifacts = []
    for p in files():
        if p.suffix.lower() in BINARY_EXTS:
            try:
                artifacts.append(f"{rel(p)} ({p.stat().st_size:,} bytes)")
            except Exception:
                pass

    report("INFO", "Artifacts", f"Detected {len(artifacts)} database/model/index artifacts.")
    if artifacts:
        print("    " + "\n    ".join(artifacts[:25]))

    py = "\n".join(text(p) for p in files() if p.suffix == ".py")
    for name, patterns in {
        "SQLite": [r"sqlite3", r"SQLite"],
        "WAL": [r"journal_mode\s*=\s*WAL", r"\bWAL\b"],
        "atomic writes": [r"os\.replace", r"atomic", r"tempfile"],
        "backup/rollback": [r"backup", r"rollback"],
        "schema migration": [r"migration", r"ALTER TABLE", r"schema_version"],
    }.items():
        if has_any(py, patterns):
            report("PASS", "Persistence", f"{name}: detected.")
        else:
            report("WARN", "Persistence", f"{name}: not obvious from static scan.")


# ---------------------------------------------------------------------------
# 8. Architecture integration
# ---------------------------------------------------------------------------

def check_architecture():
    section("8. AI / RAG / GRAPH / CACHE / MEMORY / TALENT")

    py = [(p, text(p)) for p in files() if p.suffix == ".py"]
    joined = "\n".join(s for _, s in py)

    features = {
        "FAISS": [r"\bfaiss\b", r"IndexFlat"],
        "BM25": [r"\bBM25\b", r"rank_bm25"],
        "cross-encoder reranking": [r"CrossEncoder", r"cross.?encoder"],
        "embeddings": [r"encode_documents", r"encode_query", r"SentenceTransformer"],
        "semantic cache": [r"SemanticCache", r"semantic cache", r"get_cache"],
        "LangGraph": [r"LangGraph", r"StateGraph", r"process_query_via_graph"],
        "persistent memory": [r"memory_manager", r"persistent memory", r"chat_sessions"],
        "Talent Intelligence": [r"Talent Intelligence", r"talent_", r"candidate", r"resume"],
        "JD matching": [r"job description", r"_score_candidates", r"match talent"],
        "Policy Intelligence": [r"Policy Intelligence", r"policy namespace", r"policy documents"],
        "organization scoping": [r"organization_id", r"tenant_id"],
        "namespace isolation": [r"namespace"],
    }
    for name, patterns in features.items():
        if has_any(joined, patterns):
            report("PASS", "Architecture", f"{name}: detected.")
        else:
            report("WARN", "Architecture", f"{name}: not clearly detected.",
                   recommendation="Verify the feature is wired into the actual runtime path.")

    graph = first_existing(["src/core/graph.py", "graph.py"])
    rag = first_existing(["src/core/rag_engine.py", "rag_engine.py"])
    vector = first_existing(["src/core/vector_store.py", "vector_store.py"])
    cache = first_existing(["src/core/cache.py", "cache.py"])
    embed = first_existing(["src/core/embedder_singleton.py", "embedder_singleton.py"])

    if graph and rag:
        g, r = text(graph), text(rag)
        if "retrieved_chunks" in g and ("pre_retrieved_chunks" in r or "retrieved_chunks" in r):
            report("PASS", "Integration", "Graph → RAG context handoff signal detected.")
        else:
            report("WARN", "Integration", "Graph/RAG context handoff needs runtime verification.")

    if vector and embed:
        if "organization_id" in text(vector) and "encode_documents" in text(embed):
            report("PASS", "Integration", "Tenant-aware vector store + modern embedder API detected.")
        else:
            report("WARN", "Integration", "Vector store/embedder contract needs runtime verification.")

    if cache and embed and "get_embedder" in text(cache):
        report("PASS", "Integration", "Cache uses runtime embedder getter signal.")
    elif cache:
        report("WARN", "Integration", "Cache/embedder runtime integration should be tested.")


# ---------------------------------------------------------------------------
# 9. Tenant isolation / Talent security
# ---------------------------------------------------------------------------

def check_isolation():
    section("9. TENANT ISOLATION / TALENT DATA SECURITY")

    py = "\n".join(text(p) for p in files() if p.suffix == ".py")
    for name, pattern in {
        "organization_id": r"\borganization_id\b",
        "server-side organization filtering": r"(filter|WHERE|query).{0,120}organization_id|organization_id.{0,120}(filter|WHERE|query)",
        "namespace": r"\bnamespace\b",
        "talent authorization": r"(talent|candidate|resume).{0,200}(role|authorized|organization|tenant)",
        "access-controlled route": r"AccessControlled",
    }.items():
        if re.search(pattern, py, re.I | re.S):
            report("PASS", "Isolation", f"{name}: detected.")
        else:
            sev = "FAIL" if name == "organization_id" else "WARN"
            report(sev, "Isolation", f"{name}: not clearly detected.",
                   recommendation="Never rely only on UI filtering; enforce authorization in backend queries.")


# ---------------------------------------------------------------------------
# 10. Deployment / tests
# ---------------------------------------------------------------------------

def check_deployment(deep):
    section("10. DEPLOYMENT / TESTS")

    app = ROOT / "app.py"
    if app.exists() and re.search(r"\bst\.", text(app)):
        report("PASS", "Entrypoint", "Streamlit app detected.")
    else:
        report("WARN", "Entrypoint", "Streamlit entrypoint not obvious.")

    if any(p.name.startswith("test_") and p.suffix == ".py" for p in files()):
        report("PASS", "Tests", "Python test files detected.")
    else:
        report("WARN", "Tests", "No conventional test_*.py files detected.",
               recommendation="Add automated tests before public hosting.")

    if deep:
        try:
            r = run([sys.executable, "-m", "pytest", "-q", "--disable-warnings", "--maxfail=1"], 180)
            if r.returncode == 0:
                report("PASS", "Tests", "pytest passed.", (r.stdout or "")[-1800:])
            elif r.returncode == 5:
                report("WARN", "Tests", "pytest ran but found no tests.")
            else:
                report("FAIL", "Tests", "pytest reported failures.",
                       (r.stdout + "\n" + r.stderr)[-3500:],
                       "Fix failing tests before deployment.")
        except FileNotFoundError:
            report("WARN", "Tests", "pytest is not installed.")
        except subprocess.TimeoutExpired:
            report("FAIL", "Tests", "pytest timed out.")
        except Exception as e:
            report("WARN", "Tests", f"pytest could not run: {e}")


# ---------------------------------------------------------------------------
# 11. Runtime imports
# ---------------------------------------------------------------------------

def check_runtime_imports(deep):
    section("11. RUNTIME IMPORT SMOKE TEST")

    if not deep:
        report("INFO", "Runtime", "Skipped. Run with --deep.")
        return

    modules = [
        "src/core/embedder_singleton.py",
        "src/core/vector_store.py",
        "src/core/rag_engine.py",
        "src/core/graph.py",
        "src/core/cache.py",
        "src/core/memory_manager.py",
    ]
    for relpath in modules:
        p = ROOT / relpath
        if not p.exists():
            report("WARN", "Runtime", f"Missing: {relpath}")
            continue

        code = (
            "import sys,importlib.util;"
            f"sys.path.insert(0,{str(ROOT)!r});"
            f"sys.path.insert(0,{str(ROOT/'src')!r});"
            f"spec=importlib.util.spec_from_file_location('smoke',{str(p)!r});"
            "m=importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(m);"
            "print('IMPORT_OK')"
        )
        try:
            r = run([sys.executable, "-c", code], 90)
            if r.returncode == 0 and "IMPORT_OK" in r.stdout:
                report("PASS", "Runtime", f"Import OK: {relpath}")
            else:
                report("FAIL", "Runtime", f"Import failed: {relpath}",
                       (r.stderr or r.stdout)[-2500:],
                       "Fix import-time failures.")
        except subprocess.TimeoutExpired:
            report("FAIL", "Runtime", f"Import timed out: {relpath}")


# ---------------------------------------------------------------------------
# 12. Optional live models
# ---------------------------------------------------------------------------

def check_models(enabled):
    section("12. EMBEDDING / LLM LIVE SMOKE TEST")

    if not enabled:
        report("INFO", "Models", "Skipped. Run with --models; this may use network/API quota.")
        return

    embed = first_existing(["src/core/embedder_singleton.py", "embedder_singleton.py"])
    if embed:
        code = (
            "import sys;"
            f"sys.path.insert(0,{str(ROOT)!r});"
            f"sys.path.insert(0,{str(ROOT/'src')!r});"
            "from src.core.embedder_singleton import get_embedder;"
            "e=get_embedder();"
            "v=e.encode_query('PolicyGuardAI production smoke test');"
            "print('EMBED_OK',getattr(v,'shape',None))"
        )
        try:
            r = run([sys.executable, "-c", code], 300)
            if r.returncode == 0 and "EMBED_OK" in r.stdout:
                report("PASS", "Embedding", "Embedding model produced a vector.", r.stdout[-800:])
            else:
                report("FAIL", "Embedding", "Embedding model smoke test failed.",
                       (r.stderr or r.stdout)[-3000:],
                       "Verify model, HF access, dependencies and model cache.")
        except subprocess.TimeoutExpired:
            report("FAIL", "Embedding", "Embedding smoke test timed out.")

    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        report("WARN", "LLM", "No OPENROUTER_API_KEY/OPENAI_API_KEY in current shell.",
               recommendation="Configure production LLM secrets and rerun --models.")
        return

    try:
        import requests
        base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        model = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL")
        if not model:
            report("WARN", "LLM", "No LLM model variable configured; skipping live provider call.")
            return

        r = requests.post(
            base + "/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply exactly: POLICYGUARD_SMOKE_OK"}],
                "temperature": 0,
                "max_tokens": 20,
            },
            timeout=30,
        )
        if r.ok:
            report("PASS", "LLM", f"LLM responded successfully: {model}")
        else:
            report("FAIL", "LLM", f"LLM returned HTTP {r.status_code}.",
                   r.text[:2000],
                   "Verify credentials, model, quotas and outbound network.")
    except ImportError:
        report("WARN", "LLM", "requests is not installed; live LLM check skipped.")
    except Exception as e:
        report("FAIL", "LLM", f"Live LLM check failed: {e}")


# ---------------------------------------------------------------------------
# 13. Filesystem / Git hygiene
# ---------------------------------------------------------------------------

def check_filesystem():
    section("13. FILESYSTEM / GIT HYGIENE")

    gi = ROOT / ".gitignore"
    if not gi.exists():
        report("WARN", "Git", ".gitignore is missing.",
               recommendation="Exclude .env, secrets, caches, generated DB/index/model artifacts as appropriate.")
    else:
        g = text(gi).lower()
        missing = [x for x in [".env", "__pycache__", "*.pyc"] if x not in g]
        if missing:
            report("WARN", "Git", "Common ignore rules are not obvious.", ", ".join(missing))
        else:
            report("PASS", "Git", ".gitignore contains common secret/cache protections.")

    envs = [rel(p) for p in files() if p.name.lower() in {".env", ".env.local", ".env.production"}]
    if envs:
        report("WARN", "Filesystem", "Environment files exist in project tree.",
               ", ".join(envs),
               "Ensure they are not committed or included in public deployment.")
    else:
        report("PASS", "Filesystem", "No .env files detected.")

    huge = []
    for p in files():
        try:
            if p.stat().st_size > 250 * 1024 * 1024:
                huge.append(f"{rel(p)} ({p.stat().st_size/1024/1024:.1f} MB)")
        except Exception:
            pass
    if huge:
        report("WARN", "Filesystem", "Very large artifacts detected.", ", ".join(huge))
    else:
        report("PASS", "Filesystem", "No file larger than 250 MB detected.")


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def final_report(as_json=False):
    section("FINAL READINESS RESULT")
    print(f"PASS : {COUNTS['PASS']}")
    print(f"WARN : {COUNTS['WARN']}")
    print(f"FAIL : {COUNTS['FAIL']}")
    print(f"INFO : {COUNTS['INFO']}")

    if COUNTS["FAIL"]:
        status = "NOT READY FOR PUBLIC HOSTING"
    elif COUNTS["WARN"]:
        status = "CONDITIONAL — MANUAL VALIDATION REQUIRED"
    else:
        status = "STATIC CHECKS CLEAN"

    print(f"\nSTATUS: {status}")
    print("\nPriority findings:")
    for f in FINDINGS:
        if f.severity in {"FAIL", "WARN"}:
            print(f"- [{f.severity}] {f.area}: {f.message}")
            if f.recommendation:
                print(f"  -> {f.recommendation}")

    print("\nThis checker is READ-ONLY. It does not repair code or data.")
    print("For hosting, a clean static report must still be followed by real")
    print("registration/login, persistence, two-user/two-organization, upload,")
    print("RAG, Talent, LLM, embedding, restart, and production deployment tests.")

    if as_json:
        payload = {
            "project_root": str(ROOT),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": status,
            "counts": COUNTS,
            "findings": [asdict(x) for x in FINDINGS],
        }
        print("\nJSON_REPORT_BEGIN")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("JSON_REPORT_END")

    return 1 if COUNTS["FAIL"] else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true",
                        help="Run pytest and isolated runtime import smoke tests.")
    parser.add_argument("--models", action="store_true",
                        help="Run live embedding/LLM checks; may use network/API quota.")
    parser.add_argument("--json", action="store_true",
                        help="Print machine-readable JSON after the report.")
    args = parser.parse_args()

    print("PolicyGuardAI — Deep READ-ONLY Production Readiness Checker")
    print(f"Project root: {ROOT}")
    print("NO SOURCE, CONFIG, DATABASE, PKL, FAISS OR MODEL FILE WILL BE MODIFIED.")

    try:
        check_structure()
        check_syntax()
        check_dependencies()
        check_security()
        check_config()
        check_auth()
        check_persistence()
        check_architecture()
        check_isolation()
        check_deployment(args.deep)
        check_runtime_imports(args.deep)
        check_models(args.models)
        check_filesystem()
        return final_report(args.json)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 2
    except Exception as e:
        print("\nCHECKER INTERNAL ERROR")
        print(e)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

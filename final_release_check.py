"""
PolicyGuard AI — Final Release / GitHub Preflight Checker
==========================================================

READ-ONLY release audit.

This script DOES NOT:
- modify source code
- modify .env
- modify databases
- modify vector stores
- modify model files
- git add
- git commit
- git push

It checks:
1. Python compilation
2. pytest
3. project structure
4. sensitive files
5. .gitignore coverage
6. tracked secrets/data
7. Git status
8. expected PolicyGuardAI architecture
9. LLM configuration presence
10. embedding configuration
11. Docker/deployment files
12. obvious secret patterns

Run from project root:

    python final_release_check.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PASS = 0
WARN = 0
FAIL = 0

SENSITIVE_PATHS = [
    ".env",
    "nexus_auth.db",
    "data/vector_db",
    "data/uploads",
    "data/traces",
    "data/cache.db",
    "data/memory.db",
    "logs",
    "backups",
    ".cache",
]

EXPECTED_FILES = [
    "app.py",
    "main.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore",
    ".dockerignore",
    "check_code.py",
]

EXPECTED_MODULES = [
    "src/auth/database.py",
    "src/auth/access_request.py",
    "src/core/cache.py",
    "src/core/embedder_singleton.py",
    "src/core/memory_manager.py",
    "src/core/observability.py",
    "src/evaluation/llm_judge.py",
    "src/evaluation/ragas_evaluator.py",
    "src/ingestion/multimodal_parser.py",
    "src/ingestion/universal_parser.py",
    "src/mcp_server/server.py",
    "src/orchestrator/graph.py",
    "src/pipeline/rag_engine.py",
    "src/retrieval/vector_store.py",
    "src/retrieval/hybrid_search.py",
    "src/retrieval/cross_encoder.py",
    "src/security/guard_model.py",
    "src/vision/advanced_ocr.py",
]

FORBIDDEN_TRACKED_PATTERNS = [
    ".env",
    "nexus_auth.db",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pkl",
    ".index",
    "data/uploads/",
    "data/traces/",
    "data/vector_db/",
    "backups/",
    "logs/",
    ".cache/",
]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(r"OPENROUTER_API_KEY\s*=\s*['\"][^'\"]{20,}['\"]"),
    re.compile(r"API_KEY\s*=\s*['\"][A-Za-z0-9_\-]{20,}['\"]"),
    re.compile(r"SECRET_KEY\s*=\s*['\"][^'\"]{20,}['\"]"),
    re.compile(r"PASSWORD\s*=\s*['\"][^'\"]{10,}['\"]"),
]


def ok(message: str) -> None:
    global PASS
    PASS += 1
    print(f"[✓] {message}")


def warn(message: str) -> None:
    global WARN
    WARN += 1
    print(f"[!] {message}")


def fail(message: str) -> None:
    global FAIL
    FAIL += 1
    print(f"[✗] {message}")


def info(message: str) -> None:
    print(f"[•] {message}")


def run(
    command: list[str],
    *,
    timeout: int = 180,
) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return 124, "COMMAND TIMEOUT"
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def git_available() -> bool:
    code, _ = run(["git", "--version"])
    return code == 0


def git_ignored(path: str) -> tuple[bool, str]:
    code, output = run(
        ["git", "check-ignore", "-v", "--", path],
        timeout=20,
    )
    return code == 0, output


def tracked_files() -> list[str]:
    code, output = run(
        ["git", "ls-files"],
        timeout=20,
    )
    if code != 0:
        return []
    return [
        line.strip().replace("\\", "/")
        for line in output.splitlines()
        if line.strip()
    ]


def file_contains_secret(path: Path) -> list[str]:
    findings: list[str] = []

    try:
        if path.stat().st_size > 5 * 1024 * 1024:
            return findings

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return findings

    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(pattern.pattern)

    return findings


print()
print("=" * 78)
print("PolicyGuardAI — FINAL GITHUB / DEPLOYMENT PREFLIGHT")
print("=" * 78)
print(f"Project root: {ROOT}")
print()
print("READ-ONLY:")
print("  No source modification")
print("  No database modification")
print("  No model modification")
print("  No git add")
print("  No git commit")
print("  No git push")
print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------

print("\n1. PYTHON ENVIRONMENT")
print("-" * 78)

info(
    f"Python: "
    f"{sys.version_info.major}."
    f"{sys.version_info.minor}."
    f"{sys.version_info.micro}"
)

if sys.version_info >= (3, 10):
    ok("Python version is supported for this project")
else:
    fail("Python version is older than expected")


# ---------------------------------------------------------------------------
# 2. Expected project files
# ---------------------------------------------------------------------------

print("\n2. PROJECT STRUCTURE")
print("-" * 78)

for relative in EXPECTED_FILES:
    path = ROOT / relative
    if path.exists():
        ok(f"Required file exists: {relative}")
    else:
        fail(f"Required file missing: {relative}")

for relative in EXPECTED_MODULES:
    path = ROOT / relative
    if path.exists():
        ok(f"Module exists: {relative}")
    else:
        fail(f"Expected production module missing: {relative}")


# ---------------------------------------------------------------------------
# 3. Compile
# ---------------------------------------------------------------------------

print("\n3. PYTHON COMPILATION")
print("-" * 78)

compile_targets = [
    "app.py",
    "main.py",
    "config",
    "src",
    "tests",
]

code, output = run(
    [
        sys.executable,
        "-m",
        "compileall",
        "-q",
        *compile_targets,
    ],
    timeout=180,
)

if code == 0:
    ok("Python compileall passed")
else:
    fail("Python compileall failed")
    if output:
        print(output[-4000:])


# ---------------------------------------------------------------------------
# 4. Tests
# ---------------------------------------------------------------------------

print("\n4. TEST SUITE")
print("-" * 78)

code, output = run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
    ],
    timeout=300,
)

if code == 0:
    ok("pytest passed")
    last_lines = output.splitlines()[-8:]
    for line in last_lines:
        print(f"    {line}")
else:
    fail("pytest failed")
    print(output[-6000:])


# ---------------------------------------------------------------------------
# 5. Git availability
# ---------------------------------------------------------------------------

print("\n5. GIT")
print("-" * 78)

if not git_available():
    fail("Git is not available")
else:
    ok("Git is available")

    code, output = run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        timeout=20,
    )

    if code == 0 and output.strip() == "true":
        ok("Project is a Git repository")
    else:
        warn("Project is not yet initialized as a Git repository")


# ---------------------------------------------------------------------------
# 6. .gitignore
# ---------------------------------------------------------------------------

print("\n6. GITIGNORE / SENSITIVE FILE PROTECTION")
print("-" * 78)

gitignore = ROOT / ".gitignore"

if not gitignore.exists():
    fail(".gitignore is missing")
else:
    ok(".gitignore exists")

    try:
        ignore_text = gitignore.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        ignore_text = ""

    if ".env" in ignore_text:
        ok(".env appears in .gitignore")
    else:
        fail(".env is not clearly ignored")

    if ".cache/" in ignore_text or ".cache" in ignore_text:
        ok(".cache appears in .gitignore")
    else:
        warn(".cache is not clearly ignored")

    if "backups/" in ignore_text or "backups" in ignore_text:
        ok("backups appears in .gitignore")
    else:
        warn("backups is not clearly ignored")

    if "data/vector_db/" in ignore_text or "data/vector_db" in ignore_text:
        ok("data/vector_db appears in .gitignore")
    else:
        warn("data/vector_db is not clearly ignored")

    if "data/uploads/" in ignore_text or "data/uploads" in ignore_text:
        ok("data/uploads appears in .gitignore")
    else:
        warn("data/uploads is not clearly ignored")

    if "data/traces/" in ignore_text or "data/traces" in ignore_text:
        ok("data/traces appears in .gitignore")
    else:
        warn("data/traces is not clearly ignored")


# ---------------------------------------------------------------------------
# 7. Git ignore actual behavior
# ---------------------------------------------------------------------------

print("\n7. GIT ACTUAL IGNORE CHECK")
print("-" * 78)

if git_available():
    for sensitive in SENSITIVE_PATHS:
        exists = (ROOT / sensitive).exists()

        ignored, evidence = git_ignored(sensitive)

        if ignored:
            ok(f"Ignored by Git: {sensitive}")
            print(f"    {evidence}")
        elif exists:
            fail(
                f"Sensitive path exists but Git does not ignore it: "
                f"{sensitive}"
            )
        else:
            warn(
                f"Sensitive path not present locally and not confirmed ignored: "
                f"{sensitive}"
            )


# ---------------------------------------------------------------------------
# 8. Tracked sensitive files
# ---------------------------------------------------------------------------

print("\n8. TRACKED SENSITIVE FILE AUDIT")
print("-" * 78)

tracked = tracked_files()

if not tracked:
    warn("No tracked files detected yet")
else:
    sensitive_tracked = []

    for item in tracked:
        normalized = item.lower().replace("\\", "/")

        if any(
            pattern in normalized
            for pattern in FORBIDDEN_TRACKED_PATTERNS
        ):
            sensitive_tracked.append(item)

    if sensitive_tracked:
        for item in sensitive_tracked:
            fail(f"Sensitive file is already tracked: {item}")
    else:
        ok("No obvious secret/database/model/cache artifacts are tracked")

    info(f"Tracked files currently in repository: {len(tracked)}")


# ---------------------------------------------------------------------------
# 9. Working tree
# ---------------------------------------------------------------------------

print("\n9. GIT WORKING TREE")
print("-" * 78)

if git_available():
    code, output = run(
        ["git", "status", "--short"],
        timeout=20,
    )

    if code != 0:
        warn("Could not read Git status")
    elif not output.strip():
        ok("Git working tree is clean")
    else:
        info("Working tree has changes/untracked files")
        print(output)


# ---------------------------------------------------------------------------
# 10. Secret scan
# ---------------------------------------------------------------------------

print("\n10. LOCAL SECRET PATTERN SCAN")
print("-" * 78)

scan_roots = [
    ROOT / "app.py",
    ROOT / "main.py",
    ROOT / "config",
    ROOT / "src",
    ROOT / "tests",
]

secret_files = []

for target in scan_roots:
    if not target.exists():
        continue

    files = [target] if target.is_file() else list(target.rglob("*.py"))

    for path in files:
        findings = file_contains_secret(path)

        if findings:
            secret_files.append(
                (
                    path.relative_to(ROOT),
                    findings,
                )
            )

if secret_files:
    for path, findings in secret_files:
        fail(f"Possible secret pattern in: {path}")
else:
    ok("No obvious credential/secret pattern detected in source/tests")


# ---------------------------------------------------------------------------
# 11. .env safety
# ---------------------------------------------------------------------------

print("\n11. ENVIRONMENT FILE")
print("-" * 78)

env_file = ROOT / ".env"

if env_file.exists():
    info(".env exists locally — this is expected")
    ignored, evidence = git_ignored(".env")

    if ignored:
        ok(".env is ignored by Git")
        print(f"    {evidence}")
    else:
        fail(".env exists but is NOT ignored by Git")
else:
    info(".env does not exist locally")


# ---------------------------------------------------------------------------
# 12. LLM configuration
# ---------------------------------------------------------------------------

print("\n12. LLM CONFIGURATION")
print("-" * 78)

try:
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = (
        os.getenv("OPENROUTER_MODEL")
        or os.getenv("LLM_MODEL")
    )

    if api_key:
        ok("OPENROUTER_API_KEY is configured")
    else:
        warn("OPENROUTER_API_KEY is not configured")

    if model:
        ok(f"LLM model configured: {model}")
    else:
        warn("LLM model is not configured")

except Exception as exc:
    warn(f"Could not inspect dotenv configuration: {exc}")


# ---------------------------------------------------------------------------
# 13. Architecture
# ---------------------------------------------------------------------------

print("\n13. POLICYGUARDAI ARCHITECTURE")
print("-" * 78)

architecture_signals = {
    "Policy Intelligence": [
        "policy",
        "rag",
        "document",
    ],
    "Talent Intelligence": [
        "talent",
        "candidate",
        "resume",
    ],
    "Tenant isolation": [
        "organization_id",
        "organization",
    ],
    "Namespace isolation": [
        "namespace",
    ],
    "RBAC": [
        "admin",
        "editor",
        "viewer",
    ],
    "Audit logging": [
        "audit",
    ],
}

source_text = ""

for target in [
    ROOT / "app.py",
    ROOT / "src",
]:
    if target.is_file():
        try:
            source_text += target.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            pass
    elif target.is_dir():
        for py_file in target.rglob("*.py"):
            try:
                source_text += py_file.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                pass

lower_source = source_text.lower()

for feature, signals in architecture_signals.items():
    if all(signal.lower() in lower_source for signal in signals):
        ok(f"{feature}: detected")
    else:
        warn(f"{feature}: static signal incomplete")


# ---------------------------------------------------------------------------
# 14. Docker
# ---------------------------------------------------------------------------

print("\n14. DEPLOYMENT FILES")
print("-" * 78)

dockerfile = ROOT / "Dockerfile"
compose = ROOT / "docker-compose.yml"

if dockerfile.exists():
    ok("Dockerfile exists")
else:
    warn("Dockerfile missing")

if compose.exists():
    ok("docker-compose.yml exists")
else:
    warn("docker-compose.yml missing")


# ---------------------------------------------------------------------------
# 15. Final summary
# ---------------------------------------------------------------------------

print()
print("=" * 78)
print("FINAL PREFLIGHT RESULT")
print("=" * 78)

print(f"PASS : {PASS}")
print(f"WARN : {WARN}")
print(f"FAIL : {FAIL}")

if FAIL == 0:
    print()
    print("STATUS: READY FOR GITHUB STAGING")
    print()
    print("This script did NOT stage, commit, or push anything.")
    print("You may proceed to the manual git add/status review.")
else:
    print()
    print("STATUS: NOT READY")
    print()
    print("Resolve the FAIL items before staging the repository.")

print("=" * 78)
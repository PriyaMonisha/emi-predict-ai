#!/bin/bash
# filename: .claude/hooks/pre-commit.sh
# purpose:  Block commits that violate EMI Predict AI code standards
# version:  1.0

echo "🔍 EMI Predict AI — pre-commit checks running..."
echo "================================================"

# ── 1. Type checking ──────────────────────────────────────────
echo "▶ [1/8] Type checking with mypy..."
mypy src/ --ignore-missing-imports --strict --exclude "notebooks/" 2>&1 | tail -5
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ BLOCKED: Type errors found."
    echo "   Fix with: mypy src/ --ignore-missing-imports --strict"
    exit 2
fi
echo "   ✅ Types OK"

# ── 2. Linting ────────────────────────────────────────────────
echo "▶ [2/8] Linting with ruff..."
ruff check src/ tests/ --quiet
if [ $? -ne 0 ]; then
    echo "❌ BLOCKED: Lint errors found."
    echo "   Fix with: ruff check src/ tests/ --fix"
    exit 2
fi
echo "   ✅ Lint OK"

# ── 3. Formatting ─────────────────────────────────────────────
echo "▶ [3/8] Format check with black..."
black --check src/ tests/ --quiet
if [ $? -ne 0 ]; then
    echo "❌ BLOCKED: Unformatted files found."
    echo "   Fix with: black src/ tests/"
    exit 2
fi
echo "   ✅ Format OK"

# ── 4. No print() in src/ ────────────────────────────────────
echo "▶ [4/8] Scanning for print() statements in src/..."
PRINT_FOUND=$(grep -rn "^\s*print(" src/ --include="*.py" | grep -v "__pycache__")
if [ -n "$PRINT_FOUND" ]; then
    echo "❌ BLOCKED: print() statements found. Use logging instead:"
    echo "$PRINT_FOUND"
    exit 2
fi
echo "   ✅ No print() found"

# ── 5. No hardcoded paths ─────────────────────────────────────
echo "▶ [5/8] Scanning for hardcoded paths..."
PATHS_FOUND=$(grep -rn '"data/' src/ --include="*.py" | grep -v "os.path.join\|#\|docstring")
if [ -n "$PATHS_FOUND" ]; then
    echo "⚠️  WARNING: Possible hardcoded paths (use os.path.join):"
    echo "$PATHS_FOUND"
    echo "   Continuing — but please fix these."
fi
echo "   ✅ Path check done"

# ── 6. No secrets ────────────────────────────────────────────
echo "▶ [6/8] Scanning for hardcoded secrets..."
SECRET_FOUND=$(grep -rn \
    "api_key\s*=\s*['\"][^'\"]\|password\s*=\s*['\"][^'\"]\|secret\s*=\s*['\"][^'\"]" \
    src/ --include="*.py" | grep -v "os.getenv\|os.environ\|config\|#")
if [ -n "$SECRET_FOUND" ]; then
    echo "❌ BLOCKED: Hardcoded secrets found:"
    echo "$SECRET_FOUND"
    exit 2
fi
echo "   ✅ No secrets found"

# ── 7. No model weights in commit ────────────────────────────
echo "▶ [7/8] Checking for model artifacts in commit..."
WEIGHTS=$(git diff --cached --name-only | \
    grep -E "\.(pkl|joblib|pt|pth|ckpt|h5|bin|safetensors)$")
if [ -n "$WEIGHTS" ]; then
    echo "❌ BLOCKED: Model weights staged for commit — add to .gitignore:"
    echo "$WEIGHTS"
    exit 2
fi
echo "   ✅ No model weights staged"

# ── 8. No raw data in commit ─────────────────────────────────
echo "▶ [8/8] Checking for raw data in commit..."
RAW_DATA=$(git diff --cached --name-only | grep "^data/raw/")
if [ -n "$RAW_DATA" ]; then
    echo "❌ BLOCKED: data/raw/ is READ ONLY — never commit raw data:"
    echo "$RAW_DATA"
    exit 2
fi
echo "   ✅ No raw data staged"

echo ""
echo "================================================"
echo "✅ All checks passed — commit allowed."
exit 0
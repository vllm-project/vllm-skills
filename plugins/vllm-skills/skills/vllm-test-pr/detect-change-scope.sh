#!/bin/bash
# Detect whether a vLLM fix is Python-only or touches compiled/build code.
# Usage: detect-change-scope.sh [base-ref]
#   base-ref: git ref to diff against (default: origin/main)
#
# Exit codes:
#   0 = Python-only (safe to overlay)
#   1 = Touches compiled code (needs full rebuild)
#   2 = Error (not in a git repo, bad ref, etc.)
#
# Output: JSON with fields:
#   scope: "python-only" | "compiled" | "error"
#   py_files: list of changed .py files under vllm/
#   compiled_files: list of changed compiled/build files
#   dep_files: list of changed dependency files
#   test_only: true if changes are only in tests/
#   warnings: list of warning strings

set -euo pipefail

# Default to upstream/main since origin/main on forks is typically stale.
BASE_REF="${1:-upstream/main}"

if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo '{"scope":"error","message":"Not inside a git repository"}'
    exit 2
fi

# If upstream/main is not available, try to set it up and fetch
if ! git rev-parse "$BASE_REF" &>/dev/null; then
    if [ "$BASE_REF" = "upstream/main" ]; then
        # Try to add upstream remote and fetch
        if ! git remote get-url upstream &>/dev/null; then
            git remote add upstream https://github.com/vllm-project/vllm.git 2>/dev/null || true
        fi
        git fetch upstream main --quiet 2>/dev/null || true
    fi
    if ! git rev-parse "$BASE_REF" &>/dev/null; then
        echo "{\"scope\":\"error\",\"message\":\"Unknown ref: $BASE_REF. Try: git fetch upstream main\"}"
        exit 2
    fi
fi

# Use merge-base to diff only the branch's own commits, not upstream changes.
MERGE_BASE=$(git merge-base HEAD "$BASE_REF" 2>/dev/null || echo "$BASE_REF")
CHANGED_FILES=$(git diff --name-only "$MERGE_BASE" HEAD)

if [ -z "$CHANGED_FILES" ]; then
    echo '{"scope":"error","message":"No changed files detected"}'
    exit 2
fi

# Classify files
PY_FILES=()
COMPILED_FILES=()
DEP_FILES=()
OTHER_FILES=()
WARNINGS=()

while IFS= read -r file; do
    case "$file" in
        # Compiled / native code
        csrc/*|*.cpp|*.cu|*.cuh|*.c|*.h|*.cc)
            COMPILED_FILES+=("$file") ;;
        # Build system
        setup.py|setup.cfg|pyproject.toml|CMakeLists.txt|Makefile|*.cmake)
            COMPILED_FILES+=("$file") ;;
        # Dependencies
        requirements*.txt|constraints*.txt)
            DEP_FILES+=("$file") ;;
        # Python under vllm/
        vllm/*.py|vllm/**/*.py)
            PY_FILES+=("$file") ;;
        # Python elsewhere (tests, benchmarks, examples, etc.)
        *.py)
            OTHER_FILES+=("$file") ;;
        # Everything else
        *)
            OTHER_FILES+=("$file") ;;
    esac
done <<< "$CHANGED_FILES"

# Build warnings
if [ ${#DEP_FILES[@]} -gt 0 ]; then
    WARNINGS+=("Fix touches dependency files; base image may be missing new packages")
fi

TEST_ONLY=false
if [ ${#PY_FILES[@]} -eq 0 ] && [ ${#COMPILED_FILES[@]} -eq 0 ]; then
    # Check if it's test-only
    ALL_PY=true
    ALL_TEST=true
    while IFS= read -r file; do
        case "$file" in
            *.py) ;;
            *) ALL_PY=false ;;
        esac
        case "$file" in
            tests/*|test_*) ;;
            *) ALL_TEST=false ;;
        esac
    done <<< "$CHANGED_FILES"
    if $ALL_PY && $ALL_TEST; then
        TEST_ONLY=true
        WARNINGS+=("Changes are test-only; there may be nothing to test at runtime")
    fi
fi

# Determine scope
if [ ${#COMPILED_FILES[@]} -gt 0 ]; then
    SCOPE="compiled"
    EXIT_CODE=1
else
    SCOPE="python-only"
    EXIT_CODE=0
fi

# Output JSON
json_array() {
    local arr=("$@")
    if [ ${#arr[@]} -eq 0 ]; then
        echo "[]"
        return
    fi
    printf '['
    for i in "${!arr[@]}"; do
        printf '"%s"' "${arr[$i]}"
        if [ $i -lt $((${#arr[@]} - 1)) ]; then
            printf ','
        fi
    done
    printf ']'
}

cat <<EOF
{
  "scope": "$SCOPE",
  "py_files": $(json_array "${PY_FILES[@]+"${PY_FILES[@]}"}"),
  "compiled_files": $(json_array "${COMPILED_FILES[@]+"${COMPILED_FILES[@]}"}"),
  "dep_files": $(json_array "${DEP_FILES[@]+"${DEP_FILES[@]}"}"),
  "test_only": $TEST_ONLY,
  "warnings": $(json_array "${WARNINGS[@]+"${WARNINGS[@]}"}")
}
EOF

exit $EXIT_CODE

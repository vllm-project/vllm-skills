#!/bin/bash
# Set up local development environment for vLLM testing.
# Usage: setup-local-dev.sh
#
# Run this from the root of your vLLM repo clone. It:
#   1. Creates the vllm-test-results/ directory structure
#   2. Ensures vllm-test-results/ and k8s manifests are in .git/info/exclude
#   3. Ensures the upstream remote exists and is fetched
#
# Idempotent — safe to run multiple times.

set -euo pipefail

# Must be in a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "ERROR: Not inside a git repository" >&2
    exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
EXCLUDE_FILE="$REPO_ROOT/.git/info/exclude"

echo "=== [1/3] Creating test results directory structure ==="
mkdir -p "$REPO_ROOT/vllm-test-results/issues"
mkdir -p "$REPO_ROOT/vllm-test-results/pull-requests"
echo "  Created vllm-test-results/{issues,pull-requests}/"

echo "=== [2/3] Ensuring git info exclude entries ==="
ENTRIES=(
    "vllm-test-results/"
    "k8s-test-*.yaml"
    "k8s-repro-*.yaml"
)
for entry in "${ENTRIES[@]}"; do
    if ! grep -qxF "$entry" "$EXCLUDE_FILE" 2>/dev/null; then
        echo "$entry" >> "$EXCLUDE_FILE"
        echo "  Added '$entry' to .git/info/exclude"
    else
        echo "  '$entry' already in .git/info/exclude"
    fi
done

echo "=== [3/3] Ensuring upstream remote ==="
if git remote get-url upstream &>/dev/null; then
    echo "  upstream remote already exists: $(git remote get-url upstream)"
else
    git remote add upstream https://github.com/vllm-project/vllm.git
    echo "  Added upstream remote: https://github.com/vllm-project/vllm.git"
fi
echo "  Fetching upstream tags..."
git fetch upstream --tags --quiet 2>/dev/null || git fetch upstream --tags --force --quiet 2>/dev/null || true
echo "  Done."

echo ""
echo "Local dev environment ready."

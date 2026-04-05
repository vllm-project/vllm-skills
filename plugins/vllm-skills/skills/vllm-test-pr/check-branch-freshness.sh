#!/bin/bash
# Check that the current branch is rebased on recent upstream main.
# Usage: check-branch-freshness.sh [max-commits-behind]
#   max-commits-behind: how many upstream commits behind is acceptable (default: 50)
#
# This script:
#   1. Ensures the "upstream" remote exists (adds it if not)
#   2. Fetches upstream/main
#   3. Checks how far behind the branch's merge-base is from upstream/main
#   4. Reports status and optionally suggests rebase
#
# Exit codes:
#   0 = branch is fresh (within threshold)
#   1 = branch is stale (behind by more than threshold)
#   2 = error
#
# Output: JSON with fields:
#   status: "fresh" | "stale" | "error"
#   merge_base: commit SHA of merge-base with upstream/main
#   upstream_head: current upstream/main SHA
#   commits_behind: how many commits upstream/main is ahead of merge-base
#   branch_commits: how many commits on the branch since merge-base
#   message: human-readable status

set -euo pipefail

MAX_BEHIND="${1:-50}"

if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo '{"status":"error","message":"Not inside a git repository"}'
    exit 2
fi

# Ensure upstream remote exists
if ! git remote get-url upstream &>/dev/null; then
    echo "Adding upstream remote..." >&2
    git remote add upstream https://github.com/vllm-project/vllm.git
fi

# Fetch upstream main
echo "Fetching upstream/main..." >&2
if ! git fetch upstream main --quiet 2>/dev/null; then
    echo '{"status":"error","message":"Failed to fetch upstream/main. Check network and remote URL."}'
    exit 2
fi

UPSTREAM_HEAD=$(git rev-parse upstream/main)
MERGE_BASE=$(git merge-base HEAD upstream/main 2>/dev/null || echo "")

if [ -z "$MERGE_BASE" ]; then
    echo '{"status":"error","message":"No common ancestor with upstream/main. Is this a vLLM fork?"}'
    exit 2
fi

COMMITS_BEHIND=$(git rev-list --count "$MERGE_BASE".."$UPSTREAM_HEAD")
BRANCH_COMMITS=$(git rev-list --count "$MERGE_BASE"..HEAD)
BRANCH_NAME=$(git branch --show-current)

if [ "$COMMITS_BEHIND" -le "$MAX_BEHIND" ]; then
    STATUS="fresh"
    MSG="Branch '$BRANCH_NAME' is $COMMITS_BEHIND commits behind upstream/main (threshold: $MAX_BEHIND). $BRANCH_COMMITS commit(s) on branch."
    EXIT=0
else
    STATUS="stale"
    MSG="Branch '$BRANCH_NAME' is $COMMITS_BEHIND commits behind upstream/main (threshold: $MAX_BEHIND). Rebase recommended: git rebase upstream/main"
    EXIT=1
fi

cat <<EOF
{
  "status": "$STATUS",
  "merge_base": "$MERGE_BASE",
  "upstream_head": "$UPSTREAM_HEAD",
  "commits_behind": $COMMITS_BEHIND,
  "branch_commits": $BRANCH_COMMITS,
  "branch": "$BRANCH_NAME",
  "message": "$MSG"
}
EOF

exit $EXIT

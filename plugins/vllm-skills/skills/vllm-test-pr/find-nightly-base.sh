#!/bin/bash
# Find the closest vLLM nightly base image for a given commit.
# Usage: find-nightly-base.sh [base-ref]
#   base-ref: the commit the fix branches from (default: HEAD)
#
# Exit codes:
#   0 = found a matching nightly
#   1 = no ancestor nightly found (falls back to closest by date)
#   2 = error
#
# Output: JSON with fields:
#   status: "exact" | "fallback" | "error"
#   tag: full Docker Hub tag (e.g. "nightly-abc123...")
#   image: full image reference (e.g. "vllm/vllm-openai:nightly-abc123...")
#   nightly_sha: commit SHA of the nightly
#   nightly_date: push date of the nightly
#   message: human-readable status message

set -euo pipefail

BASE_REF="${1:-HEAD}"

if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo '{"status":"error","message":"Not inside a git repository"}'
    exit 2
fi

BASE_SHA=$(git rev-parse "$BASE_REF" 2>/dev/null)
if [ -z "$BASE_SHA" ]; then
    echo "{\"status\":\"error\",\"message\":\"Unknown ref: $BASE_REF\"}"
    exit 2
fi

# Fetch nightly tags from Docker Hub (most recent first)
NIGHTLIES=$(curl -s "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/?page_size=50&ordering=last_updated&name=nightly-" 2>/dev/null)

if [ -z "$NIGHTLIES" ] || ! echo "$NIGHTLIES" | python3 -c "import sys,json; json.load(sys.stdin)" &>/dev/null; then
    echo '{"status":"error","message":"Failed to fetch nightly tags from Docker Hub"}'
    exit 2
fi

# Extract nightly SHA and date pairs (skip arch-specific tags)
NIGHTLY_LIST=$(echo "$NIGHTLIES" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for r in data.get('results', []):
    name = r['name']
    if name.startswith('nightly-') and not name.endswith(('aarch64', 'x86_64')):
        sha = name[len('nightly-'):]
        date = r['tag_last_pushed'][:10]
        print(f'{sha} {date} {name}')
" 2>/dev/null)

if [ -z "$NIGHTLY_LIST" ]; then
    echo '{"status":"error","message":"No nightly tags found on Docker Hub"}'
    exit 2
fi

# Try to find a nightly whose commit is an ancestor of our base
FOUND_EXACT=""
while IFS=' ' read -r nsha ndate ntag; do
    # Check if this nightly commit exists in our local repo
    if ! git cat-file -t "$nsha" &>/dev/null; then
        continue
    fi
    # Check if it's an ancestor of our base
    if git merge-base --is-ancestor "$nsha" "$BASE_SHA" 2>/dev/null; then
        FOUND_EXACT="$nsha $ndate $ntag"
        break
    fi
done <<< "$NIGHTLY_LIST"

if [ -n "$FOUND_EXACT" ]; then
    read -r nsha ndate ntag <<< "$FOUND_EXACT"
    cat <<EOF
{
  "status": "exact",
  "tag": "$ntag",
  "image": "vllm/vllm-openai:$ntag",
  "nightly_sha": "$nsha",
  "nightly_date": "$ndate",
  "message": "Found nightly $ntag ($ndate) which is an ancestor of $BASE_REF"
}
EOF
    exit 0
fi

# Fallback: pick the most recent nightly that has a commit present in our repo
FOUND_FALLBACK=""
while IFS=' ' read -r nsha ndate ntag; do
    if git cat-file -t "$nsha" &>/dev/null; then
        FOUND_FALLBACK="$nsha $ndate $ntag"
        break
    fi
done <<< "$NIGHTLY_LIST"

if [ -n "$FOUND_FALLBACK" ]; then
    read -r nsha ndate ntag <<< "$FOUND_FALLBACK"
    cat <<EOF
{
  "status": "fallback",
  "tag": "$ntag",
  "image": "vllm/vllm-openai:$ntag",
  "nightly_sha": "$nsha",
  "nightly_date": "$ndate",
  "message": "No ancestor nightly found. Using closest by date: $ntag ($ndate). There may be drift between this nightly and your fix base."
}
EOF
    exit 1
fi

# Nothing found at all
cat <<EOF
{
  "status": "error",
  "message": "No nightly commits found in local repo. Try running 'git fetch origin' to update."
}
EOF
exit 2

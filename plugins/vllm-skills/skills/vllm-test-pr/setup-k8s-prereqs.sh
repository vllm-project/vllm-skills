#!/bin/bash
# Set up K8s prerequisites for vLLM test pods.
# Usage: setup-k8s-prereqs.sh
#
# Prerequisites:
#   - kubectl configured with the target cluster
#   - HF_TOKEN environment variable set (or HFTOKEN)
#
# What it creates:
#   - Secret "hf-token" with HF_TOKEN key (idempotent via apply)

set -euo pipefail

TOKEN="${HF_TOKEN:-${HFTOKEN:-}}"

if [ -z "$TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN or HFTOKEN environment variable first" >&2
    echo "  export HF_TOKEN=hf_..." >&2
    exit 1
fi

echo "Creating/updating hf-token secret..."
kubectl create secret generic hf-token \
  --from-literal=HF_TOKEN="$TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Done. Secret 'hf-token' is ready."

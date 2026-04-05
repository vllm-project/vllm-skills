---
name: vllm-test-image
description: Generate a K8s manifest to test a vLLM Python-only fix via editable install onto a nightly base image — no container build or registry push required.
---

# vLLM: Test a Fix in K8s

Your job is to generate a K8s Pod manifest that tests a vLLM code fix by cloning the user's fork into a nightly container and performing a zero-download editable install. No container image build or registry push is needed.

## Prerequisites

### Local dev environment

Run once per repo clone to set up test results directories, git excludes, and upstream remote:

```bash
~/.claude/skills/vllm-test-image/setup-local-dev.sh
```

This creates `vllm-test-results/{issues,pull-requests}/`, adds entries to `.git/info/exclude`, and ensures the `upstream` remote exists with tags fetched. Idempotent — safe to run multiple times.

### K8s cluster

Ensure the cluster has the HF token secret:

```bash
export HF_TOKEN=hf_...
~/.claude/skills/vllm-test-image/setup-k8s-prereqs.sh
```

Or manually:

```bash
kubectl create secret generic hf-token \
  --from-literal=HF_TOKEN=$HF_TOKEN \
  --dry-run=client -o yaml | kubectl apply -f -
```

The secret name is `hf-token` with key `HF_TOKEN`. All generated manifests reference this.

## Inputs

Determine these from the user or from the current repo state:

1. **Path to the local vLLM repo** with the fix applied (required)
2. **Git remote + branch** to clone from inside the container (required — the fork must be pushed)
3. **GitHub issue number or URL** (optional but strongly recommended) — used to extract deployment context

## Step 0: Extract deployment context from the issue

If a GitHub issue number or URL is provided, fetch the issue and extract deployment-relevant details. Use `gh issue view <number> --repo vllm-project/vllm --json title,body,comments` to get the full issue content including comments.

Scan the issue title, body, and comments for the following. Record whatever you find — these will be used in Step 4 to generate the manifest:

| Field | What to look for | Examples in issue text |
|-------|------------------|-----------------------|
| **Model** | Model name/path mentioned in repro steps or error logs | `meta-llama/Llama-3.1-70B`, `mistralai/Mixtral-8x7B-v0.1` |
| **GPU count** | Number of GPUs used in repro | `--tensor-parallel-size 4`, `8xA100`, `4 GPUs` |
| **Tensor parallelism (TP)** | TP degree | `--tensor-parallel-size`, `tp=4`, `TP 8` |
| **Pipeline parallelism (PP)** | PP degree | `--pipeline-parallel-size`, `pp=2`, `PP 4` |
| **Data parallelism (DP)** | DP or multi-instance | `--data-parallel-size`, `dp=2`, replicas |
| **Max model length** | Context length used | `--max-model-len 4096`, `context_length` |
| **Quantization** | Quantization method | `--quantization awq`, `gptq`, `fp8` |
| **Additional vLLM args** | Any other CLI flags in the repro | `--enforce-eager`, `--enable-chunked-prefill`, `--dtype float16` |
| **Multi-node** | Whether the repro spans multiple nodes | `multi-node`, `2 nodes`, `ray`, `LeaderWorkerSet` |
| **GPU type** | GPU model mentioned | `A100`, `H100`, `L40S`, `A10G`, `B200` |
| **CUDA version** | CUDA version mentioned in logs, environment, or image tags | `CUDA 13.0`, `cu130`, `CUDA Version: 12.9` |
| **Error condition** | What to look for when testing | `OOM`, `CUDA error`, `hang`, `wrong output` |

If a field is not found in the issue, leave it unset (it will be asked in Step 4 or defaulted).

**Important**: If the issue contains an explicit reproduction command (e.g., `python -m vllm.entrypoints.openai.api_server --model X --tensor-parallel-size 4 ...`), extract the full command and all its args. This is the most reliable source of deployment config.

### CUDA version detection

Determine the appropriate nightly image variant from the issue context:

1. **Explicit CUDA version**: If the issue mentions `CUDA 13`, `cu130`, or Blackwell GPUs (B200, B100, GB200) → use `vllm/vllm-openai:cu130-nightly`
2. **Explicit CUDA 12**: If the issue mentions `CUDA 12`, `cu129`, or only Hopper/Ampere GPUs → use `vllm/vllm-openai:nightly` (default, CUDA 12)
3. **Ambiguous or unspecified**: Default to `cu130-nightly`. If the issue mentions both GPU generations or is unclear, flag this to the user:
   > "The issue doesn't specify a CUDA version. Defaulting to cu130-nightly. If you need to test on CUDA 12 as well, let me know and I'll generate a second manifest."

Store the extracted context as a structured set of values to pass into Step 4.

## Step 1: Check branch freshness (advisory)

vLLM moves very fast — `origin/main` on forks goes stale within hours. Run the freshness check to inform the user:

```bash
~/.claude/skills/vllm-test-image/check-branch-freshness.sh [max-commits-behind]
```

The script:
1. Ensures the `upstream` remote exists (adds `https://github.com/vllm-project/vllm.git` if not)
2. Fetches `upstream/main`
3. Checks how many commits `upstream/main` is ahead of the branch's merge-base

Exit codes:
- **Exit 0** (`"status": "fresh"`): branch is within the threshold (default: 50 commits behind)
- **Exit 1** (`"status": "stale"`): branch is behind — inform the user but **do not block**:
  > "Your branch is N commits behind upstream/main. This is advisory — testing can proceed, but if you hit unrelated test failures, rebasing may help: `git rebase upstream/main`"
- **Exit 2** (`"status": "error"`): something went wrong

**This is advisory, not a gate.** Rebasing can be painful in a fast-moving codebase and isn't always necessary for testing a fix. The change scope detection (Step 2) works correctly regardless of staleness since it diffs from the merge-base.

However, if the nightly base image is much newer than the branch's fork point, there could be drift between the nightly's installed code and what the branch expects. Flag this to the user if `commits_behind` > 100.

## Step 2: Detect change scope

Run the detection script bundled with this skill:

```bash
~/.claude/skills/vllm-test-image/detect-change-scope.sh [base-ref]
```

The default base-ref is `upstream/main` (not `origin/main`) since forks' origin is typically stale. The script auto-fetches `upstream/main` if needed. It diffs from the merge-base, so it correctly shows only the branch's own changes regardless of how many commits upstream has moved ahead.

The script outputs JSON and uses exit codes:
- **Exit 0** (`"scope": "python-only"`): safe to proceed with editable install approach
- **Exit 1** (`"scope": "compiled"`): touches C++/CUDA/build files — stop and tell the user:
  > "This fix touches compiled code. The editable install approach only works for Python-only changes. A full image rebuild would be needed, which is not yet supported by this skill."
- **Exit 2** (`"scope": "error"`): something went wrong (not a git repo, bad ref, no changes)

The JSON response also includes:
- `py_files`: list of changed `.py` files under `vllm/`
- `compiled_files`: list of compiled/build files (if any)
- `dep_files`: list of changed dependency files — warn if non-empty
- `test_only`: true if all changes are test files — warn there may be nothing to test at runtime
- `warnings`: list of warning strings to show the user

Parse the JSON output. You don't need `py_files` for the manifest (the editable install handles everything), but do show warnings to the user.

## Step 3: Find the closest nightly base image

Run the nightly-finder script bundled with this skill:

```bash
~/.claude/skills/vllm-test-image/find-nightly-base.sh [base-ref]
```

The script queries Docker Hub for `vllm/vllm-openai:nightly-<sha>` tags and checks git ancestry. It outputs JSON with exit codes:
- **Exit 0** (`"status": "exact"`): found a nightly whose commit is an ancestor of the fix base
- **Exit 1** (`"status": "fallback"`): no exact ancestor; used closest nightly by date — warn the user about potential drift
- **Exit 2** (`"status": "error"`): something went wrong

The JSON response includes:
- `tag`: the Docker Hub tag
- `image`: full image reference
- `nightly_sha`: commit SHA the nightly was built from
- `nightly_date`: when the nightly was pushed
- `message`: human-readable status

**Note**: The `find-nightly-base.sh` script currently searches for `nightly-<sha>` tags (CUDA 12 variants). For CUDA 13 (`cu130-nightly`), the rolling tag `vllm/vllm-openai:cu130-nightly` is typically used since SHA-tagged cu130 nightlies may not be available. If the CUDA version from Step 0 indicates cu130, use `vllm/vllm-openai:cu130-nightly` as the base image and note the `nightly_sha` from the script output for reference only.

## Step 4: Confirm deployment config with user

Before generating the manifest, show the user what was extracted and what will be generated:

```
## Deployment config (from issue #XXXX)

- Model: meta-llama/Llama-3.1-70B
- Base image: vllm/vllm-openai:cu130-nightly
- GPUs per pod: 4 (tensor-parallel-size=4)
- Quantization: awq
- Additional args: --enforce-eager
- Fork: github.com/user/vllm @ branch-name
- Mode: editable install (zero-download)
- Missing (will use defaults): max-model-len, dtype

Proceed? (or provide overrides)
```

This lets the user correct anything before the manifest is written.

## Step 5: Generate K8s Pod manifest

Generate a Pod manifest that:
1. Starts the nightly container with `sleep infinity`
2. Runs an init script (via a `command` override or `lifecycle.postStart`) that:
   - Installs git
   - Clones the user's fork+branch
   - Packs existing `.so` files into a local wheel (zero network download)
   - Performs an editable install
   - Optionally starts the vLLM server

### Manifest template

The manifest uses a single container with a startup script. The script is embedded as a multi-line `command` + `args` using `bash -c`.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-test-<tag>
  labels:
    app: vllm-test-<tag>
spec:
  containers:
  - name: vllm
    image: <base-image>
    command: ["bash", "-c"]
    args:
    - |
      set -e

      echo "=== [1/5] Installing git ==="
      apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1

      echo "=== [2/5] Cloning fork ==="
      git clone -b <branch> <fork-url> /tmp/vllm-fix
      cd /tmp/vllm-fix

      echo "=== [3/5] Setting up upstream remote and syncing tags ==="
      git remote add upstream https://github.com/vllm-project/vllm.git
      git fetch upstream --tags --quiet

      echo "=== [4/5] Packing local wheel from existing .so files ==="
      python3 -c '
      import zipfile, os, glob
      SITE = "/usr/local/lib/python3.12/dist-packages"
      OUTPUT = "/tmp/vllm-local-precompiled.whl"
      files = glob.glob(os.path.join(SITE, "vllm", "**", "*.abi3.so"), recursive=True)
      for p in ["vllm/vllm_flash_attn/**/*.py", "vllm/third_party/triton_kernels/**/*.py", "vllm/third_party/flashmla/**/*.py"]:
          files.extend(glob.glob(os.path.join(SITE, p), recursive=True))
      print(f"Packing {len(files)} files")
      with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
          for f in files:
              zf.write(f, os.path.relpath(f, SITE))
      print(f"Created {OUTPUT} ({os.path.getsize(OUTPUT)/1048576:.0f} MB)")
      '

      echo "=== [5/5] Editable install ==="
      VLLM_USE_PRECOMPILED=1 \
      VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm-local-precompiled.whl \
      pip install -e . --no-build-isolation 2>&1 | tail -5

      echo "=== Ready ==="
      python3 -c "import vllm; print(f'vLLM {vllm.__version__} from {vllm.__file__}')"

      # --- Test then serve (default) ---
      # Run tests first; if they pass, start serving the model as a secondary check.
      mkdir -p /tmp/vllm-test-results
      echo "=== Running tests ==="
      python3 -m pytest <test-path> -v 2>&1 | tee /tmp/vllm-test-results/pytest.log
      echo $? > /tmp/vllm-test-results/pytest-exit-code

      if [ "$(cat /tmp/vllm-test-results/pytest-exit-code)" = "0" ]; then
          echo "=== Tests PASSED — starting vLLM server ==="
          exec python3 -m vllm.entrypoints.openai.api_server \
            --model <model> \
            --port 8000 \
            <additional-args>
      else
          echo "=== Tests FAILED — not starting server ==="
          echo "Logs at /tmp/vllm-test-results/pytest.log"
          echo "Attach with: kubectl exec -it vllm-test-<tag> -- bash"
          sleep infinity
      fi
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token
          key: HF_TOKEN
    ports:
    - containerPort: 8000
    resources:
      limits:
        nvidia.com/gpu: "<gpu-count>"
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 16Gi
  restartPolicy: Never
```

### Manifest modes

The default mode is **test then serve**: run the relevant tests, and if they pass, start the vLLM server with the issue's deployment config as a secondary validation. If tests fail, the pod stays alive with `sleep infinity` for debugging.

**Serve-only mode**: If no tests are relevant (or the user explicitly skips), remove the test block and go straight to `exec python3 -m vllm.entrypoints.openai.api_server ...`.

**Test-only mode**: If there's no model to serve (e.g., kernel-level unit tests), replace the server block with:

```bash
      if [ "$(cat /tmp/vllm-test-results/pytest-exit-code)" = "0" ]; then
          echo "=== Tests PASSED ==="
      else
          echo "=== Tests FAILED ==="
      fi
      echo "Attach with: kubectl exec -it vllm-test-<tag> -- bash"
      sleep infinity
```

### Test log capture

Test output is written to consistent locations both **inside the pod** and **locally**.

**Inside the pod** (during test run):

```
/tmp/vllm-test-results/pytest.log        # Full pytest output (stdout+stderr)
/tmp/vllm-test-results/pytest-exit-code  # Exit code from pytest
```

**Locally** (after retrieval), organized by issue or PR number:

```
vllm-test-results/
  issues/
    39025/                    # One directory per issue number
      pytest.log
  pull-requests/
    38997/                    # One directory per PR number
      pytest.log
```

This directory is excluded from git via `.git/info/exclude` (local only, not in upstream `.gitignore`).

**To retrieve logs from the pod into the local structure:**

```bash
# For an issue:
mkdir -p vllm-test-results/issues/<issue-number>
kubectl cp <pod-name>:/tmp/vllm-test-results/pytest.log ./vllm-test-results/issues/<issue-number>/pytest.log

# For a PR:
mkdir -p vllm-test-results/pull-requests/<pr-number>
kubectl cp <pod-name>:/tmp/vllm-test-results/pytest.log ./vllm-test-results/pull-requests/<pr-number>/pytest.log
```

When generating test commands interactively (outside the manifest), always tee to the log:

```bash
cd /tmp/vllm-fix && \
  mkdir -p /tmp/vllm-test-results && \
  python3 -m pytest tests/path/to/test.py -v 2>&1 | tee /tmp/vllm-test-results/pytest.log; \
  echo $? > /tmp/vllm-test-results/pytest-exit-code
```

### Shared K8s manifest constants

All generated manifests must use these shared values:

| Field | Value | Notes |
|-------|-------|-------|
| HF token secret name | `hf-token` | Created by `setup-k8s-prereqs.sh` |
| HF token secret key | `HF_TOKEN` | |
| Shared memory mount | `/dev/shm`, `emptyDir` with `medium: Memory`, `sizeLimit: 16Gi` | Required for NCCL/multi-GPU |
| Source clone path | `/tmp/vllm-fix` | Where the fork is cloned inside the container |
| Local wheel path | `/tmp/vllm-local-precompiled.whl` | Packed .so files |
| Test results path | `/tmp/vllm-test-results/` | Consistent log output location |
| Container port | `8000` | vLLM server default |
| Restart policy | `Never` | Test pods should not auto-restart |

### Resource type selection

Choose based on the extracted context:
- **Pod** — default, single-node, single vLLM instance
- **Deployment** — if the user requests restart resilience or if DP > 1 (data parallel replicas)
- **LeaderWorkerSet** — if multi-node is needed (PP across nodes, or TP exceeds single-node GPU count)

### GPU resources

Set `nvidia.com/gpu` based on:
- If TP was extracted: GPUs per pod = TP value
- If both TP and PP: GPUs per pod = TP value, number of pods = PP value (LeaderWorkerSet)
- If only GPU count mentioned: use that
- Default: 1

### Image tagging convention for the Pod name

- If there's a linked issue: `vllm-test-fix-<issue-number>` (e.g., `vllm-test-fix-12345`)
- Otherwise: `vllm-test-<short-sha>` where short-sha is the first 8 chars of HEAD

Save the manifest to `k8s-test-fix-<issue-number>.yaml` or `k8s-test-<short-sha>.yaml` in the repo root.

## Output

When done, present a summary:

```
## Test Pod Ready

- **Base image:** vllm/vllm-openai:cu130-nightly
- **Fork:** github.com/user/vllm @ fix/my-branch
- **Install method:** zero-download editable install (local .so repack)
- **Mode:** test then serve
- **Tests:** tests/path/to/test.py
- **Model:** model-name (served after tests pass)
- **Manifest:** k8s-test-fix-39025.yaml

### Deploy:
kubectl apply -f k8s-test-fix-39025.yaml

### Watch startup + tests + server:
kubectl logs -f vllm-test-fix-39025

### Retrieve test logs:
mkdir -p vllm-test-results/issues/39025  # or pull-requests/<pr-number>
kubectl cp vllm-test-fix-39025:/tmp/vllm-test-results/pytest.log ./vllm-test-results/issues/39025/pytest.log

### Quick server test (after tests pass and server starts):
kubectl port-forward pod/vllm-test-fix-39025 8000:8000
curl http://localhost:8000/v1/models

### If tests failed:
kubectl exec -it vllm-test-fix-39025 -- bash
cat /tmp/vllm-test-results/pytest.log
```

## How it works (for reference)

The editable install approach avoids building or pushing container images entirely:

1. The nightly container already has vLLM installed with precompiled `.so` files (CUDA extensions)
2. We pack those `.so` files into a local `.whl` (zip) — ~200 MB, stays on-disk, no network
3. We clone the user's fork into the container
4. `VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/local.whl pip install -e .` does:
   - Extracts `.so` files from our local wheel into the source tree (instant, no download)
   - Skips compilation (precompiled_build_ext is a no-op)
   - Sets up the editable install pointing at the cloned source
5. Python now loads `.py` files from the fork and `.so` files from the source tree

**Why this solves the CUDA version problem:** The `.so` files are repacked from whatever is already in the container. If the container is cu130-nightly, the `.so` files are cu130. If it's cu129, they're cu129. No version detection or matching needed.

## Limitations (current)

- **Python-only fixes**: This skill only works for `.py` file changes. Fixes that touch C++/CUDA code require a full image rebuild (not yet supported).
- **Dependency changes**: If the fix adds new Python dependencies, those won't be in the base image. The skill will warn but not handle this case.
- **Nightly lag**: Nightly images are built once per day. There may be up to ~24 hours of commits between the nightly base and the fix's base commit.
- **Git required**: The container needs `apt-get install git` at startup, which requires network access. This adds ~10-15 seconds to pod startup.

## Future enhancements (not implemented yet)

- Full image rebuild for C++/CUDA changes
- Automatic dependency installation for new pip requirements
- Multi-arch image builds
- Integration with CI pipelines
- Dual CUDA version testing (generate both cu129 and cu130 manifests)

## Style

Be direct. Show the user what's happening at each step. If something fails, diagnose and suggest a fix rather than retrying blindly.

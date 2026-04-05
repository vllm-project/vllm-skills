#!/usr/bin/env python3
"""
Pack existing vLLM .so files and third-party Python files from a container's
installed package into a local .whl (zip) that can be used with
VLLM_PRECOMPILED_WHEEL_LOCATION to avoid downloading a remote wheel.

Usage: python3 pack-local-wheel.py [output-path]
  output-path: where to write the .whl (default: /tmp/vllm-local-precompiled.whl)

This script runs INSIDE a vLLM nightly container where vllm is already
installed at SITE_PACKAGES/vllm/.

Exit codes:
  0 = .whl created
  1 = no .so files found
  2 = error
"""

import glob
import os
import sys
import zipfile

SITE = "/usr/local/lib/python3.12/dist-packages"
OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vllm-local-precompiled.whl"

if not os.path.isdir(os.path.join(SITE, "vllm")):
    print(f"ERROR: {SITE}/vllm not found. Is this a vLLM container?",
          file=sys.stderr)
    sys.exit(2)

files = []

# Collect all .abi3.so files (wildcard — future-proof)
files.extend(
    glob.glob(os.path.join(SITE, "vllm", "**", "*.abi3.so"), recursive=True)
)

if not files:
    print("ERROR: No .abi3.so files found", file=sys.stderr)
    sys.exit(1)

# Collect third-party Python files that setup.py also extracts from wheels
for pattern in [
    "vllm/vllm_flash_attn/**/*.py",
    "vllm/third_party/triton_kernels/**/*.py",
    "vllm/third_party/flashmla/**/*.py",
]:
    files.extend(glob.glob(os.path.join(SITE, pattern), recursive=True))

print(f"Packing {len(files)} files into {OUTPUT}")

with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for fpath in files:
        arcname = os.path.relpath(fpath, SITE)
        zf.write(fpath, arcname)

size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
print(f"Created {OUTPUT} ({size_mb:.1f} MB)")

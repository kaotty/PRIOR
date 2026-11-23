#!/usr/bin/env bash
# Install dependencies inside the Python 3.12 environment selected by setup_env.sh.
#
#   bash scripts/setup_env.sh
#
# Installs a single mutually-compatible, pinned stack for the SGLang rollout
# path. A few packages must be installed in a specific order (torch first; cuDNN
# last to override torch's bundled version), which is why this is a script
# rather than a bare `pip install -r requirements.txt`.
#
# Requirements: Python 3.12, Linux x86_64, NVIDIA CUDA 12.x.
#
# The FlashAttention wheel below is built for x86_64. Other architectures
# require a matching wheel or a source build.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
python - <<'PY'
import platform
import sys
assert sys.version_info[:2] == (3, 12), "Use the Python 3.12 environment created by setup_env.sh"
assert platform.system() == "Linux" and platform.machine() == "x86_64", "This installer targets Linux x86_64"
PY

# 1. PyTorch first, so nothing downgrades it (sglang 0.5.8 strict-pins torch==2.9.1).
python -m pip install -c "${PROJECT_ROOT}/requirements.lock.txt" torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

# 2. sglang + the rest of the pinned stack, resolved against the installed torch.
python -m pip install -c "${PROJECT_ROOT}/requirements.lock.txt" -r "${PROJECT_ROOT}/requirements.txt"

# 3. FlashAttention prebuilt wheel matched to torch 2.9. MUST be cxx11abiTRUE
#    (torch 2.9 on linux x86_64 uses C++11 ABI = TRUE); the abiFALSE wheel loads
#    but crashes later with std::string ABI mismatches. cp312 / linux x86_64.
python -m pip install -c "${PROJECT_ROOT}/requirements.lock.txt" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810"

# 4. Newer cuDNN must win over torch's bundled one (sglang 0.5.8 needs cuDNN >= 9.15).
#    Installed LAST so it overrides torch's transitive pin.
python -m pip install nvidia-cudnn-cu12==9.16.0.29

# 5. Install the verl package in-place so `python -m verl.trainer.*` resolves
#    from anywhere (--no-deps: deps are already pinned above).
python -m pip install -e "${PROJECT_ROOT}" --no-deps

echo "=== install complete — versions ==="
python - <<'PY'
import importlib.metadata as md
for p in ["torch", "sglang", "sgl-kernel", "flashinfer-python", "flash-attn",
          "transformers", "nvidia-cudnn-cu12"]:
    try:
        print(f"{p}=={md.version(p)}")
    except Exception as e:
        print(f"{p}: MISSING ({e})")
PY

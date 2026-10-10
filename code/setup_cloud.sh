#!/usr/bin/env bash
# Cloud routine setup for CALREP Step 1 (CPU). Records the machine.
set -euo pipefail
python3 -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python3 -m pip install -q numpy==2.1.3 scipy==1.14.1 huggingface_hub==0.36.0
mkdir -p "$(dirname "$0")/../results"
{ echo "nproc $(nproc)"; free -g | head -2; python3 -c 'import torch,sys;print("torch",torch.__version__,"threads",torch.get_num_threads(),"py",sys.version.split()[0])'; } \
  > "$(dirname "$0")/../results/machine_$(hostname).txt" 2>&1 || true

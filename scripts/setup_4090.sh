#!/usr/bin/env bash
set -e
export UV_CACHE_DIR=/mnt/data1/qiuwenjie/uv-cache HF_HOME=/mnt/data1/qiuwenjie/hf-home
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
ROOT=/mnt/data1/qiuwenjie/shop-rl
UV=$(ls ~/.local/bin/uv 2>/dev/null | head -1)
if [ -z "$UV" ]; then python3 -m pip install --user -q uv && UV=$(ls ~/.local/bin/uv 2>/dev/null | head -1); fi
echo "uv at $UV"
[ -d $ROOT/venv ] || $UV venv $ROOT/venv --python python3.10
$UV pip install --python $ROOT/venv/bin/python torch transformers==5.20.2 peft tensorboard
$ROOT/venv/bin/python -c "import torch,transformers,peft;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__,'peft',peft.__version__)"
$ROOT/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p=snapshot_download("Qwen/Qwen3.5-4B", local_dir="/mnt/data1/qiuwenjie/shop-rl/models/Qwen3.5-4B", max_workers=8)
print("WEIGHTS DONE", p)
PY
echo "SETUP DONE"

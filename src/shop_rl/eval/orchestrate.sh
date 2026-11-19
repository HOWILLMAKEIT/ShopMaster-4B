#!/bin/bash
# SFT-B 评测编排（在 4080 上运行，经免密 ssh 操控 4090，不经外部路由）
set -u
cd /data/qiuwenjie/FES/shop-rl
ENVU="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY"
G4="qiuwenjie@222.201.145.4"
MARK=/tmp/B_EVAL_DONE; rm -f "$MARK"

ensure_tunnel() {
  curl -s -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && return 0
  pkill -f "ssh -fN -L 8000" 2>/dev/null; sleep 1
  ssh -fN -L 8000:127.0.0.1:8000 -o ConnectTimeout=15 "$G4" 2>/dev/null
  sleep 2
  curl -s -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1
}

serve_ep() {  # $1 = epoch
  ssh -o ConnectTimeout=15 "$G4" 'pkill -9 -f "launch_serve[r]"; sleep 2' 2>/dev/null
  ssh -o ConnectTimeout=15 "$G4" "cd /mnt/data1/qiuwenjie/shop-rl && PATH=/mnt/data1/qiuwenjie/shop-rl/venv-eval/bin:\$PATH HOME=/mnt/data1/qiuwenjie/home-eval XDG_CACHE_HOME=/mnt/data1/qiuwenjie/xdg-cache TRITON_CACHE_DIR=/mnt/data1/qiuwenjie/triton-cache HF_HOME=/mnt/data1/qiuwenjie/hf-home CUDA_VISIBLE_DEVICES=0 nohup venv-eval/bin/python -m sglang.launch_server --model-path models/qwen3.5-4b-sftb-mm-ep$1 --served-model-name qwen3.5-4b-sft --host 0.0.0.0 --port 8000 --mem-fraction-static 0.85 --context-length 32768 --attention-backend triton --sampling-backend pytorch --tool-call-parser qwen3_coder --reasoning-parser qwen3 > logs/sglang.log 2>&1 & echo SERVED_$1" 2>/dev/null
  local n=0
  until ensure_tunnel; do sleep 20; n=$((n+1)); [ $n -gt 60 ] && return 1; done
  return 0
}

for ep in 1 2 3; do
  if [ "$ep" -gt 1 ]; then
    echo "[$(date +%H:%M:%S)] ep$ep 换服务"
    serve_ep "$ep" || { echo "ep$ep 服务失败"; exit 1; }
  else
    ensure_tunnel || { echo "ep1 隧道/服务失败"; exit 1; }
  fi
  echo "[$(date +%H:%M:%S)] ep$ep 全量启动"
  $ENVU .venv/bin/python -I env/single_eval/collect_teacher.py collect \
      --yaml "env/single_eval/configs/standard/eval_local_b_ep${ep}.yaml" \
      --workers 6 >> "/tmp/eval_b_ep${ep}.log" 2>&1
  n=$(ls "outputs/eval/eval-b-ep${ep}-dev200"/*.json 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] ep$ep collect 退出，完成 ${n}/200"
  [ "$n" -lt 200 ] && { echo "ep$ep 未满 200，中止"; exit 1; }
done

ssh -o ConnectTimeout=15 "$G4" 'pkill -9 -f "launch_serve[r]"' 2>/dev/null
touch "$MARK"
echo "[$(date +%H:%M:%S)] ALL_DONE"

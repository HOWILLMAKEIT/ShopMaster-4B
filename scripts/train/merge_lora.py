#!/usr/bin/env python
"""把 LoRA adapter merge 进文本骨干模型并存为独立目录。

背景：训练用 AutoModelForCausalLM 文本视图加载（4.21B，视觉塔不加载），
adapter 挂在文本视图上；本脚本产出 merge 后的文本模型目录，供
graft_sft_mm.py 嫁接回多模态壳后由 sglang 服务（sglang 0.5.9 未注册
文本版 model_type qwen3_5_text）。

用法（gong-4090 项目根）：
  venv/bin/python scripts/train/merge_lora.py --src models/Qwen3.5-4B \
      --adapter-tmpl "outputs/v1/sft-a/adapter-ep{}" --out-tmpl "models/qwen3.5-4b-sft-ep{}"
  # SFT-B：--adapter-tmpl "outputs/v1/sft-b/adapter-ep{}" --out-tmpl "models/qwen3.5-4b-sftb-ep{}"
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/Qwen3.5-4B", help="原始多模态 checkpoint")
    ap.add_argument("--adapter-tmpl", required=True, help="adapter 目录模板，如 outputs/v1/sft-a/adapter-ep{}")
    ap.add_argument("--out-tmpl", required=True, help="merge 产物目录模板")
    ap.add_argument("--epochs", default="1,2,3", help="要处理的 epoch 列表")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.src)
    for ep in [int(x) for x in args.epochs.split(",")]:
        model = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, args.adapter_tmpl.format(ep))
        model = model.merge_and_unload()
        out = args.out_tmpl.format(ep)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        print(f"merge ep{ep} -> {out}", flush=True)
        del model


if __name__ == "__main__":
    main()

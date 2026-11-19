#!/usr/bin/env python
"""把 merge 后的文本骨干权重嫁接回多模态 checkpoint 壳。

背景：训练用 AutoModelForCausalLM 文本视图（module 路径 model.layers.*），
merge 产物是纯文本模型（model_type qwen3_5_text）；推理服务 sglang 0.5.9
只注册了多模态 model_type qwen3_5，故需把文本权重写回原始多模态分片，
统一按多模态 checkpoint 服务（视觉塔权重原样保留，文本评测不经过）。

处理：读原始 index 的张量名与文本 merge 产物的张量名，按去前缀后缀匹配
（text "model.layers.0.x" ↔ full "...language_model.layers.0.x"），逐一替换
后按原分片结构写出到 models/qwen3.5-4b-sft-mm-ep{N}；其余文件原样复制。
运行（gong-4090 项目根）：venv/bin/python shop_rl.train.graft
"""
import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def tensor_files(d: Path):
    return sorted(p for p in d.glob("*.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/Qwen3.5-4B")
    ap.add_argument("--text-tmpl", default="models/qwen3.5-4b-sft-ep{}",
                    help="merge 后的文本模型目录模板（A 系）；B 系传 models/qwen3.5-4b-sftb-ep{}")
    ap.add_argument("--out-tmpl", default="models/qwen3.5-4b-sft-mm-ep{}")
    args = ap.parse_args()
    SRC = Path(args.src)
    OUT_TMPL, TEXT_TMPL = args.out_tmpl, args.text_tmpl

    for ep in (1, 2, 3):
        text_dir = Path(TEXT_TMPL.format(ep))
        text_state = {}
        for f in tensor_files(text_dir):
            text_state.update(load_file(str(f)))
        # 后缀匹配：去掉 text 名的 "model." 前缀后应唯一匹配 full 名
        core = {t[len("model."):] if t.startswith("model.") else t: t
                for t in text_state}
        out_dir = Path(OUT_TMPL.format(ep))
        out_dir.mkdir(parents=True, exist_ok=True)
        matched, total_full = 0, 0
        for f in tensor_files(SRC):
            state = load_file(str(f))
            for name in list(state):
                total_full += 1
                hits = [c for c in core if name == c or name.endswith("." + c)]
                if len(hits) == 1:
                    state[name] = text_state[core[hits[0]]]
                    matched += 1
                elif len(hits) > 1:
                    raise SystemExit(f"多义匹配: {name} -> {hits}")
            save_file(state, str(out_dir / f.name), metadata={"format": "pt"})
        # 复制非权重文件（config/tokenizer/chat_template/preprocessor 等）
        for f in SRC.iterdir():
            if f.suffix != ".safetensors" and f.is_file():
                shutil.copy(f, out_dir / f.name)
        shutil.copy(SRC / "model.safetensors.index.json", out_dir / "model.safetensors.index.json")
        print(f"ep{ep}: 文本张量 {len(core)}，替换 {matched}/{total_full} 个全量张量 -> {out_dir}",
              flush=True)


if __name__ == "__main__":
    main()

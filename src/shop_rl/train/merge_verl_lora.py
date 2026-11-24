#!/usr/bin/env python3
"""将 verl FSDP2 导出的纯 CPU LoRA 权重直接 merge 回多模态 BF16 Safetensors 模型。

用法:
  python -m shop_rl.train.merge_verl_lora \
      --base-model models/qwen3.5-4b-sftb-mm-ep2 \
      --lora-pt models/rl_loras/v2-a800-001-step100-lora-cpu.pt \
      --lora-meta models/rl_loras/v2-a800-001-step100-meta.json \
      --output-dir models/qwen3.5-4b-grpo100-mm
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_LORA_PREFIX = "base_model.model."
_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"


def parse_lora_pairs(
    lora_state: Dict[str, torch.Tensor],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """把 verl LoRA dict 的 lora_A/lora_B 配对并映射回 base model safetensors 键名。"""
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in lora_state.items():
        if not key.startswith(_LORA_PREFIX):
            raise ValueError(f"非预期的 LoRA 键前缀: {key}")
        stripped = key[len(_LORA_PREFIX) :]
        if stripped.endswith(_LORA_A_SUFFIX):
            base_key = stripped[: -len(_LORA_A_SUFFIX)] + ".weight"
            slot = "A"
        elif stripped.endswith(_LORA_B_SUFFIX):
            base_key = stripped[: -len(_LORA_B_SUFFIX)] + ".weight"
            slot = "B"
        else:
            raise ValueError(f"非预期的 LoRA 键后缀: {key}")
        entry = grouped.setdefault(base_key, {})
        if slot in entry:
            raise ValueError(f"重复的 LoRA {slot} 键: {key}")
        entry[slot] = tensor

    pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for base_key, entry in grouped.items():
        if "A" not in entry or "B" not in entry:
            raise ValueError(f"LoRA 权重配对不完整 ({base_key}): keys={list(entry.keys())}")
        pairs[base_key] = (entry["A"], entry["B"])
    if not pairs:
        raise ValueError("LoRA 权重为空，未解析到任何可 merge 层")
    return pairs


def load_lora_scale(
    lora_pt_path: pathlib.Path, lora_meta_path: Optional[pathlib.Path]
) -> Tuple[float, int, float]:
    """读取 r 与 lora_alpha 并计算 LoRA merge 缩放系数 alpha / r。"""
    candidate = lora_meta_path
    if candidate is None:
        sibling = lora_pt_path.parent / "lora_train_meta.json"
        if sibling.is_file():
            candidate = sibling

    if candidate is not None:
        meta = json.loads(candidate.read_text(encoding="utf-8"))
        r = int(meta["r"])
        lora_alpha = float(meta["lora_alpha"])
    else:
        r = 16
        lora_alpha = 32.0
    if r <= 0:
        raise ValueError(f"无效的 LoRA rank r={r}")
    return lora_alpha / r, r, lora_alpha


def merge_verl_lora(
    base_model_dir: pathlib.Path,
    lora_pt_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    lora_meta_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """将纯 CPU LoRA `.pt` 精确 merge 到 base model 的 safetensors 分片并输出完整目录。"""
    base_dir = base_model_dir.expanduser().resolve()
    out_dir = output_dir.expanduser().resolve()
    pt_path = lora_pt_path.expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"base model 目录不存在: {base_dir}")
    if not pt_path.is_file():
        raise FileNotFoundError(f"LoRA .pt 文件不存在: {pt_path}")

    scale, r, lora_alpha = load_lora_scale(pt_path, lora_meta_path)
    lora_state = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(lora_state, dict):
        raise TypeError(f"期望 dict 类型的 LoRA state_dict，实际为 {type(lora_state).__name__}")
    lora_pairs = parse_lora_pairs(lora_state)

    shards = sorted(base_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"base model 目录中未找到 *.safetensors: {base_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for item in base_dir.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            shutil.copy2(item, out_dir / item.name)

    merged_keys: set[str] = set()
    max_abs_diff = 0.0
    total_abs_diff = 0.0
    total_elems = 0

    for shard_path in shards:
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {"format": "pt"}
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in lora_pairs:
                    lora_a, lora_b = lora_pairs[key]
                    delta = (lora_b.float() @ lora_a.float()) * scale
                    if tuple(delta.shape) != tuple(tensor.shape):
                        raise ValueError(
                            f"形状不匹配 ({key}): base={tuple(tensor.shape)}, delta={tuple(delta.shape)}"
                        )
                    base_f32 = tensor.float()
                    merged_f32 = base_f32 + delta
                    diff = (merged_f32 - base_f32).abs()
                    shard_max = float(diff.max().item())
                    if shard_max > max_abs_diff:
                        max_abs_diff = shard_max
                    total_abs_diff += float(diff.sum().item())
                    total_elems += tensor.numel()
                    tensor = merged_f32.to(dtype=tensor.dtype).contiguous()
                    merged_keys.add(key)
                tensors[key] = tensor

        save_file(tensors, str(out_dir / shard_path.name), metadata=metadata)

    unmatched = sorted(set(lora_pairs.keys()) - merged_keys)
    if unmatched:
        raise RuntimeError(
            f"存在未匹配的 LoRA 层 ({len(unmatched)} 个)，示例: {unmatched[:5]}"
        )
    if max_abs_diff <= 0.0:
        raise RuntimeError("Merge 后权重变化为 0，请检查 LoRA 权重是否全零")

    mean_abs_diff = total_abs_diff / max(1, total_elems)
    return {
        "base_model": str(base_dir),
        "lora_pt": str(pt_path),
        "output_dir": str(out_dir),
        "r": r,
        "lora_alpha": lora_alpha,
        "scale": scale,
        "merged_layers": len(merged_keys),
        "expected_layers": len(lora_pairs),
        "unmatched_lora_keys": len(unmatched),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="将 verl CPU LoRA 权重合并进 BF16 Safetensors 模型。")
    parser.add_argument("--base-model", required=True, type=pathlib.Path)
    parser.add_argument("--lora-pt", required=True, type=pathlib.Path)
    parser.add_argument("--lora-meta", default=None, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    summary = merge_verl_lora(
        args.base_model,
        args.lora_pt,
        args.output_dir,
        lora_meta_path=args.lora_meta,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

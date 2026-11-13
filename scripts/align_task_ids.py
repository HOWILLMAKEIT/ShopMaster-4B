#!/usr/bin/env python3
"""对齐 HF 官方 standard 任务集与环境 task_id。

items_eval_train.json（全集，23,421 条）的顺序即环境 goals 顺序：
get_existed_goals 按全集顺序展开 instructions，本项目数据中每条恰好一个
instruction 且 attributes 非空，因此第 i 条商品对应 task_id=i。

用法（仓库根目录）：
    python3 scripts/align_task_ids.py

输出：
    data/tasks/eval_standard_task_ids.jsonl   {"task_id": 0..1458}
    data/tasks/train_standard_task_ids.jsonl  {"task_id": 1459..23420}

依据与限制见 docs/v0/PROGRESS.md 2026-11-13 记录；task_id 与环境 reset
返回 instruction 的一致性尚未在运行中的环境验证。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL_ITEMS = ROOT / "env/shop_env/data/items_eval_train.json"
SPLITS = {
    "eval": ROOT / "data/fine_items_eval_standard.jsonl",
    "train": ROOT / "data/fine_items_train_standard.jsonl",
}
OUT_DIR = ROOT / "data/tasks"


def iter_json_array(path: Path, chunk_size: int = 1 << 20):
    """流式读取顶层 JSON 数组，避免整份加载 134MB 商品全集。"""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open(encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            eof = not chunk
            buffer += chunk
            position = 0
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError("product file must be a top-level JSON array")
                    started = True
                    position += 1
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield value
                position = end
            if eof:
                raise ValueError("product JSON array is truncated")
            buffer = buffer[position:]


def split_asins(path: Path) -> list[str]:
    asins = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                asins.append(json.loads(line)["asin"])
    return asins


def main() -> None:
    # 模拟 web_agent_site/engine/goal.py get_existed_goals 的顺序与过滤：
    # 无 instructions 或 attributes 为空的条目不生成 goal。当前官方数据
    # 中这两类条目数量为 0，脚本仍保留过滤逻辑以检测未来数据变化。
    task_id_of_asin = {}
    next_id = 0
    skipped_no_instructions = 0
    skipped_empty_attributes = 0
    for item in iter_json_array(FULL_ITEMS):
        instructions = item.get("instructions") or []
        if not instructions:
            skipped_no_instructions += 1
            continue
        for product in instructions:
            if not (product.get("attributes") or []):
                skipped_empty_attributes += 1
                continue
            task_id_of_asin.setdefault(item["asin"], next_id)
            next_id += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"goals={next_id} skipped_no_instructions={skipped_no_instructions} "
          f"skipped_empty_attributes={skipped_empty_attributes}")
    for name, path in SPLITS.items():
        asins = split_asins(path)
        missing = [a for a in asins if a not in task_id_of_asin]
        out_path = OUT_DIR / f"{name}_standard_task_ids.jsonl"
        with out_path.open("w", encoding="utf-8") as out:
            for asin in asins:
                task_id = task_id_of_asin.get(asin)
                if task_id is not None:
                    out.write(json.dumps({"task_id": task_id}, ensure_ascii=False) + "\n")
        ids = [task_id_of_asin[a] for a in asins if a in task_id_of_asin]
        print(f"{name}: items={len(asins)} mapped={len(ids)} missing={len(missing)} "
              f"task_id_range=[{min(ids)}, {max(ids)}] -> {out_path.relative_to(ROOT)}")
        if missing:
            raise SystemExit(f"{name} has {len(missing)} asins absent from full items")


if __name__ == "__main__":
    main()

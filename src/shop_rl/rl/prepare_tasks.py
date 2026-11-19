"""为 verl Vanilla GRPO 生成训练/开发/冒烟 Parquet 任务文件。

严格遵循项目评测防泄露边界：
- 训练任务仅使用 data/tasks/grpo_tasks.jsonl（task_id >= 1459），并校验与官方评测集、
  SFT 候选集、eval_dev_500 零交集；
- 按固定随机种子（默认 42）打乱训练任务并记录清单 manifest.json；
- 每行保留公开 system/user 占位提示词、data_source 与 extra_info.task_id，实际初始观察
  由 ShoppingAgentLoop 在 env.reset(task_id) 后写入。
"""

import argparse
import hashlib
import json
import pathlib
import random
from typing import Any, Dict, List, Set

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EVAL_TEMPLATE_YAML = REPO_ROOT / "env/single_eval/configs/standard/eval_local_b_ep2.yaml"

FALLBACK_SYSTEM_PROMPT = (
    "你在中文购物平台帮用户完成购买任务。每一轮你会看到当前页面的公开观察\n"
    "（JSON）：page_type 是页面类型，search_available 表示当前是否可以搜索，\n"
    "actions 列出当前页面的全部可点击项，搜索结果页还包含候选商品的摘要\n"
    "（标题、价格、部分规格）。\n\n"
    "规则：\n"
    "- 每轮必须恰好调用一个工具；点击的 value 必须原样来自最新观察的 actions，\n"
    "  搜索仅在 search_available 为 true 时可用。\n"
    "- 购买前打开候选商品核验价格、品牌、型号与规格属性；确认完整规格（variant）\n"
    "  的实际价格满足需求中的预算后再点击 Buy Now。Buy Now 不可撤销。\n"
    "- 经过多次有实质差异的搜索和候选核验后仍没有可接受商品时，调用\n"
    "  finish_without_purchase 结束。\n"
    "- 不要在正文里输出动作字符串；动作全部通过工具调用表达。\n"
)


def load_system_prompt() -> str:
    """读取与 v1 评测协议一致的系统提示词。"""
    if EVAL_TEMPLATE_YAML.is_file():
        cfg = yaml.safe_load(EVAL_TEMPLATE_YAML.read_text(encoding="utf-8"))
        prompt = (cfg or {}).get("agent_config", {}).get("system_prompt")
        if prompt:
            return prompt
    return FALLBACK_SYSTEM_PROMPT


def _read_jsonl_task_ids(path: pathlib.Path) -> List[int]:
    task_ids: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_ids.append(int(row["task_id"]))
    return task_ids


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_disjoint_splits(
    train_ids: List[int],
    val_ids: List[int],
    eval_standard_path: pathlib.Path,
    sft_candidates_path: pathlib.Path,
) -> None:
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("grpo_tasks.jsonl 中存在重复 task_id")
    if len(set(val_ids)) != len(val_ids):
        raise ValueError("eval_dev_500.jsonl 中存在重复 task_id")

    bad_min = [tid for tid in train_ids if tid <= 1458]
    if bad_min:
        raise ValueError(f"训练集包含 task_id <= 1458 的官方评测任务: {bad_min[:5]}")

    train_set: Set[int] = set(train_ids)
    val_set: Set[int] = set(val_ids)
    if train_set & val_set:
        raise ValueError(f"训练集与开发集存在重叠: {sorted(train_set & val_set)[:5]}")

    if eval_standard_path.is_file():
        official_eval = set(_read_jsonl_task_ids(eval_standard_path))
        overlap = train_set & official_eval
        if overlap:
            raise ValueError(f"训练集与官方评测集存在重叠: {sorted(overlap)[:5]}")

    if sft_candidates_path.is_file():
        sft_set = set(_read_jsonl_task_ids(sft_candidates_path))
        overlap = train_set & sft_set
        if overlap:
            raise ValueError(f"训练集与 SFT 候选集存在重叠: {sorted(overlap)[:5]}")


def _build_rows(
    task_ids: List[int],
    split_name: str,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, task_id in enumerate(task_ids):
        rows.append(
            {
                "data_source": "shop_simulator",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"task_id={task_id}"},
                ],
                "ability": "shopping",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "task_id": int(task_id),
                    "split": split_name,
                    "index": idx,
                },
            }
        )
    return rows


def _write_parquet(rows: List[Dict[str, Any]], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)


def prepare_rl_parquets(
    *,
    train_jsonl: pathlib.Path,
    val_jsonl: pathlib.Path,
    output_dir: pathlib.Path,
    seed: int = 42,
    val_limit: int = 100,
    smoke_train_limit: int = 8,
    smoke_val_limit: int = 2,
    eval_standard_jsonl: pathlib.Path = REPO_ROOT / "data/tasks/eval_standard_task_ids.jsonl",
    sft_candidate_jsonl: pathlib.Path = REPO_ROOT / "data/tasks/sft_candidate_tasks.jsonl",
) -> Dict[str, Any]:
    """校验防泄露边界并写出 train/val/smoke Parquet 与 manifest。"""
    train_ids = _read_jsonl_task_ids(train_jsonl)
    val_ids = _read_jsonl_task_ids(val_jsonl)
    _validate_disjoint_splits(train_ids, val_ids, eval_standard_jsonl, sft_candidate_jsonl)

    shuffled_train = list(train_ids)
    random.Random(seed).shuffle(shuffled_train)

    system_prompt = load_system_prompt()
    outputs = {
        "train.parquet": (shuffled_train, "train"),
        "val_dev500.parquet": (val_ids, "val_dev500"),
        f"val_dev{val_limit}.parquet": (val_ids[:val_limit], f"val_dev{val_limit}"),
        f"smoke_train{smoke_train_limit}.parquet": (
            shuffled_train[:smoke_train_limit],
            "smoke_train",
        ),
        f"smoke_val{smoke_val_limit}.parquet": (
            val_ids[:smoke_val_limit],
            "smoke_val",
        ),
    }

    written: Dict[str, Any] = {}
    for filename, (ids, split_name) in outputs.items():
        rows = _build_rows(ids, split_name, system_prompt)
        out_path = output_dir / filename
        _write_parquet(rows, out_path)
        written[filename] = {
            "count": len(rows),
            "split": split_name,
            "sha256": _sha256_file(out_path),
            "first_10_task_ids": ids[:10],
        }

    manifest = {
        "seed": seed,
        "train_source": str(train_jsonl),
        "train_source_sha256": _sha256_file(train_jsonl),
        "val_source": str(val_jsonl),
        "val_source_sha256": _sha256_file(val_jsonl),
        "train_min_task_id": min(shuffled_train),
        "train_max_task_id": max(shuffled_train),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "files": written,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成 verl Vanilla GRPO 训练与评测 Parquet 数据")
    parser.add_argument("--train-jsonl", type=pathlib.Path, default=REPO_ROOT / "data/tasks/grpo_tasks.jsonl")
    parser.add_argument("--val-jsonl", type=pathlib.Path, default=REPO_ROOT / "data/tasks/eval_dev_500.jsonl")
    parser.add_argument("--output-dir", type=pathlib.Path, default=REPO_ROOT / "data/rl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-limit", type=int, default=100)
    parser.add_argument("--smoke-train-limit", type=int, default=8)
    parser.add_argument("--smoke-val-limit", type=int, default=2)
    args = parser.parse_args(argv)

    manifest = prepare_rl_parquets(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        seed=args.seed,
        val_limit=args.val_limit,
        smoke_train_limit=args.smoke_train_limit,
        smoke_val_limit=args.smoke_val_limit,
    )
    print(json.dumps(manifest["files"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""为 verl Vanilla GRPO 与统一评测生成训练/开发/留出/冒烟 Parquet 任务文件。

严格遵循项目评测防泄露与价格可核验边界：
- 训练任务仅使用 data/tasks/grpo_tasks.jsonl（task_id >= 1459），并校验与官方评测集、
  SFT 候选集、eval_dev_474、eval_holdout_915 零交集；
- 当 eval_holdout_915.jsonl（或 eval_standard_task_ids.jsonl）存在时，导出冻结留出集
  eval_holdout{N}.parquet（剔除价格不可核验题目后的标准口径为 915 题）；
- 当 env/shop_env/data/items_eval_train.json 存在时，校验所有导出集合均不包含
  price-unverifiable（multiple_effective_price_axes / effective_price_axis_unselected）任务；
- 按固定随机种子（默认 42）打乱训练任务并记录清单 manifest.json；
- 每行保留公开 system/user 占位提示词、data_source 与 extra_info.task_id，实际初始观察
  由 ShoppingAgentLoop 在 env.reset(task_id) 后写入。
"""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import random
from typing import Any, Dict, List, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

SHOP_SYSTEM_PROMPT = (
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
    """返回与评测协议一致的规范系统提示词。"""
    return SHOP_SYSTEM_PROMPT


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


def _load_engine_module(filename: str, module_name: str):
    path = REPO_ROOT / "env/shop_env/web_agent_site/engine" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载环境模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_price_verifiability(
    task_groups: Dict[str, List[int]],
    items_json: Optional[pathlib.Path],
) -> None:
    if items_json is None or not items_json.is_file():
        return

    constraints_mod = _load_engine_module("constraints.py", "shop_rl._prep_constraints")
    variant_price_mod = _load_engine_module("variant_price.py", "shop_rl._prep_variant_price")
    explicit_budget_from_instruction = constraints_mod.explicit_budget_from_instruction
    normalize_option_text = variant_price_mod.normalize_option_text
    resolve_variant_price = variant_price_mod.resolve_variant_price

    items = json.loads(items_json.read_text(encoding="utf-8"))
    for split_label, task_ids in task_groups.items():
        bad: List[tuple[int, str]] = []
        for tid in task_ids:
            item = items[tid]
            inst = item["instructions"][0]
            budget = explicit_budget_from_instruction(inst["instruction"])
            if budget is None:
                continue
            selected: Dict[str, Any] = {}
            co = item.get("customization_options") or {}
            for goal_opt in inst.get("instruction_options") or []:
                goal_norm = normalize_option_text(goal_opt)
                for axis, entries in co.items():
                    for entry in entries or []:
                        if isinstance(entry, dict) and normalize_option_text(entry.get("value")) == goal_norm:
                            selected[axis] = entry["value"]
                            break
            pr = resolve_variant_price(item, selected)
            if pr.get("status") != "pass":
                bad.append((tid, str(pr.get("method"))))
        if bad:
            raise ValueError(
                f"{split_label} 包含 {len(bad)} 个价格不可核验任务（示例: {bad[:5]}）"
            )


def _validate_disjoint_splits(
    train_ids: List[int],
    val_ids: List[int],
    eval_standard_path: Optional[pathlib.Path],
    sft_candidates_path: Optional[pathlib.Path],
) -> Optional[List[int]]:
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("训练集 JSONL 中存在重复 task_id")
    if len(set(val_ids)) != len(val_ids):
        raise ValueError("开发集 JSONL 中存在重复 task_id")

    bad_min = [tid for tid in train_ids if tid <= 1458]
    if bad_min:
        raise ValueError(f"训练集包含 task_id <= 1458 的官方评测任务: {bad_min[:5]}")

    train_set: Set[int] = set(train_ids)
    val_set: Set[int] = set(val_ids)
    if train_set & val_set:
        raise ValueError(f"训练集与开发集存在重叠: {sorted(train_set & val_set)[:5]}")

    official_eval_ids: Optional[List[int]] = None
    if eval_standard_path is not None and eval_standard_path.is_file():
        official_eval_ids = _read_jsonl_task_ids(eval_standard_path)
        if len(set(official_eval_ids)) != len(official_eval_ids):
            raise ValueError("eval_standard_task_ids.jsonl 中存在重复 task_id")
        overlap = train_set & set(official_eval_ids)
        if overlap:
            raise ValueError(f"训练集与官方评测集存在重叠: {sorted(overlap)[:5]}")

    if sft_candidates_path is not None and sft_candidates_path.is_file():
        sft_set = set(_read_jsonl_task_ids(sft_candidates_path))
        overlap = train_set & sft_set
        if overlap:
            raise ValueError(f"训练集与 SFT 候选集存在重叠: {sorted(overlap)[:5]}")

    return official_eval_ids


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
    holdout_jsonl: Optional[pathlib.Path] = REPO_ROOT / "data/tasks/eval_holdout_915.jsonl",
    seed: int = 42,
    val_limit: int = 100,
    smoke20_limit: int = 20,
    smoke_train_limit: int = 8,
    smoke_val_limit: int = 2,
    eval_standard_jsonl: Optional[pathlib.Path] = REPO_ROOT / "data/tasks/eval_standard_task_ids.jsonl",
    sft_candidate_jsonl: Optional[pathlib.Path] = REPO_ROOT / "data/tasks/sft_candidate_tasks.jsonl",
    items_json: Optional[pathlib.Path] = REPO_ROOT / "env/shop_env/data/items_eval_train.json",
) -> Dict[str, Any]:
    """校验防泄露边界与价格可核验性，并写出 train/val/holdout/smoke Parquet 与 manifest。"""
    train_ids = _read_jsonl_task_ids(train_jsonl)
    val_ids = _read_jsonl_task_ids(val_jsonl)
    official_eval_ids = _validate_disjoint_splits(
        train_ids, val_ids, eval_standard_jsonl, sft_candidate_jsonl
    )

    val_set = set(val_ids)
    train_set = set(train_ids)
    holdout_ids: List[int] = []
    used_holdout_file: Optional[pathlib.Path] = None
    if holdout_jsonl is not None and holdout_jsonl.is_file():
        holdout_ids = _read_jsonl_task_ids(holdout_jsonl)
        used_holdout_file = holdout_jsonl
    elif official_eval_ids is not None:
        holdout_ids = [tid for tid in official_eval_ids if tid not in val_set]

    if holdout_ids:
        holdout_set = set(holdout_ids)
        if len(holdout_set) != len(holdout_ids):
            raise ValueError("holdout 切分中存在重复 task_id")
        if holdout_set & train_set:
            raise ValueError(f"holdout 与训练集存在重叠: {sorted(holdout_set & train_set)[:5]}")
        if holdout_set & val_set:
            raise ValueError(f"holdout 与开发集存在重叠: {sorted(holdout_set & val_set)[:5]}")
        if official_eval_ids is not None:
            official_set = set(official_eval_ids)
            if not holdout_set <= official_set:
                raise ValueError(f"holdout 包含非官方评测任务: {sorted(holdout_set - official_set)[:5]}")
            if not val_set <= official_set:
                raise ValueError(f"开发集包含非官方评测任务: {sorted(val_set - official_set)[:5]}")

    check_groups: Dict[str, List[int]] = {"train": train_ids, "val": val_ids}
    if holdout_ids:
        check_groups["holdout"] = holdout_ids
    _validate_price_verifiability(check_groups, items_json)

    shuffled_train = list(train_ids)
    random.Random(seed).shuffle(shuffled_train)

    system_prompt = load_system_prompt()
    val_split_name = f"val_dev{len(val_ids)}"
    outputs: Dict[str, tuple[List[int], str]] = {
        "train.parquet": (shuffled_train, "train"),
        f"{val_split_name}.parquet": (val_ids, val_split_name),
        f"val_dev{val_limit}.parquet": (val_ids[:val_limit], f"val_dev{val_limit}"),
        f"val_smoke{smoke20_limit}.parquet": (
            val_ids[:smoke20_limit],
            f"val_smoke{smoke20_limit}",
        ),
        f"smoke_train{smoke_train_limit}.parquet": (
            shuffled_train[:smoke_train_limit],
            "smoke_train",
        ),
        f"smoke_val{smoke_val_limit}.parquet": (
            val_ids[:smoke_val_limit],
            "smoke_val",
        ),
    }

    if holdout_ids:
        holdout_split = f"eval_holdout{len(holdout_ids)}"
        outputs[f"{holdout_split}.parquet"] = (holdout_ids, holdout_split)

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

    manifest: Dict[str, Any] = {
        "seed": seed,
        "train_source": str(train_jsonl),
        "train_source_sha256": _sha256_file(train_jsonl),
        "val_source": str(val_jsonl),
        "val_source_sha256": _sha256_file(val_jsonl),
    }
    if used_holdout_file is not None:
        manifest["holdout_source"] = str(used_holdout_file)
        manifest["holdout_source_sha256"] = _sha256_file(used_holdout_file)
    if eval_standard_jsonl is not None and eval_standard_jsonl.is_file():
        manifest["eval_standard_source"] = str(eval_standard_jsonl)
        manifest["eval_standard_source_sha256"] = _sha256_file(eval_standard_jsonl)
    manifest.update(
        {
            "train_min_task_id": min(shuffled_train),
            "train_max_task_id": max(shuffled_train),
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "files": written,
        }
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成 verl Vanilla GRPO 训练与评测 Parquet 数据")
    parser.add_argument("--train-jsonl", type=pathlib.Path, default=REPO_ROOT / "data/tasks/grpo_tasks.jsonl")
    parser.add_argument("--val-jsonl", type=pathlib.Path, default=REPO_ROOT / "data/tasks/eval_dev_474.jsonl")
    parser.add_argument(
        "--holdout-jsonl",
        type=pathlib.Path,
        default=REPO_ROOT / "data/tasks/eval_holdout_915.jsonl",
    )
    parser.add_argument(
        "--eval-standard-jsonl",
        type=pathlib.Path,
        default=REPO_ROOT / "data/tasks/eval_standard_task_ids.jsonl",
    )
    parser.add_argument(
        "--sft-candidate-jsonl",
        type=pathlib.Path,
        default=REPO_ROOT / "data/tasks/sft_candidate_tasks.jsonl",
    )
    parser.add_argument(
        "--items-json",
        type=pathlib.Path,
        default=REPO_ROOT / "env/shop_env/data/items_eval_train.json",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=REPO_ROOT / "data/rl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-limit", type=int, default=100)
    parser.add_argument("--smoke20-limit", type=int, default=20)
    parser.add_argument("--smoke-train-limit", type=int, default=8)
    parser.add_argument("--smoke-val-limit", type=int, default=2)
    args = parser.parse_args(argv)

    manifest = prepare_rl_parquets(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        holdout_jsonl=args.holdout_jsonl,
        seed=args.seed,
        val_limit=args.val_limit,
        smoke20_limit=args.smoke20_limit,
        smoke_train_limit=args.smoke_train_limit,
        smoke_val_limit=args.smoke_val_limit,
        eval_standard_jsonl=args.eval_standard_jsonl,
        sft_candidate_jsonl=args.sft_candidate_jsonl,
        items_json=args.items_json,
    )
    print(json.dumps(manifest["files"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

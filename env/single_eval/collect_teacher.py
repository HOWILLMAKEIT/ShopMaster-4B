"""Teacher 轨迹采集入口：任务划分、批量采集、回放校验与统计。

子命令（均在项目根目录运行）：
    python env/single_eval/collect_teacher.py prepare
    python env/single_eval/collect_teacher.py collect --yaml env/single_eval/configs/standard/teacher.yaml [--limit N] [--workers N]
    python env/single_eval/collect_teacher.py replay  --yaml ... [--limit N]
    python env/single_eval/collect_teacher.py stats   --yaml ...

prepare 按 docs/v0 设计（shop-grpo-v0.md 第一节）划分任务：排除与测试需求
重复的任务（含已核对的 5 条），训练需求去重，固定种子 42 按商品大类分层
抽取 6,000 个 SFT 采集候选，其余 15,950 个预留给 GRPO，保存清单供后续读取。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import run_task, get_finished_teacher_tasks  # noqa: E402
from env import ShopEnv  # noqa: E402
from tool_adapter import ToolCallError, tool_call_to_action  # noqa: E402

sys.path.insert(0, str(ROOT / "env"))
from get_score import is_success  # noqa: E402

TRAIN_FILE = ROOT / "data/fine_items_train_standard.jsonl"
EVAL_FILE = ROOT / "data/fine_items_eval_standard.jsonl"
FULL_ITEMS = ROOT / "env/shop_env/data/items_eval_train.json"
SFT_LIST = ROOT / "data/tasks/sft_candidate_tasks.jsonl"
GRPO_LIST = ROOT / "data/tasks/grpo_tasks.jsonl"
SUCCESS_LIST = ROOT / "data/sft/success_tasks.jsonl"

# 回放校验器版本：校验规则或环境 reward 解析变化时递增，使旧轨迹里
# 缓存的 replay_passed 失效并重新校验，避免旧判定永久免检。
REPLAY_CHECKER_VERSION = 2

# 与官方测试需求重复的训练任务，来源：docs/v0 设计第一节（此前已核对）。
KNOWN_OVERLAP_TASK_IDS = {5423, 7085, 8413, 9075, 13358}
SFT_TARGET = 6000
SPLIT_SEED = 42


def iter_json_array(path: Path, chunk_size: int = 1 << 20):
    """流式读取顶层 JSON 数组，与 scripts/data/align_task_ids.py 同一规则。"""
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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def instruction_of(item: Dict[str, Any]) -> str:
    instructions = item.get("instructions") or []
    return (instructions[0] or {}).get("instruction", "") if instructions else ""


def build_task_id_map() -> Dict[str, int]:
    """按 get_existed_goals 顺序建立全集 asin -> task_id。"""
    mapping: Dict[str, int] = {}
    next_id = 0
    for item in iter_json_array(FULL_ITEMS):
        for product in item.get("instructions") or []:
            if not (product.get("attributes") or []):
                continue
            mapping.setdefault(item["asin"], next_id)
            next_id += 1
    return mapping


def stratified_sample(groups: Dict[str, List[int]], total: int, rng: random.Random) -> set:
    """按组规模比例分层抽取 total 个 task_id（最大余数法补齐）。"""
    all_ids = [tid for ids in groups.values() for tid in ids]
    quotas = {key: int(len(ids) * total / len(all_ids)) for key, ids in groups.items()}
    assigned = sum(quotas.values())
    remainders = sorted(
        groups.items(), key=lambda kv: (len(kv[1]) * total / len(all_ids)) % 1, reverse=True
    )
    for i in range(total - assigned):
        quotas[remainders[i % len(remainders)][0]] += 1

    sampled: set = set()
    for key, ids in groups.items():
        ids_sorted = sorted(ids)
        sampled.update(rng.sample(ids_sorted, quotas[key]))
    return sampled


def cmd_prepare(args: argparse.Namespace) -> None:
    if not FULL_ITEMS.exists():
        raise SystemExit(f"缺少商品全集（可由 gz 解压重建）: {FULL_ITEMS}")

    eval_instructions = {instruction_of(row) for row in load_jsonl(EVAL_FILE)}
    task_id_map = build_task_id_map()

    train_rows = []
    overlap_with_eval = []
    missing_asin = 0
    for row in load_jsonl(TRAIN_FILE):
        asin = row["asin"]
        task_id = task_id_map.get(asin)
        if task_id is None:
            missing_asin += 1
            continue
        instruction = instruction_of(row)
        entry = {
            "task_id": task_id,
            "instruction": instruction,
            "domain_zh": row.get("domain_zh", ""),
        }
        if instruction and instruction in eval_instructions:
            overlap_with_eval.append(task_id)
            continue
        train_rows.append(entry)

    known_missing = KNOWN_OVERLAP_TASK_IDS - set(overlap_with_eval)
    if known_missing:
        print(f"警告：已知重复任务未检出与测试需求重复，请核对: {sorted(known_missing)}")

    # 训练需求去重：同一需求保留最小 task_id。
    by_instruction: Dict[str, Dict[str, Any]] = {}
    for entry in sorted(train_rows, key=lambda e: e["task_id"]):
        key = entry["instruction"]
        if key not in by_instruction:
            by_instruction[key] = entry
    deduped = sorted(by_instruction.values(), key=lambda e: e["task_id"])

    groups: Dict[str, List[int]] = defaultdict(list)
    for entry in deduped:
        groups[entry["domain_zh"] or "未知"].append(entry["task_id"])
    rng = random.Random(SPLIT_SEED)
    sft_ids = stratified_sample(groups, SFT_TARGET, rng)

    sft_rows = [e for e in deduped if e["task_id"] in sft_ids]
    grpo_rows = [e for e in deduped if e["task_id"] not in sft_ids]
    write_jsonl(SFT_LIST, sft_rows)
    write_jsonl(GRPO_LIST, grpo_rows)

    print(
        json.dumps(
            {
                "train_items": len(load_jsonl(TRAIN_FILE)),
                "overlap_with_eval_excluded": len(overlap_with_eval),
                "overlap_task_ids": sorted(overlap_with_eval),
                "missing_asin": missing_asin,
                "after_dedup": len(deduped),
                "sft_candidates": len(sft_rows),
                "grpo_reserved": len(grpo_rows),
                "domains": len(groups),
                "seed": SPLIT_SEED,
                "sft_list": str(SFT_LIST.relative_to(ROOT)),
                "grpo_list": str(GRPO_LIST.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_config(yaml_path: Path) -> Dict[str, Any]:
    with yaml_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def batch_dir(config: Dict[str, Any]) -> Path:
    agent_config = config["agent_config"]
    output_path = Path(agent_config["output_path"])
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    return output_path / agent_config["batch_name"]


def count_success(bdir: Path) -> int:
    """统计批次目录中通过 is_success 的任务数。"""
    count = 0
    for path in sorted(bdir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                if is_success(json.load(handle)):
                    count += 1
        except (OSError, json.JSONDecodeError):
            continue
    return count


def execute_batch(todo: List[int], config: Dict[str, Any], workers: int) -> List[Dict[str, Any]]:
    """执行一批任务并返回结果。"""
    results: List[Dict[str, Any]] = []
    if workers <= 1:
        for task_id in todo:
            results.append(run_task(task_id, config))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_task, tid, config): tid for tid in todo}
            for future in as_completed(futures):
                results.append(future.result())
    return results


def cmd_collect(args: argparse.Namespace) -> None:
    config = load_config(Path(args.yaml))
    # agent_config.task_file（相对仓库根）可指定任务清单；缺省用 SFT 候选清单。
    ac = config["agent_config"]
    task_path = ROOT / ac["task_file"] if ac.get("task_file") else SFT_LIST
    sft_rows = load_jsonl(task_path)
    finished = get_finished_teacher_tasks(str(batch_dir(config)))

    todo = [
        row["task_id"]
        for row in sft_rows
        if row["task_id"] not in finished
        or (finished.get(row["task_id"]) == "incomplete" and args.retry_incomplete)
    ]
    if args.limit:
        todo = todo[: args.limit]

    print(
        f"SFT 候选 {len(sft_rows)}，已完成（成功/失败跳过）"
        f" {sum(1 for s in finished.values() if s != 'incomplete')}，"
        f"本次执行 {len(todo)}，workers={args.workers}"
    )
    if not todo:
        return

    if args.target > 0:
        run_collection_loop(config, args, todo)
        return

    summarize(config, execute_batch(todo, config, args.workers))


def run_collection_loop(config: Dict[str, Any], args: argparse.Namespace, todo: List[int]) -> None:
    """无人值守采集：首轮 10 题单线程冒烟，之后并发采集至成功数达标。

    停止条件：达到 target、候选耗尽，或连续 2 轮没有任何新增成功
    （防止鉴权/余额类 fatal 空转烧钱）。停止后自动回放并写成功清单。
    """
    bdir = batch_dir(config)
    queue = list(todo)
    first_round = True
    zero_success_rounds = 0

    while queue:
        success_before = count_success(bdir)
        if success_before >= args.target:
            print(f"已达目标：{success_before}/{args.target} 条成功轨迹，停止调度")
            break

        if first_round:
            batch, workers = queue[:10], 1  # 冒烟：单线程 10 题
        else:
            batch, workers = queue[: max(args.workers * 10, 20)], args.workers
        queue = queue[len(batch):]

        print(
            f"[loop] 已成功 {success_before}/{args.target}，本轮执行 {len(batch)} 题"
            f"（workers={workers}），剩余候选 {len(queue)}"
        )
        results = execute_batch(batch, config, workers)

        gained = count_success(bdir) - success_before
        summarize(config, results)
        if first_round and gained == 0:
            print("[loop] 首轮 10 题全部未成功，停止采集，请检查轨迹文件")
            break
        zero_success_rounds = zero_success_rounds + 1 if gained == 0 else 0
        if zero_success_rounds >= 2:
            print("[loop] 连续 2 轮无新增成功，停止采集（检查失败原因与余额）")
            break
        first_round = False
    else:
        print(f"[loop] 候选耗尽，停止调度")

    final = count_success(bdir)
    print(f"[loop] 采集结束：成功 {final} 条。开始回放校验……")
    run_replay(config, limit=0)


def summarize(config: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    success = [r for r in results if is_success(r)]
    failure_reasons: Dict[str, int] = defaultdict(int)
    for r in results:
        if not is_success(r):
            failure_reasons[(r.get("failure_reason") or r.get("status") or "?").split(":")[0]] += 1
    usage = {
        "prompt_tokens": sum(r.get("token_usage_summary", {}).get("prompt_tokens", 0) for r in results),
        "completion_tokens": sum(
            r.get("token_usage_summary", {}).get("completion_tokens", 0) for r in results
        ),
    }
    print(
        json.dumps(
            {
                "executed": len(results),
                "success": len(success),
                "failure_reasons": dict(failure_reasons),
                "token_usage": usage,
                "avg_prompt_tokens_per_success": usage["prompt_tokens"] / len(success) if success else None,
                "avg_completion_tokens_per_success": usage["completion_tokens"] / len(success) if success else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _diff_state(actual: Any, stored: Any, path: str = "") -> List[str]:
    """逐字段比较回放观察与采集记录，返回差异描述列表。"""
    if type(actual) is not type(stored):
        return [f"{path or '<root>'}: 类型 {type(actual).__name__} != {type(stored).__name__}"]
    if isinstance(actual, dict):
        diffs = []
        for key in sorted(set(actual) | set(stored)):
            child = f"{path}.{key}" if path else key
            if key not in actual:
                diffs.append(f"{child}: 回放缺失")
            elif key not in stored:
                diffs.append(f"{child}: 采集记录缺失")
            else:
                diffs.extend(_diff_state(actual[key], stored[key], child))
        return diffs
    if isinstance(actual, list):
        if len(actual) != len(stored):
            return [f"{path or '<root>'}: 列表长度 {len(actual)} != {len(stored)}"]
        diffs = []
        for index, (a, s) in enumerate(zip(actual, stored)):
            diffs.extend(_diff_state(a, s, f"{path}[{index}]"))
        return diffs
    return [] if actual == stored else [f"{path or '<root>'}: {actual!r} != {stored!r}"]


PURCHASE_MATCH_FIELDS = ("asin", "price", "options", "attributes")


def replay_one(config: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """按 steps 重放一条轨迹：不调用 Teacher，只执行经转接层校验的动作。

    与旧版的区别：用回放实际得到的观察校验并执行下一动作（不再使用采集时
    的 observation_before）；逐步比较公开观察是否与采集记录一致；终局除
    reward_type 外，还核对购买商品、规格、价格与成功状态。
    """
    task_id = record["task_id"]
    shop_env: Optional[ShopEnv] = None
    result: Dict[str, Any] = {"task_id": task_id, "replay_passed": False}
    try:
        shop_env = ShopEnv(config["env_config"])
        reset_result = shop_env.reset(task_id)
        if reset_result.get("instruction") != record.get("instruction"):
            result["mismatch"] = "instruction 与采集时不一致（task_id 对应关系存疑）"
            return result

        # 实际回放观察：从 reset 返回开始，每步 interact 后更新。
        actual_state = reset_result.get("observation_state") or {}
        initial_diffs = _diff_state(actual_state, record.get("initial_observation_state") or {})
        if initial_diffs:
            result["state_mismatches"] = {"step": "initial", "diffs": initial_diffs[:10]}

        done_seen = False
        for step in record.get("steps", []):
            step_no = step.get("step")
            # 用实际观察校验动作合法性，并核对转换结果与采集时一致。
            action = tool_call_to_action(
                step.get("tool_name", ""), step.get("arguments") or {}, actual_state
            )
            if action != step.get("action"):
                result["mismatch"] = f"step {step_no}: 动作转换不一致（页面已偏离采集时的状态）"
                return result

            env_result = shop_env.interact(action)
            actual_state = env_result.get("observation_state") or {}
            diffs = _diff_state(actual_state, step.get("observation_after") or {})
            if diffs:
                result.setdefault("state_mismatches", {})[f"step_{step_no}"] = diffs[:10]

            if env_result.get("done") or env_result.get("over"):
                done_seen = True
                mismatch = _compare_terminal(env_result, record.get("terminal") or {})
                if mismatch:
                    result["mismatch"] = mismatch
                    return result
                break

        if not done_seen and (record.get("terminal") or {}).get("done"):
            result["mismatch"] = "回放耗尽 steps 未到达终局"
            return result

        # 公开状态逐步一致（或仅记录差异）且终局一致才算通过。
        result["replay_passed"] = "mismatch" not in result and not result.get("state_mismatches")
        return result
    except ToolCallError as exc:
        result["mismatch"] = f"动作在重放页面不合法: {exc}"
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if shop_env is not None and shop_env.env_idx is not None:
            try:
                shop_env.release()
            except Exception as release_exc:  # noqa: BLE001
                result["release_error"] = str(release_exc)


def _compare_terminal(env_result: Dict[str, Any], terminal: Dict[str, Any]) -> Optional[str]:
    """核对终局：reward_type、reward、reward_valid、purchase_success 与购买明细。"""
    replay_detail = env_result.get("reward_detail") or {}
    stored_detail = terminal.get("reward_detail") or {}
    for field in ("reward_type", "purchase_success"):
        if replay_detail.get(field) != stored_detail.get(field):
            return (
                f"终局不一致: {field} replay={replay_detail.get(field)!r} "
                f"stored={stored_detail.get(field)!r}"
            )
    if env_result.get("reward") != terminal.get("reward"):
        return (
            f"终局不一致: reward replay={env_result.get('reward')!r} "
            f"stored={terminal.get('reward')!r}"
        )
    if env_result.get("reward_valid") != terminal.get("reward_valid"):
        return (
            f"终局不一致: reward_valid replay={env_result.get('reward_valid')!r} "
            f"stored={terminal.get('reward_valid')!r}"
        )

    replay_purchase = env_result.get("purchase") or {}
    stored_purchase = terminal.get("purchase") or {}
    for field in PURCHASE_MATCH_FIELDS:
        if replay_purchase.get(field) != stored_purchase.get(field):
            return (
                f"终局不一致: purchase.{field} replay={replay_purchase.get(field)!r} "
                f"stored={stored_purchase.get(field)!r}"
            )
    return None


def run_replay(config: Dict[str, Any], limit: int = 0) -> None:
    """回放批次内成功轨迹，结果写回轨迹文件并汇总成功清单。

    缓存的通过判定只在 checker_version 与当前一致时复用；成功清单按
    本次批次内全部（含缓存复用的）通过记录整体重建，失败可被撤销。
    """
    bdir = batch_dir(config)
    rows = sorted(bdir.glob("*.json"))
    if limit:
        rows = rows[:limit]

    passed: List[Dict[str, Any]] = []
    results = []
    for path in rows:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if not is_success(record):
            continue
        cached = record.get("replay") or {}
        if cached.get("replay_passed") and cached.get("checker_version") == REPLAY_CHECKER_VERSION:
            passed.append({"task_id": record["task_id"], "file": cached["file"]})
            results.append(cached)
            continue
        outcome = replay_one(config, record)
        outcome["file"] = str(path.relative_to(ROOT))
        outcome["checker_version"] = REPLAY_CHECKER_VERSION
        record["replay"] = outcome
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        results.append(outcome)
        if outcome.get("replay_passed"):
            passed.append({"task_id": record["task_id"], "file": outcome["file"]})

    # 整体重建：清单始终等于当前批次内新校验通过的集合。
    write_jsonl(SUCCESS_LIST, passed)

    print(
        json.dumps(
            {
                "success_records": len(results),
                "replay_passed": len(passed),
                "mismatches": [r for r in results if not r.get("replay_passed")][:5],
                "success_list": str(SUCCESS_LIST.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_replay(args: argparse.Namespace) -> None:
    config = load_config(Path(args.yaml))
    run_replay(config, limit=args.limit)


def cmd_stats(args: argparse.Namespace) -> None:
    config = load_config(Path(args.yaml))
    bdir = batch_dir(config)
    records = []
    for path in sorted(bdir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            records.append(json.load(handle))
    summarize(config, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="划分 SFT / GRPO 任务清单")

    p_collect = sub.add_parser("collect", help="批量采集 Teacher 轨迹")
    p_collect.add_argument("--yaml", required=True)
    p_collect.add_argument("--limit", type=int, default=0, help="本次最多执行的任务数")
    p_collect.add_argument("--workers", type=int, default=1)
    p_collect.add_argument(
        "--target", type=int, default=0,
        help="成功轨迹目标数；>0 时进入无人值守驱动模式（首轮单线程冒烟，达标后自动回放）",
    )
    p_collect.add_argument("--retry-incomplete", action="store_true")

    p_replay = sub.add_parser("replay", help="回放成功轨迹并写 success_tasks.jsonl")
    p_replay.add_argument("--yaml", required=True)
    p_replay.add_argument("--limit", type=int, default=0)

    p_stats = sub.add_parser("stats", help="统计批次结果")
    p_stats.add_argument("--yaml", required=True)

    args = parser.parse_args()
    {"prepare": cmd_prepare, "collect": cmd_collect, "replay": cmd_replay, "stats": cmd_stats}[
        args.command
    ](args)


if __name__ == "__main__":
    main()

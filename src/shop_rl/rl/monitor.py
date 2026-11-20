"""RL 实时监控聚合器：增量读取 rollouts-*.jsonl，按步聚合购物行为与 GRPO 组信号并写入 TensorBoard。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
import math
import os
import pathlib
import statistics
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_ABSTAIN_REASONS = frozenset(
    {"early_abstain", "graceful_abstain", "abstain", "finish_without_purchase"}
)
_MAX_STEPS_REASONS = frozenset({"max_steps", "env_over"})
_NON_SCALAR_KEYS = frozenset({"global_step", "split", "termination_counts"})


def normalize_split(raw_split: Any) -> str:
    """将轨迹中的 split 规范化为 'val' 或 'train'。"""
    text = str(raw_split or "").strip().lower()
    if any(tok in text for tok in ("val", "dev", "test")):
        return "val"
    return "train"


def percentile(values: Sequence[float], q: float) -> float:
    """线性插值计算百分位数（0 <= q <= 100），空序列返回 0.0。"""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (max(0.0, min(100.0, float(q))) / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _is_valid_record(record: Dict[str, Any]) -> bool:
    return bool(record.get("reward_valid", 1)) and record.get("reward_score") is not None


def _is_purchased(record: Dict[str, Any]) -> bool:
    return int(record.get("purchased") or 0) == 1 or str(record.get("termination_reason") or "") == "purchase"


def load_rollout_records(log_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """从 log_dir 读取所有 rollouts-*.jsonl，容忍末尾半行尚未写完的 JSON。"""
    if not log_dir.exists():
        return []
    records: List[Dict[str, Any]] = []
    for path in sorted(log_dir.glob("rollouts-*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("读取轨迹日志失败 (%s): %s", path, exc)
            continue
        last_idx = len(lines) - 1
        for idx, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                if idx == last_idx:
                    logger.debug("忽略文件末尾未写完的半行 (%s): %s", path.name, exc)
                else:
                    logger.warning("跳过损坏的轨迹行 (%s:%d): %s", path.name, idx + 1, exc)
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _compute_group_metrics(
    records: Sequence[Dict[str, Any]], *, split: str, group_size: int
) -> Dict[str, Any]:
    """按 uid 计算 GRPO 同题组内信号统计（仅对 train split 计算完整组指标）。"""
    if split != "train":
        return {
            "num_groups_total": 0,
            "num_groups_complete": 0,
            "num_groups_incomplete": 0,
            "group_has_variance_ratio": None,
            "group_all_zero_ratio": None,
            "group_all_one_ratio": None,
            "group_same_fractional_ratio": None,
            "group_reward_std_mean": None,
        }

    by_uid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, rec in enumerate(records):
        uid = str(rec.get("uid") or "").strip() or f"__missing_uid_{idx}"
        by_uid[uid].append(rec)

    complete_scores: List[List[float]] = []
    for grp in by_uid.values():
        size_ok = len(grp) == group_size if group_size > 0 else len(grp) >= 2
        if size_ok and all(_is_valid_record(r) for r in grp):
            complete_scores.append([float(r["reward_score"]) for r in grp])

    n_total = len(by_uid)
    n_complete = len(complete_scores)
    n_incomplete = n_total - n_complete
    if n_complete == 0:
        return {
            "num_groups_total": n_total,
            "num_groups_complete": 0,
            "num_groups_incomplete": n_incomplete,
            "group_has_variance_ratio": None,
            "group_all_zero_ratio": None,
            "group_all_one_ratio": None,
            "group_same_fractional_ratio": None,
            "group_reward_std_mean": None,
        }

    has_var = 0
    all_zero = 0
    all_one = 0
    same_frac = 0
    stds: List[float] = []
    for scores in complete_scores:
        lo, hi = min(scores), max(scores)
        stds.append(float(statistics.pstdev(scores)))
        if hi - lo > 1e-9:
            has_var += 1
        elif hi <= 1e-9:
            all_zero += 1
        elif lo >= 1.0 - 1e-6:
            all_one += 1
        else:
            same_frac += 1

    return {
        "num_groups_total": n_total,
        "num_groups_complete": n_complete,
        "num_groups_incomplete": n_incomplete,
        "group_has_variance_ratio": has_var / n_complete,
        "group_all_zero_ratio": all_zero / n_complete,
        "group_all_one_ratio": all_one / n_complete,
        "group_same_fractional_ratio": same_frac / n_complete,
        "group_reward_std_mean": sum(stds) / n_complete,
    }


def _compute_termination_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    counts: Counter[str] = Counter()
    bucket_hits: Counter[str] = Counter()
    for rec in records:
        reason = str(rec.get("termination_reason") or "unknown")
        counts[reason] += 1
        if _is_purchased(rec):
            bucket_hits["purchase"] += 1
        elif reason in _ABSTAIN_REASONS:
            bucket_hits["abstain"] += 1
        elif reason == "invalid_tool_call":
            bucket_hits["invalid_tool"] += 1
        elif reason == "context_limit":
            bucket_hits["context_limit"] += 1
        elif reason == "finish_length_truncated":
            bucket_hits["length_truncated"] += 1
        elif reason == "repeat_loop":
            bucket_hits["repeat_loop"] += 1
        elif reason in _MAX_STEPS_REASONS:
            bucket_hits["max_steps"] += 1
        elif reason == "no_tool_call":
            bucket_hits["no_tool_call"] += 1

    denom = max(1, total)
    return {
        "term_purchase_rate": bucket_hits["purchase"] / denom,
        "term_abstain_rate": bucket_hits["abstain"] / denom,
        "term_invalid_tool_rate": bucket_hits["invalid_tool"] / denom,
        "term_context_limit_rate": bucket_hits["context_limit"] / denom,
        "term_length_truncated_rate": bucket_hits["length_truncated"] / denom,
        "term_repeat_loop_rate": bucket_hits["repeat_loop"] / denom,
        "term_max_steps_rate": bucket_hits["max_steps"] / denom,
        "term_no_tool_call_rate": bucket_hits["no_tool_call"] / denom,
        "termination_counts": dict(sorted(counts.items())),
    }


def _compute_cost_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    steps = [float(r.get("num_steps") or 0.0) for r in records]
    gen_tokens = [float(r.get("assistant_gen_tokens") or 0.0) for r in records]
    total_tokens = [
        float((r.get("prompt_tokens") or 0) + (r.get("response_tokens") or 0))
        for r in records
    ]
    elapsed = [float(r.get("elapsed_sec") or 0.0) for r in records]
    return {
        "num_steps_mean": sum(steps) / len(steps),
        "num_steps_p95": percentile(steps, 95.0),
        "assistant_gen_tokens_mean": sum(gen_tokens) / len(gen_tokens),
        "assistant_gen_tokens_p95": percentile(gen_tokens, 95.0),
        "total_tokens_mean": sum(total_tokens) / len(total_tokens),
        "total_tokens_p95": percentile(total_tokens, 95.0),
        "elapsed_sec_mean": sum(elapsed) / len(elapsed),
        "elapsed_sec_p95": percentile(elapsed, 95.0),
    }


def aggregate_step_records(
    records: Sequence[Dict[str, Any]],
    *,
    split: str,
    global_step: int,
    group_size: int = 8,
) -> Dict[str, Any]:
    """聚合同一 `(split, global_step)` 的全部轨迹记录。"""
    if not records:
        raise ValueError("aggregate_step_records 需要至少一条轨迹记录")

    total = len(records)
    valid = [r for r in records if _is_valid_record(r)]
    n_valid = len(valid)
    n_invalid = total - n_valid

    valid_scores = [float(r["reward_score"]) for r in valid]
    if n_valid > 0:
        reward_mean: Optional[float] = sum(valid_scores) / n_valid
        reward_std: Optional[float] = float(statistics.pstdev(valid_scores))
        zero_ratio: Optional[float] = sum(1 for s in valid_scores if s <= 1e-9) / n_valid
        one_ratio: Optional[float] = sum(1 for s in valid_scores if s >= 1.0 - 1e-6) / n_valid
        frac_ratio: Optional[float] = (
            sum(1 for s in valid_scores if 1e-9 < s < 1.0 - 1e-6) / n_valid
        )
        strict_success_rate: Optional[float] = (
            sum(int(bool(r.get("paper_strict_success", 0))) for r in valid) / n_valid
        )
        gold_purchase_rate: Optional[float] = (
            sum(int(bool(r.get("gold_purchase", 0))) for r in valid) / n_valid
        )
        purchased_valid = [r for r in valid if _is_purchased(r)]
        purchase_rate: Optional[float] = len(purchased_valid) / n_valid
    else:
        reward_mean = reward_std = zero_ratio = one_ratio = frac_ratio = None
        strict_success_rate = gold_purchase_rate = purchase_rate = None
        purchased_valid = []

    if purchased_valid:
        pur_r_type = _safe_mean([float(r.get("r_type") or 0.0) for r in purchased_valid])
        pur_r_att = _safe_mean([float(r.get("r_att") or 0.0) for r in purchased_valid])
        pur_r_option = _safe_mean([float(r.get("r_option") or 0.0) for r in purchased_valid])
        pur_r_price = _safe_mean([float(r.get("r_price") or 0.0) for r in purchased_valid])
        pur_strict = _safe_mean(
            [float(r.get("paper_strict_score") or r.get("reward_score") or 0.0) for r in purchased_valid]
        )
        over_budget: Optional[float] = (
            sum(1 for r in purchased_valid if float(r.get("r_price") or 0.0) <= 1e-9)
            / len(purchased_valid)
        )
    else:
        pur_r_type = pur_r_att = pur_r_option = pur_r_price = pur_strict = over_budget = None

    summary: Dict[str, Any] = {
        "global_step": int(global_step),
        "split": split,
        "num_trajectories": total,
        "num_valid": n_valid,
        "num_invalid": n_invalid,
        "valid_coverage_rate": n_valid / total,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "reward_zero_ratio": zero_ratio,
        "reward_fractional_ratio": frac_ratio,
        "reward_one_ratio": one_ratio,
        "paper_strict_success_rate": strict_success_rate,
        "gold_purchase_rate": gold_purchase_rate,
        "purchase_rate": purchase_rate,
        "purchased_r_type": pur_r_type,
        "purchased_r_att": pur_r_att,
        "purchased_r_option": pur_r_option,
        "purchased_r_price": pur_r_price,
        "purchased_strict_score": pur_strict,
        "purchased_over_budget_rate": over_budget,
    }
    summary.update(_compute_group_metrics(records, split=split, group_size=group_size))
    summary.update(_compute_termination_metrics(records))
    summary.update(_compute_cost_metrics(records))
    return summary


def aggregate_all_steps(
    records: Iterable[Dict[str, Any]], *, group_size: int = 8
) -> List[Dict[str, Any]]:
    """将混合轨迹记录按 `(global_step, split)` 分组并有序聚合。"""
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        raw_step = rec.get("global_step")
        if raw_step is None:
            continue
        try:
            step = int(raw_step)
        except (TypeError, ValueError):
            continue
        split = normalize_split(rec.get("split"))
        grouped[(step, split)].append(rec)

    rows: List[Dict[str, Any]] = []
    for step, split in sorted(grouped):
        rows.append(
            aggregate_step_records(
                grouped[(step, split)], split=split, global_step=step, group_size=group_size
            )
        )
    return rows


def _write_metrics_jsonl(out_jsonl: pathlib.Path, rows: Sequence[Dict[str, Any]]) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_jsonl.with_name(f".{out_jsonl.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_jsonl)


def _create_tb_writer(tb_dir: pathlib.Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        logger.warning("未安装 torch.utils.tensorboard，跳过 TensorBoard 写入: %s", exc)
        return None
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


def _flush_tb_row(writer, row: Dict[str, Any]) -> None:
    split = str(row["split"])
    step = int(row["global_step"])
    prefix = f"shop_{split}"
    for key, val in row.items():
        if key in _NON_SCALAR_KEYS or val is None:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            writer.add_scalar(f"{prefix}/{key}", float(val), global_step=step)


def run_monitor(
    log_dir: pathlib.Path,
    *,
    out_jsonl: Optional[pathlib.Path] = None,
    tb_dir: Optional[pathlib.Path] = None,
    enable_tb: bool = True,
    group_size: int = 8,
    expected_train_size: int = 0,
    watch: bool = False,
    interval: float = 15.0,
) -> List[Dict[str, Any]]:
    """执行一次或持续轮询聚合，写出 `shop_metrics.jsonl` 与 TensorBoard 标量。"""
    resolved_out = out_jsonl or (log_dir / "shop_metrics.jsonl")
    resolved_tb = tb_dir or (log_dir / "tensorboard")
    writer = _create_tb_writer(resolved_tb) if enable_tb else None

    flushed_steps: set[Tuple[int, str]] = set()
    prev_counts: Dict[Tuple[int, str], int] = {}
    stable_polls: Dict[Tuple[int, str], int] = {}
    last_snapshot_sig: Optional[List[Tuple[int, str, int]]] = None
    latest_rows: List[Dict[str, Any]] = []

    try:
        while True:
            records = load_rollout_records(log_dir)
            latest_rows = aggregate_all_steps(records, group_size=group_size)
            snapshot_sig = [
                (int(r["global_step"]), str(r["split"]), int(r["num_trajectories"]))
                for r in latest_rows
            ]
            if latest_rows and snapshot_sig != last_snapshot_sig:
                _write_metrics_jsonl(resolved_out, latest_rows)
                last_snapshot_sig = snapshot_sig

            if writer is not None and latest_rows:
                max_step = max(int(r["global_step"]) for r in latest_rows)
                flushed_any = False
                for row in latest_rows:
                    key = (int(row["global_step"]), str(row["split"]))
                    count = int(row["num_trajectories"])
                    if prev_counts.get(key) == count:
                        stable_polls[key] = stable_polls.get(key, 0) + 1
                    else:
                        stable_polls[key] = 0
                    prev_counts[key] = count

                    if key in flushed_steps:
                        continue
                    if not watch or key[0] < max_step:
                        ready = True
                    elif key[1] == "train" and expected_train_size > 0:
                        ready = count >= expected_train_size
                    else:
                        ready = stable_polls[key] >= 3
                    if ready:
                        _flush_tb_row(writer, row)
                        flushed_steps.add(key)
                        flushed_any = True
                if flushed_any:
                    writer.flush()

            if not watch:
                break
            time.sleep(max(0.5, float(interval)))
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在刷新最新监控结果后退出。")
        records = load_rollout_records(log_dir)
        latest_rows = aggregate_all_steps(records, group_size=group_size)
        if latest_rows:
            _write_metrics_jsonl(resolved_out, latest_rows)
            if writer is not None:
                for row in latest_rows:
                    key = (int(row["global_step"]), str(row["split"]))
                    if key not in flushed_steps:
                        _flush_tb_row(writer, row)
                        flushed_steps.add(key)
                writer.flush()
    finally:
        if writer is not None:
            writer.close()

    return latest_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="增量扫描 rollouts-*.jsonl，按步聚合购物指标与 GRPO 组信号并写入 TensorBoard。"
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        required=True,
        help="包含 rollouts-*.jsonl 的执行段日志目录",
    )
    parser.add_argument(
        "--out-jsonl",
        type=pathlib.Path,
        default=None,
        help="输出聚合指标 JSONL 路径（默认：<log-dir>/shop_metrics.jsonl）",
    )
    parser.add_argument(
        "--tb-dir",
        type=pathlib.Path,
        default=None,
        help="TensorBoard 日志目录（默认：<log-dir>/tensorboard）",
    )
    parser.add_argument("--no-tb", action="store_true", help="仅输出 shop_metrics.jsonl，不写 TensorBoard")
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="GRPO 每题采样轨迹数 G（用于完整组统计；设为 0 则按样本数 >= 2 视为完整组）",
    )
    parser.add_argument(
        "--expected-train-size",
        type=int,
        default=0,
        help="可选：每训练步期望总轨迹数（如 train_batch_size * G），达到即刷入 TensorBoard",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="执行一次聚合后退出（默认）")
    mode.add_argument("--watch", action="store_true", help="持续轮询写入实时监控曲线")
    parser.add_argument("--interval", type=float, default=15.0, help="--watch 轮询间隔秒数（默认 15.0）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    rows = run_monitor(
        args.log_dir,
        out_jsonl=args.out_jsonl,
        tb_dir=args.tb_dir,
        enable_tb=not args.no_tb,
        group_size=args.group_size,
        expected_train_size=args.expected_train_size,
        watch=bool(args.watch),
        interval=args.interval,
    )
    logger.info("完成聚合，共输出 %d 条 (step, split) 记录。", len(rows))
    return 0


if __name__ == "__main__":
    _rc = main()
    if _rc:
        raise SystemExit(_rc)

"""统一汇总 verl 评测轨迹（rollouts-*.jsonl），计算 paper_strict 指标与精确成对检验。

输入一个或多个 `shop_rl eval run` 产出目录（自动查找 `<dir>/rollouts/rollouts-*.jsonl` 或
`<dir>/rollouts-*.jsonl`），按 `task_id` 去重后统计有效覆盖率、严格成功率与 Wilson 95%
置信区间、已购子分、终止分布与开销分位，并在多组运行间计算精确双侧 McNemar 检验。
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from shop_rl.rl.monitor import load_rollout_records, percentile

_ABSTAIN_REASONS = frozenset(
    {"early_abstain", "graceful_abstain", "abstain", "finish_without_purchase"}
)
_MAX_STEPS_REASONS = frozenset({"max_steps", "env_over"})
_WILSON_Z_95 = 1.959963984540054


def resolve_rollouts_dir(run_path: pathlib.Path) -> pathlib.Path:
    """定位包含 rollouts-*.jsonl 的目录（优先 `<run_path>/rollouts`，其次 `<run_path>`）。"""
    nested = run_path / "rollouts"
    if nested.is_dir() and any(nested.glob("rollouts-*.jsonl")):
        return nested
    if run_path.is_dir() and any(run_path.glob("rollouts-*.jsonl")):
        return run_path
    if nested.is_dir():
        return nested
    return run_path


def load_task_ids(path: pathlib.Path) -> List[int]:
    """从 `.jsonl`（含 `task_id`）或 `.parquet`（含 `extra_info.task_id`）读取有序去重 task_id。"""
    suffix = path.suffix.lower()
    raw_ids: List[int] = []
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                if "task_id" not in obj:
                    raise ValueError(f"{path}:{line_no} 缺少 task_id 字段")
                raw_ids.append(int(obj["task_id"]))
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        names = set(table.column_names)
        if "extra_info" in names:
            for item in table.column("extra_info").to_pylist():
                if not isinstance(item, dict) or "task_id" not in item:
                    raise ValueError(f"{path} 的 extra_info 缺少 task_id 字段")
                raw_ids.append(int(item["task_id"]))
        elif "task_id" in names:
            for tid in table.column("task_id").to_pylist():
                raw_ids.append(int(tid))
        else:
            raise ValueError(f"{path} 不包含 extra_info 或 task_id 列: {sorted(names)}")
    else:
        raise ValueError(f"不支持的任务清单格式（仅支持 .jsonl/.parquet）: {path}")

    seen: Set[int] = set()
    ordered: List[int] = []
    for tid in raw_ids:
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    return ordered


def wilson_interval(
    successes: int, total: int, *, z: float = _WILSON_Z_95
) -> Optional[List[float]]:
    """计算二项比例的 Wilson 置信区间，返回 [0, 1] 内的二元列表；total <= 0 返回 None。"""
    if total <= 0:
        return None
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def mcnemar_exact_p(first_only: int, second_only: int) -> float:
    """使用精确二项分布（stdlib math.comb）计算双侧 McNemar 检验 p 值。"""
    if first_only < 0 or second_only < 0:
        raise ValueError(f"discordant 计数不能为负: first_only={first_only}, second_only={second_only}")
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    k = min(first_only, second_only)
    tail_sum = sum(math.comb(discordant, i) for i in range(k + 1))
    # Python 大整数除法会自动缩放位宽，避免 discordant > 1023 时 2**discordant 转浮点溢出。
    return min(1.0, 2.0 * (tail_sum / (1 << discordant)))


def _strict_score_of(record: Mapping[str, Any]) -> Optional[float]:
    val = record.get("paper_strict_score")
    if val is None:
        val = record.get("reward_score")
    if val is None:
        return None
    return float(val)


def _is_valid_record(record: Mapping[str, Any]) -> bool:
    return bool(record.get("reward_valid", 1)) and _strict_score_of(record) is not None


def _is_strict_success(record: Mapping[str, Any]) -> bool:
    raw = record.get("paper_strict_success")
    if raw is not None:
        return bool(int(raw)) if isinstance(raw, (int, bool)) else bool(raw)
    score = _strict_score_of(record)
    return score is not None and score >= 1.0 - 1e-6


def _is_purchased(record: Mapping[str, Any]) -> bool:
    return (
        int(record.get("purchased") or 0) == 1
        or str(record.get("termination_reason") or "") == "purchase"
    )


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def deduplicate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    allowed_task_ids: Optional[Set[int]] = None,
) -> Tuple[Dict[int, Dict[str, Any]], int, int]:
    """按 task_id 去重（保留最后一条），返回 `(by_task_id, duplicate_count, unexpected_count)`。"""
    by_task: Dict[int, Dict[str, Any]] = {}
    duplicate_count = 0
    unexpected_ids: Set[int] = set()

    for rec in records:
        raw_tid = rec.get("task_id")
        if raw_tid is None:
            continue
        tid = int(raw_tid)
        if allowed_task_ids is not None and tid not in allowed_task_ids:
            unexpected_ids.add(tid)
            continue
        if tid in by_task:
            duplicate_count += 1
        by_task[tid] = dict(rec)

    return by_task, duplicate_count, len(unexpected_ids)


def _compute_termination_breakdown(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    counts: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    for rec in records:
        reason = str(rec.get("termination_reason") or "unknown")
        counts[reason] += 1
        if _is_purchased(rec):
            buckets["purchase"] += 1
        elif reason in _ABSTAIN_REASONS:
            buckets["abstain"] += 1
        elif reason == "invalid_tool_call":
            buckets["invalid_tool"] += 1
        elif reason == "context_limit":
            buckets["context_limit"] += 1
        elif reason == "finish_length_truncated":
            buckets["length_truncated"] += 1
        elif reason == "repeat_loop":
            buckets["repeat_loop"] += 1
        elif reason in _MAX_STEPS_REASONS:
            buckets["max_steps"] += 1
        elif reason == "no_tool_call":
            buckets["no_tool_call"] += 1
        else:
            buckets["other"] += 1

    denom = max(1, total)
    rates = {
        key: (buckets[key] / denom if total > 0 else 0.0)
        for key in (
            "purchase",
            "abstain",
            "invalid_tool",
            "context_limit",
            "length_truncated",
            "repeat_loop",
            "max_steps",
            "no_tool_call",
            "other",
        )
    }
    return {
        "rates": rates,
        "counts": dict(sorted(counts.items())),
    }


def _compute_cost_stats(records: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    if not records:
        return {
            "num_steps_mean": None,
            "num_steps_p95": None,
            "assistant_gen_tokens_mean": None,
            "assistant_gen_tokens_p95": None,
            "total_tokens_mean": None,
            "total_tokens_p95": None,
            "elapsed_sec_mean": None,
            "elapsed_sec_p95": None,
        }
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


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str = "run",
    expected_task_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """对一组原始轨迹记录去重并计算 paper_strict 评测指标。"""
    expected_set = set(int(tid) for tid in expected_task_ids) if expected_task_ids is not None else None
    by_task, duplicate_count, unexpected_count = deduplicate_records(
        records, allowed_task_ids=expected_set
    )
    deduped = [by_task[tid] for tid in sorted(by_task)]

    expected_count: Optional[int] = len(expected_set) if expected_set is not None else None
    total_records = len(deduped)
    missing_count = (expected_count - total_records) if expected_count is not None else 0

    valid = [r for r in deduped if _is_valid_record(r)]
    valid_count = len(valid)
    invalid_count = total_records - valid_count

    coverage_denom = expected_count if expected_count is not None else total_records
    valid_coverage_rate = (valid_count / coverage_denom) if coverage_denom > 0 else 0.0

    if valid_count > 0:
        valid_scores = [_strict_score_of(r) for r in valid]
        score_mean: Optional[float] = sum(float(s) for s in valid_scores) / valid_count
        success_count = sum(1 for r in valid if _is_strict_success(r))
        success_rate: Optional[float] = success_count / valid_count
        success_ci = wilson_interval(success_count, valid_count)
        purchased_valid = [r for r in valid if _is_purchased(r)]
        purchase_rate: Optional[float] = len(purchased_valid) / valid_count
    else:
        score_mean = None
        success_count = 0
        success_rate = None
        success_ci = None
        purchased_valid = []
        purchase_rate = None

    full_denom_success_rate: Optional[float] = (
        (success_count / expected_count)
        if (expected_count is not None and expected_count > 0)
        else None
    )

    if purchased_valid:
        purchased_subscores: Dict[str, Optional[float]] = {
            "r_type": _safe_mean([float(r.get("r_type") or 0.0) for r in purchased_valid]),
            "r_att": _safe_mean([float(r.get("r_att") or 0.0) for r in purchased_valid]),
            "r_option": _safe_mean([float(r.get("r_option") or 0.0) for r in purchased_valid]),
            "r_price": _safe_mean([float(r.get("r_price") or 0.0) for r in purchased_valid]),
            "over_budget_rate": (
                sum(1 for r in purchased_valid if float(r.get("r_price") or 0.0) <= 1e-9)
                / len(purchased_valid)
            ),
        }
    else:
        purchased_subscores = {
            "r_type": None,
            "r_att": None,
            "r_option": None,
            "r_price": None,
            "over_budget_rate": None,
        }

    return {
        "label": label,
        "expected_tasks": expected_count,
        "raw_records": len(records),
        "duplicate_count": duplicate_count,
        "unexpected_count": unexpected_count,
        "total_records": total_records,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "missing_count": missing_count,
        "valid_coverage_rate": valid_coverage_rate,
        "paper_strict_score_mean": score_mean,
        "paper_strict_success_count": success_count,
        "paper_strict_success_rate": success_rate,
        "paper_strict_success_wilson_95": success_ci,
        "full_denominator_success_rate": full_denom_success_rate,
        "purchase_rate": purchase_rate,
        "purchased_count": len(purchased_valid),
        "purchased_subscores": purchased_subscores,
        "termination": _compute_termination_breakdown(deduped),
        "cost": _compute_cost_stats(deduped),
    }


def summarize_run(
    run_path: pathlib.Path,
    *,
    label: Optional[str] = None,
    expected_tasks_path: Optional[pathlib.Path] = None,
    expected_task_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """读取单个 verl 评测输出目录并汇总指标。"""
    resolved_run = pathlib.Path(run_path).expanduser().resolve()
    rollouts_dir = resolve_rollouts_dir(resolved_run)
    records = load_rollout_records(rollouts_dir)

    if expected_task_ids is None and expected_tasks_path is not None:
        expected_task_ids = load_task_ids(pathlib.Path(expected_tasks_path).expanduser().resolve())

    summary = summarize_records(
        records,
        label=label or resolved_run.name,
        expected_task_ids=expected_task_ids,
    )
    summary["run_dir"] = str(resolved_run)
    summary["rollouts_dir"] = str(rollouts_dir)
    if expected_tasks_path is not None:
        summary["expected_tasks_path"] = str(pathlib.Path(expected_tasks_path).expanduser().resolve())
    return summary


def compare_runs(
    first_label: str,
    first_records: Sequence[Mapping[str, Any]],
    second_label: str,
    second_records: Sequence[Mapping[str, Any]],
    *,
    allowed_task_ids: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """在两组运行的有效 task_id 交集上计算成对差值与精确双侧 McNemar 检验。"""
    first_by_task, _, _ = deduplicate_records(first_records, allowed_task_ids=allowed_task_ids)
    second_by_task, _, _ = deduplicate_records(second_records, allowed_task_ids=allowed_task_ids)

    paired_ids = sorted(
        tid
        for tid in (set(first_by_task) & set(second_by_task))
        if _is_valid_record(first_by_task[tid]) and _is_valid_record(second_by_task[tid])
    )
    n_paired = len(paired_ids)
    if n_paired == 0:
        return {
            "first": first_label,
            "second": second_label,
            "n_paired": 0,
            "first_success_rate": None,
            "second_success_rate": None,
            "delta_success_rate": None,
            "first_score_mean": None,
            "second_score_mean": None,
            "delta_score_mean": None,
            "first_only": 0,
            "second_only": 0,
            "both_success": 0,
            "both_fail": 0,
            "mcnemar_p_value": 1.0,
        }

    first_only = 0
    second_only = 0
    both_success = 0
    both_fail = 0
    first_scores: List[float] = []
    second_scores: List[float] = []

    for tid in paired_ids:
        rec_a = first_by_task[tid]
        rec_b = second_by_task[tid]
        succ_a = _is_strict_success(rec_a)
        succ_b = _is_strict_success(rec_b)
        if succ_a and succ_b:
            both_success += 1
        elif succ_a and not succ_b:
            first_only += 1
        elif not succ_a and succ_b:
            second_only += 1
        else:
            both_fail += 1
        first_scores.append(float(_strict_score_of(rec_a)))
        second_scores.append(float(_strict_score_of(rec_b)))

    first_succ_rate = (both_success + first_only) / n_paired
    second_succ_rate = (both_success + second_only) / n_paired
    first_score_mean = sum(first_scores) / n_paired
    second_score_mean = sum(second_scores) / n_paired

    return {
        "first": first_label,
        "second": second_label,
        "n_paired": n_paired,
        "first_success_rate": first_succ_rate,
        "second_success_rate": second_succ_rate,
        "delta_success_rate": second_succ_rate - first_succ_rate,
        "first_score_mean": first_score_mean,
        "second_score_mean": second_score_mean,
        "delta_score_mean": second_score_mean - first_score_mean,
        "first_only": first_only,
        "second_only": second_only,
        "both_success": both_success,
        "both_fail": both_fail,
        "mcnemar_p_value": mcnemar_exact_p(first_only, second_only),
    }


def _select_pairs(labels: Sequence[str], baseline: Optional[str]) -> List[Tuple[str, str]]:
    if baseline is not None:
        if baseline not in labels:
            raise ValueError(f"--baseline {baseline!r} 不在运行标签中: {list(labels)}")
        return [(baseline, other) for other in labels if other != baseline]
    pairs: List[Tuple[str, str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((labels[i], labels[j]))
    return pairs


def build_comparison_report(
    runs: Sequence[Tuple[str, pathlib.Path]],
    *,
    expected_tasks_path: Optional[pathlib.Path] = None,
    subsets: Optional[Mapping[str, pathlib.Path]] = None,
    baseline: Optional[str] = None,
) -> Dict[str, Any]:
    """加载多组 verl 评测目录并生成整体及子集汇总与成对检验报告。"""
    if not runs:
        raise ValueError("至少需要提供一个评测目录")

    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"运行标签存在重复: {labels}")

    pair_specs = _select_pairs(labels, baseline)

    expected_ids: Optional[List[int]] = None
    expected_set: Optional[Set[int]] = None
    if expected_tasks_path is not None:
        expected_ids = load_task_ids(pathlib.Path(expected_tasks_path).expanduser().resolve())
        expected_set = set(expected_ids)

    loaded_records: Dict[str, List[Dict[str, Any]]] = {}
    run_meta: Dict[str, Tuple[str, str]] = {}
    for label, path in runs:
        resolved_run = pathlib.Path(path).expanduser().resolve()
        rollouts_dir = resolve_rollouts_dir(resolved_run)
        loaded_records[label] = load_rollout_records(rollouts_dir)
        run_meta[label] = (str(resolved_run), str(rollouts_dir))

    def _summarize_group(
        group_ids: Optional[Sequence[int]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        group_set = set(group_ids) if group_ids is not None else None
        per_run: Dict[str, Dict[str, Any]] = {}
        for label in labels:
            s = summarize_records(
                loaded_records[label],
                label=label,
                expected_task_ids=group_ids,
            )
            s["run_dir"], s["rollouts_dir"] = run_meta[label]
            per_run[label] = s

        pairs: Dict[str, Dict[str, Any]] = {}
        for first, second in pair_specs:
            pairs[f"{first}_vs_{second}"] = compare_runs(
                first,
                loaded_records[first],
                second,
                loaded_records[second],
                allowed_task_ids=group_set,
            )
        return per_run, pairs

    overall_runs, overall_pairs = _summarize_group(expected_ids)

    subset_reports: Dict[str, Dict[str, Any]] = {}
    if subsets:
        for name, subset_path in subsets.items():
            resolved_sub = pathlib.Path(subset_path).expanduser().resolve()
            sub_ids = load_task_ids(resolved_sub)
            if expected_set is not None:
                sub_ids = [tid for tid in sub_ids if tid in expected_set]
            sub_runs, sub_pairs = _summarize_group(sub_ids)
            subset_reports[name] = {
                "path": str(resolved_sub),
                "expected_tasks": len(sub_ids),
                "runs": sub_runs,
                "pairs": sub_pairs,
            }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_tasks_path": (
            str(pathlib.Path(expected_tasks_path).expanduser().resolve())
            if expected_tasks_path is not None
            else None
        ),
        "expected_tasks": len(expected_ids) if expected_ids is not None else None,
        "baseline": baseline,
        "run_order": labels,
        "runs": overall_runs,
        "pairs": overall_pairs,
        "subsets": subset_reports,
    }


def _fmt_pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100.0:.2f}%"


def _fmt_delta_pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100.0:+.2f}pp"


def _fmt_float(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _fmt_ci(ci: Optional[Sequence[float]]) -> str:
    if not ci or len(ci) != 2:
        return "-"
    return f"[{ci[0] * 100.0:.1f}%, {ci[1] * 100.0:.1f}%]"


def _format_section_tables(
    run_order: Sequence[str],
    runs: Mapping[str, Mapping[str, Any]],
    pairs: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    lines: List[str] = [
        "| Run | Valid / Denom | Strict Succ | Wilson 95% CI | Full-Denom | Score | Purchase | Steps (p95) | GenTok (p95) |",
        "| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in run_order:
        r = runs[label]
        denom = r["expected_tasks"] if r["expected_tasks"] is not None else r["total_records"]
        cost = r["cost"]
        steps_str = (
            f"{cost['num_steps_mean']:.1f} ({cost['num_steps_p95']:.0f})"
            if cost["num_steps_mean"] is not None
            else "-"
        )
        tok_str = (
            f"{cost['assistant_gen_tokens_mean']:.0f} ({cost['assistant_gen_tokens_p95']:.0f})"
            if cost["assistant_gen_tokens_mean"] is not None
            else "-"
        )
        lines.append(
            f"| {label} | {r['valid_count']} / {denom} | {_fmt_pct(r['paper_strict_success_rate'])} | "
            f"{_fmt_ci(r['paper_strict_success_wilson_95'])} | {_fmt_pct(r['full_denominator_success_rate'])} | "
            f"{_fmt_float(r['paper_strict_score_mean'])} | {_fmt_pct(r['purchase_rate'])} | "
            f"{steps_str} | {tok_str} |"
        )

    if pairs:
        lines.append("")
        lines.append(
            "| Pair (A -> B) | N | dSucc | dScore | A-only | B-only | Both+ | Both- | McNemar p |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for pair in pairs.values():
            d_score = (
                "-"
                if pair["delta_score_mean"] is None
                else f"{pair['delta_score_mean']:+.4f}"
            )
            lines.append(
                f"| {pair['first']} -> {pair['second']} | {pair['n_paired']} | "
                f"{_fmt_delta_pct(pair['delta_success_rate'])} | {d_score} | "
                f"{pair['first_only']} | {pair['second_only']} | "
                f"{pair['both_success']} | {pair['both_fail']} | "
                f"{pair['mcnemar_p_value']:.4g} |"
            )
    return lines


def format_report_markdown(report: Mapping[str, Any]) -> str:
    """将汇总报告格式化为可读 Markdown 表格。"""
    run_order = report["run_order"]
    lines: List[str] = ["## Overall", ""]
    lines.extend(_format_section_tables(run_order, report["runs"], report["pairs"]))

    for sub_name, sub_data in (report.get("subsets") or {}).items():
        lines.extend(["", f"## Subset: {sub_name} (n={sub_data['expected_tasks']})", ""])
        lines.extend(_format_section_tables(run_order, sub_data["runs"], sub_data["pairs"]))

    return "\n".join(lines)


def _parse_named_path(spec: str, flag_name: str) -> Tuple[str, pathlib.Path]:
    if "=" not in spec:
        raise ValueError(f"{flag_name} 需要 LABEL=PATH 格式，收到: {spec!r}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"{flag_name} 的名称与路径均不能为空: {spec!r}")
    return name, pathlib.Path(raw_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="汇总一个或多个 verl 评测目录的 paper_strict 指标并执行精确成对检验。"
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="可重复：指定评测标签与输出目录（支持目录根或 rollouts/ 子目录）",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        help="可选位置参数：评测输出目录（默认使用目录名作为标签）",
    )
    parser.add_argument(
        "--expected-tasks",
        type=pathlib.Path,
        default=None,
        help="可选期望任务清单（.jsonl 含 task_id，或 .parquet 含 extra_info.task_id）",
    )
    parser.add_argument(
        "--subset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="可重复：按子集任务清单分别输出指标与成对检验",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="可选基线运行标签：仅计算 baseline 对其余各组的成对比较（默认两两全比较）",
    )
    parser.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=None,
        help="可选：将完整汇总 JSON 写入指定路径",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    runs: List[Tuple[str, pathlib.Path]] = []
    try:
        for spec in args.run:
            runs.append(_parse_named_path(spec, "--run"))
        for path in args.paths:
            runs.append((path.resolve().name, path))
        if not runs:
            parser.error("至少需要通过 --run LABEL=PATH 或位置参数提供一个评测目录")

        subsets: Dict[str, pathlib.Path] = {}
        for spec in args.subset:
            name, sub_path = _parse_named_path(spec, "--subset")
            if name in subsets:
                raise ValueError(f"--subset 名称重复: {name}")
            subsets[name] = sub_path

        report = build_comparison_report(
            runs,
            expected_tasks_path=args.expected_tasks,
            subsets=subsets or None,
            baseline=args.baseline,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(format_report_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""面向已启动 SGLang `/generate` 的独立多轮购物评测器（零 verl 依赖）。

使用与 `ShoppingAgentLoop` 完全一致的本地 `AutoTokenizer.apply_chat_template`
（`tools=SHOP_TOOL_SCHEMAS`, `enable_thinking=...`）与连续 Token 追加策略，
直接向 SGLang `/generate` 发送精确 `input_ids`，避免 OpenAI `/v1/chat/completions`
注入 `"strict": false` 导致 system prompt 偏移，并用 SGLang `qwen3_coder` 解析器与
`tool_adapter.parse_tool_calls` 执行 1:1 校验，输出 `rollouts/rollouts-0.jsonl`、
`eval_meta.json` 与 `eval_summary.json`。
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import httpx
from sglang.srt.entrypoints.openai.protocol import Tool as SGLangTool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from transformers import AutoTokenizer

from shop_rl.eval.summarize import load_task_ids, summarize_records
from shop_rl.rl.monitor import load_rollout_records
from shop_rl.rl.prepare_tasks import load_system_prompt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load_single_eval_module(rel_path: str, module_name: str):
    full_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 single_eval 模块: {full_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_env_mod = _load_single_eval_module("env/single_eval/env.py", "shop_rl._single_eval_env")
_tool_mod = _load_single_eval_module(
    "env/single_eval/tool_adapter.py", "shop_rl._single_eval_tool_adapter"
)

ShopEnv = _env_mod.ShopEnv
SHOP_TOOL_SCHEMAS = _tool_mod.SHOP_TOOL_SCHEMAS
ToolCallError = _tool_mod.ToolCallError
parse_tool_calls = _tool_mod.parse_tool_calls
DEFAULT_SYSTEM_PROMPT = load_system_prompt()


def _format_initial_user_message(
    instruction: str, observation_state: Dict[str, Any]
) -> str:
    return (
        f"购物任务需求：\n{instruction}\n\n"
        f"当前观察（公开状态）：\n"
        f"{json.dumps(observation_state, ensure_ascii=False)}"
    )


def _extract_terminal_metrics(
    termination_reason: str,
    terminal_detail: Dict[str, Any],
    purchase: Dict[str, Any],
    *,
    allow_unverifiable_purchase: bool = True,
) -> Tuple[float, Dict[str, float | int | bool]]:
    purchased = bool(purchase) or termination_reason == "purchase"

    if purchased:
        r_type = float(terminal_detail.get("r_type") or 0.0)
        r_att = float(terminal_detail.get("r_att") or 0.0)
        r_option = float(terminal_detail.get("r_option") or 0.0)
        reward_valid = bool(terminal_detail.get("reward_valid", False))
        strict_score = terminal_detail.get("paper_strict_score")
        if not reward_valid or strict_score is None:
            if not allow_unverifiable_purchase:
                raise RuntimeError(f"paper_strict 购买终局不可核验: {terminal_detail}")
            r_price = 0.0
            strict_success = 0
            reward_score = 0.0
            reward_valid_flag = 0
        else:
            r_price = float(terminal_detail.get("r_price") or 0.0)
            strict_success = int(bool(terminal_detail.get("paper_strict_success", False)))
            reward_score = float(strict_score)
            reward_valid_flag = 1
    else:
        r_type = 0.0
        r_att = 0.0
        r_option = 0.0
        r_price = 0.0
        strict_success = 0
        reward_score = 0.0
        reward_valid_flag = 1

    extras: Dict[str, float | int | bool] = {
        "paper_strict_score": reward_score,
        "paper_strict_success": strict_success,
        "purchased": int(purchased),
        "r_type": r_type,
        "r_att": r_att,
        "r_option": r_option,
        "r_price": r_price,
        "reward_valid": reward_valid_flag,
    }
    return reward_score, extras


def _apply_template_ids(
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> List[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=SHOP_TOOL_SCHEMAS,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    if isinstance(rendered, dict) or hasattr(rendered, "keys"):
        rendered = rendered["input_ids"]
    return [int(x) for x in rendered]


def _compute_stop_token_ids(tokenizer: Any) -> Tuple[List[int], Optional[int]]:
    stop_ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, int) and eos_id >= 0:
        stop_ids.add(int(eos_id))
    elif isinstance(eos_id, (list, tuple, set)):
        for tid in eos_id:
            if isinstance(tid, int) and tid >= 0:
                stop_ids.add(int(tid))
    im_end_id: Optional[int] = None
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0:
            stop_ids.add(int(tid))
            if tok_str == "<|im_end|>":
                im_end_id = int(tid)
    return sorted(stop_ids), im_end_id


def _build_assistant_messages(
    normal_text: str, parsed_items: List[Any], step_idx: int
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """分别构造供 `parse_tool_calls` 校验与供 HF `apply_chat_template` 渲染的 assistant 消息。"""
    content = (normal_text or "").strip()
    if not parsed_items:
        msg = {"role": "assistant", "content": content}
        return msg, msg

    calls_for_validator: List[Dict[str, Any]] = []
    calls_for_template: List[Dict[str, Any]] = []
    for idx, item in enumerate(parsed_items):
        call_id = f"call_{step_idx}_{idx}"
        raw_params = item.parameters if item.parameters is not None else ""
        calls_for_validator.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": item.name, "arguments": raw_params},
            }
        )
        try:
            tmpl_args: Any = json.loads(raw_params) if raw_params.strip() else {}
        except Exception:
            tmpl_args = raw_params
        calls_for_template.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": item.name, "arguments": tmpl_args},
            }
        )

    msg_validator = {"role": "assistant", "content": content, "tool_calls": calls_for_validator}
    msg_template = {"role": "assistant", "content": content, "tool_calls": calls_for_template}
    return msg_validator, msg_template


def _strip_end_special_tokens(text: str) -> str:
    for tok in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(tok):
            text = text[: -len(tok)]
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "")


class _ContextExceededError(Exception):
    """SGLang 返回上下文长度超出限制。"""


async def _sglang_generate(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: Dict[str, Any],
    *,
    max_retries: int = 3,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(endpoint, json=payload)
            if resp.status_code == 400:
                body_lower = resp.text.lower()
                if any(
                    tok in body_lower
                    for tok in ("context length", "maximum context", "too many tokens", "max_new_tokens")
                ):
                    raise _ContextExceededError(resp.text)
                resp.raise_for_status()
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"SGLang 5xx ({resp.status_code}): {resp.text[:300]}",
                    request=resp.request,
                    response=resp,
                )
            resp.raise_for_status()
            return resp.json()
        except _ContextExceededError:
            raise
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            if attempt + 1 >= max_retries:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
    assert last_exc is not None
    raise last_exc


async def _run_one_task(
    task_id: int,
    *,
    client: httpx.AsyncClient,
    generate_endpoint: str,
    env_url: str,
    tokenizer: Any,
    sglang_tools: List[SGLangTool],
    stop_token_ids: List[int],
    im_end_id: Optional[int],
    split_name: str,
    max_model_len: int,
    turn_max_new_tokens: int,
    max_steps: int,
    temperature: float,
    enable_thinking: bool,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    started_at = time.monotonic()
    shop_env = ShopEnv({"base_url": env_url, "if_persona": False})

    termination_reason = "unknown"
    invalid_reason: Optional[str] = None
    terminal_detail: Dict[str, Any] = {}
    purchase: Dict[str, Any] = {}
    action_trace: List[str] = []
    subpage_clicks = 0
    num_steps = 0
    initial_prompt_tokens = 0
    assistant_gen_tokens = 0
    runtime_token_ids: List[int] = []

    async with sem:
        try:
            reset_res = await loop.run_in_executor(None, shop_env.reset, task_id)
            instruction = reset_res["instruction"]
            observation_state = reset_res.get("observation_state") or {}

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _format_initial_user_message(instruction, observation_state),
                },
            ]
            runtime_token_ids = _apply_template_ids(
                tokenizer,
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            initial_prompt_tokens = len(runtime_token_ids)

            for step_idx in range(max_steps):
                remaining_model = max_model_len - len(runtime_token_ids) - 1
                if step_idx > 0 and remaining_model < 16:
                    termination_reason = "context_limit"
                    break
                if remaining_model <= 0:
                    raise ValueError(
                        f"首轮可用生成预算不足 (task_id={task_id}, "
                        f"prompt_tokens={len(runtime_token_ids)}, max_model_len={max_model_len})"
                    )
                turn_budget = min(turn_max_new_tokens, remaining_model)

                payload = {
                    "input_ids": runtime_token_ids,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": turn_budget,
                        "stop_token_ids": stop_token_ids,
                    },
                }

                try:
                    data = await _sglang_generate(client, generate_endpoint, payload)
                except _ContextExceededError:
                    termination_reason = "context_limit"
                    break

                meta_info = data.get("meta_info") or {}
                output_ids = [
                    int(x) for x in (data.get("output_ids") or meta_info.get("output_ids") or [])
                ]
                finish_raw = meta_info.get("finish_reason") or {}
                finish_type = (
                    finish_raw.get("type") if isinstance(finish_raw, dict) else str(finish_raw)
                )
                hit_turn_len_cap = (finish_type == "length") or (len(output_ids) >= turn_budget)

                assistant_gen_tokens += len(output_ids)
                if output_ids:
                    runtime_token_ids.extend(output_ids)
                if (
                    finish_type == "stop"
                    and im_end_id is not None
                    and (not output_ids or output_ids[-1] != im_end_id)
                ):
                    runtime_token_ids.append(im_end_id)

                if len(runtime_token_ids) >= max_model_len:
                    termination_reason = "context_limit"
                    break

                raw_text = _strip_end_special_tokens(
                    tokenizer.decode(output_ids, skip_special_tokens=False)
                )
                parser = FunctionCallParser(tools=sglang_tools, tool_call_parser="qwen3_coder")
                normal_text, parsed_items = parser.parse_non_stream(raw_text)
                msg_validator, msg_template = _build_assistant_messages(
                    normal_text, parsed_items, step_idx
                )

                try:
                    parsed_calls = parse_tool_calls(msg_validator, observation_state)
                except ToolCallError as exc:
                    if hit_turn_len_cap:
                        termination_reason = "finish_length_truncated"
                    else:
                        termination_reason = "invalid_tool_call"
                        invalid_reason = str(exc)
                    break

                if not parsed_calls:
                    termination_reason = (
                        "finish_length_truncated" if hit_turn_len_cap else "no_tool_call"
                    )
                    break

                tool_call_id, action, _arguments, _tool_name = parsed_calls[0]
                action_trace.append(action)
                num_steps += 1

                env_res = await loop.run_in_executor(None, shop_env.interact, action)
                observation_state = env_res.get("observation_state") or {}
                if observation_state.get("page_type") == "information_subpage":
                    subpage_clicks += 1
                done = bool(env_res.get("done", False))
                over = bool(env_res.get("over", False))

                if done or over:
                    terminal_detail = env_res.get("reward_detail") or {}
                    purchase = env_res.get("purchase") or {}
                    termination_reason = str(
                        env_res.get("termination_reason")
                        or terminal_detail.get("termination_reason")
                        or ("purchase" if purchase else "env_over")
                    )
                    break

                if step_idx + 1 >= max_steps:
                    termination_reason = "max_steps"
                    break

                messages.append(msg_template)
                prev_messages = list(messages)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(observation_state, ensure_ascii=False),
                    }
                )

                prev_ids = _apply_template_ids(
                    tokenizer,
                    prev_messages,
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                )
                next_ids = _apply_template_ids(
                    tokenizer,
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                if im_end_id is not None and im_end_id in prev_ids:
                    last_im_end = len(prev_ids) - 1 - prev_ids[::-1].index(im_end_id)
                    obs_delta = next_ids[last_im_end + 1 :]
                else:
                    obs_delta = next_ids[len(prev_ids) :]

                if len(runtime_token_ids) + len(obs_delta) + 16 > max_model_len:
                    termination_reason = "context_limit"
                    break
                runtime_token_ids.extend(obs_delta)
        finally:
            if shop_env.env_idx is not None:
                await loop.run_in_executor(None, shop_env.release)

    reward_score, reward_extras = _extract_terminal_metrics(
        termination_reason,
        terminal_detail,
        purchase,
        allow_unverifiable_purchase=True,
    )
    response_tokens = max(assistant_gen_tokens, len(runtime_token_ids) - initial_prompt_tokens)
    return {
        "timestamp": time.time(),
        "elapsed_sec": round(time.monotonic() - started_at, 3),
        "global_step": 0,
        "split": split_name,
        "task_id": task_id,
        "termination_reason": termination_reason,
        "invalid_reason": invalid_reason,
        "reward_type": str(terminal_detail.get("reward_type") or ""),
        "num_steps": num_steps,
        "prompt_tokens": initial_prompt_tokens,
        "response_tokens": response_tokens,
        "assistant_gen_tokens": assistant_gen_tokens,
        "reward_score": reward_score,
        **reward_extras,
        **(
            {"paper_strict_score": None}
            if reward_extras["reward_valid"] == 0
            else {}
        ),
        "is_invalid_tool": int(termination_reason == "invalid_tool_call"),
        "is_context_limit": int(termination_reason == "context_limit"),
        "subpage_clicks": int(subpage_clicks),
        "action_trace": action_trace,
    }


def _parse_bool(value: str) -> bool:
    norm = value.strip().lower()
    if norm in {"1", "true", "yes", "y"}:
        return True
    if norm in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value!r}")


async def _run_eval(args: argparse.Namespace) -> int:
    if not args.model_path:
        raise ValueError("--model-path 不能为空，用于加载离线 AutoTokenizer")
    tasks_path = args.tasks.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    rollouts_dir = out_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    rollout_file = rollouts_dir / "rollouts-0.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    stop_token_ids, im_end_id = _compute_stop_token_ids(tokenizer)
    sglang_tools = [SGLangTool.model_validate(t) for t in SHOP_TOOL_SCHEMAS]

    all_task_ids = load_task_ids(tasks_path)
    if args.limit > 0:
        all_task_ids = all_task_ids[: args.limit]

    done_ids: Set[int] = set()
    if args.resume and any(rollouts_dir.glob("rollouts-*.jsonl")):
        existing = load_rollout_records(rollouts_dir)
        done_ids = {int(r["task_id"]) for r in existing if r.get("task_id") is not None}
    elif not args.resume and rollout_file.exists():
        rollout_file.unlink()

    pending_ids = [tid for tid in all_task_ids if tid not in done_ids]
    print(
        f"[serve_eval] label={args.label} total={len(all_task_ids)} "
        f"resumed={len(done_ids)} pending={len(pending_ids)} concurrency={args.concurrency}",
        flush=True,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    base_root = args.base_url.rstrip("/")
    if base_root.endswith("/v1"):
        base_root = base_root[: -len("/v1")]
    generate_endpoint = base_root + "/generate"

    if pending_ids:
        sem = asyncio.Semaphore(args.concurrency)
        write_lock = asyncio.Lock()
        completed = 0
        success_so_far = 0
        valid_so_far = 0

        limits = httpx.Limits(
            max_connections=max(16, args.concurrency * 2),
            max_keepalive_connections=max(8, args.concurrency),
        )
        timeout = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(
            limits=limits, timeout=timeout, trust_env=False
        ) as client:
            with rollout_file.open("a", encoding="utf-8") as writer:

                async def _worker(tid: int) -> None:
                    nonlocal completed, success_so_far, valid_so_far
                    record = await _run_one_task(
                        tid,
                        client=client,
                        generate_endpoint=generate_endpoint,
                        env_url=args.env_url.rstrip("/"),
                        tokenizer=tokenizer,
                        sglang_tools=sglang_tools,
                        stop_token_ids=stop_token_ids,
                        im_end_id=im_end_id,
                        split_name=args.split,
                        max_model_len=args.max_model_len,
                        turn_max_new_tokens=args.turn_max_new_tokens,
                        max_steps=args.max_steps,
                        temperature=args.temperature,
                        enable_thinking=args.enable_thinking,
                        sem=sem,
                    )
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    async with write_lock:
                        writer.write(line)
                        writer.flush()
                        completed += 1
                        if record.get("reward_valid") == 1:
                            valid_so_far += 1
                            if record.get("paper_strict_success") == 1:
                                success_so_far += 1
                        if completed % 25 == 0 or completed == len(pending_ids):
                            elapsed = time.monotonic() - t0
                            rate = (success_so_far / valid_so_far * 100.0) if valid_so_far else 0.0
                            print(
                                f"[serve_eval] {args.label}: {completed}/{len(pending_ids)} "
                                f"({elapsed:.1f}s, strict_succ={success_so_far}/{valid_so_far}={rate:.2f}%)",
                                flush=True,
                            )

                await asyncio.gather(*(_worker(tid) for tid in pending_ids))

    finished_at = datetime.now(timezone.utc).isoformat()
    elapsed_sec = round(time.monotonic() - t0, 3)

    meta = {
        "label": args.label,
        "mode": "sglang_standalone_generate",
        "model": args.model,
        "model_path": args.model_path,
        "base_url": base_root,
        "env_url": args.env_url,
        "tasks_path": str(tasks_path),
        "split": args.split,
        "expected_tasks": len(all_task_ids),
        "concurrency": args.concurrency,
        "max_model_len": args.max_model_len,
        "turn_max_new_tokens": args.turn_max_new_tokens,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_sec": elapsed_sec,
        "command": sys.argv,
    }
    (out_dir / "eval_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    records = load_rollout_records(rollouts_dir)
    summary = summarize_records(
        records, label=args.label, expected_task_ids=all_task_ids
    )
    summary["run_dir"] = str(out_dir)
    summary["rollouts_dir"] = str(rollouts_dir)
    summary["expected_tasks_path"] = str(tasks_path)
    (out_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[serve_eval] DONE {args.label}: valid={summary['valid_count']}/{summary['expected_tasks']} "
        f"strict_succ={summary['paper_strict_success_count']} "
        f"({(summary['paper_strict_success_rate'] or 0.0) * 100.0:.2f}%)",
        flush=True,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="独立 SGLang /generate 多轮购物评测器。")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="default")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-url", default="http://127.0.0.1:5705")
    parser.add_argument("--tasks", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", default="eval_holdout915")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--turn-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", type=_parse_bool, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return asyncio.run(_run_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())

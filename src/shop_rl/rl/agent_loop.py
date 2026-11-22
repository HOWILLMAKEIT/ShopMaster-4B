"""verl ShoppingAgentLoop：多步 ShopSimulator 交互与 Continuous Token 对齐。

每条轨迹独占一个 ShopEnv 会话（`reset -> interact* -> release`），由工作进程级
`asyncio.Semaphore` 限制同时占用环境服务的并发槽位数；严格按 `paper_strict` 口径返回
终局标量奖励，并把购物行为诊断写入 `rollouts-<pid>.jsonl` 与 `reward_extra_info`。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolSchema,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from shop_rl.rl.prepare_tasks import load_system_prompt

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load_single_eval_module(rel_path: str, module_name: str):
    """按绝对文件路径加载 single_eval 模块，避免 Ray worker 内 sys.path 冲突。"""
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


def _append_jsonl_record(log_dir: Optional[str], record: Dict[str, Any]) -> None:
    if not log_dir:
        return
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = pathlib.Path(log_dir) / f"rollouts-{os.getpid()}.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@register("shopping_agent")
class ShoppingAgentLoop(AgentLoopBase):
    """对接 ShopSimulator HTTP 环境服务的多轮购物 AgentLoop。"""

    _loop_semaphores: Dict[int, asyncio.Semaphore] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length: int = int(self.rollout_config.prompt_length)
        self.response_length: int = int(self.rollout_config.response_length)
        configured_max_len = getattr(self.rollout_config, "max_model_len", None)
        self.max_model_len: int = (
            int(configured_max_len)
            if configured_max_len is not None
            else (self.prompt_length + self.response_length)
        )
        self.turn_max_new_tokens: int = int(
            os.environ.get("SHOP_RL_TURN_MAX_NEW_TOKENS", "512")
        )
        self.max_steps: int = int(os.environ.get("SHOP_RL_MAX_STEPS", "35"))
        self.env_slots: int = int(os.environ.get("SHOP_RL_ENV_SLOTS", "8"))
        self.env_base_url: str = os.environ.get(
            "SHOP_ENV_BASE_URL", "http://127.0.0.1:5700"
        ).rstrip("/")
        self.log_dir: Optional[str] = os.environ.get("SHOP_RL_LOG_DIR") or None
        parser_name = getattr(self.rollout_config.multi_turn, "format", None) or "qwen3_coder"
        self.tool_parser = ToolParser.get_tool_parser(parser_name, self.tokenizer)
        stop_ids: set[int] = set()
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int) and eos_id >= 0:
            stop_ids.add(int(eos_id))
        elif isinstance(eos_id, (list, tuple, set)):
            for tid in eos_id:
                if isinstance(tid, int) and tid >= 0:
                    stop_ids.add(int(tid))
        for special_tok in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(special_tok)
            if isinstance(tid, int) and tid >= 0:
                stop_ids.add(int(tid))
        for tid in self.tool_parser.stop_token_ids or []:
            if isinstance(tid, int) and tid >= 0:
                stop_ids.add(int(tid))
        self._stop_token_ids: List[int] = sorted(stop_ids)
        self._tool_schemas: List[OpenAIFunctionToolSchema] = [
            OpenAIFunctionToolSchema.model_validate(schema) for schema in SHOP_TOOL_SCHEMAS
        ]

    @classmethod
    def _get_env_semaphore(cls, max_slots: int) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        sem = cls._loop_semaphores.get(loop_id)
        if sem is None:
            sem = asyncio.Semaphore(max(1, max_slots))
            cls._loop_semaphores[loop_id] = sem
        return sem

    def _make_request_id(self, priority: int = 0) -> str:
        if getattr(self.rollout_config, "full_determinism", False):
            return f"det-{priority}"
        return uuid4().hex

    @staticmethod
    def _build_assistant_message(
        content: str, func_calls: List[Any], step_idx: int
    ) -> Dict[str, Any]:
        """构造与 HF Qwen3.5 chat_template 兼容的 assistant 消息。"""
        msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        if not func_calls:
            return msg

        formatted_calls: List[Dict[str, Any]] = []
        for idx, fc in enumerate(func_calls):
            schema_obj, has_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                OpenAIFunctionParsedSchema(name=fc.name, arguments=fc.arguments)
            )
            func_dict = (
                {"name": fc.name, "arguments": fc.arguments}
                if has_error
                else schema_obj.model_dump()
            )
            formatted_calls.append(
                {
                    "id": fc.tool_call_id or f"call_{step_idx}_{idx}",
                    "type": "function",
                    "function": func_dict,
                }
            )
        msg["tool_calls"] = formatted_calls
        return msg

    @staticmethod
    def _extract_terminal_metrics(
        termination_reason: str,
        terminal_detail: Dict[str, Any],
        purchase: Dict[str, Any],
        task_id: int,
    ) -> Tuple[float, Dict[str, float | int | bool]]:
        """提取 strict 奖励与标量分量；购买但不可核验时抛出异常中止该组。"""
        purchased = bool(purchase) or termination_reason == "purchase"

        if purchased:
            reward_valid = bool(terminal_detail.get("reward_valid", False))
            strict_score = terminal_detail.get("paper_strict_score")
            if not reward_valid or strict_score is None:
                raise RuntimeError(
                    f"paper_strict 购买终局不可核验 (task_id={task_id}, detail={terminal_detail})"
                )
            r_type = float(terminal_detail.get("r_type") or 0.0)
            r_att = float(terminal_detail.get("r_att") or 0.0)
            r_option = float(terminal_detail.get("r_option") or 0.0)
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
            "task_id": int(task_id),
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

    @rollout_trace_op
    async def run(
        self, sampling_params: dict[str, Any], priority: int = 0, **kwargs
    ) -> AgentLoopOutput:
        started_at = time.monotonic()
        extra_info = kwargs.get("extra_info") or {}
        if "task_id" not in extra_info:
            raise KeyError(f"extra_info 缺少 task_id: {extra_info!r}")
        task_id = int(extra_info["task_id"])
        split = str(extra_info.get("split", "train"))
        uid = str(kwargs.get("uid", ""))
        session_id = kwargs.get("session_id")
        global_steps = kwargs.get("global_steps")

        shop_env = ShopEnv({"base_url": self.env_base_url, "if_persona": False})
        sem = self._get_env_semaphore(self.env_slots)

        metrics_dict: Dict[str, Any] = {}
        messages: List[Dict[str, Any]] = []
        runtime_token_ids: List[int] = []
        response_mask: List[int] = []
        track_logprobs = bool(sampling_params.get("logprobs", False))
        response_logprobs: Optional[List[float]] = [] if track_logprobs else None

        termination_reason = "unknown"
        invalid_reason: Optional[str] = None
        terminal_detail: Dict[str, Any] = {}
        purchase: Dict[str, Any] = {}
        action_trace: List[str] = []
        num_steps = 0
        assistant_turns = 0
        user_turns = 0
        request_id = self._make_request_id(priority)
        extra_fields: Dict[str, Any] = {}

        try:
            async with sem:
                with simple_timer("tool_calls", metrics_dict):
                    reset_res = await self.loop.run_in_executor(None, shop_env.reset, task_id)
                instruction = reset_res["instruction"]
                observation_state = reset_res.get("observation_state") or {}

                messages = [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _format_initial_user_message(instruction, observation_state),
                    },
                ]
                runtime_token_ids = await self.ct_build_initial_tokens(
                    messages, tools=SHOP_TOOL_SCHEMAS
                )

                for step_idx in range(self.max_steps):
                    remaining_resp = self.response_length - len(response_mask)
                    remaining_model = self.max_model_len - len(runtime_token_ids) - 1
                    turn_budget = min(self.turn_max_new_tokens, remaining_resp, remaining_model)
                    if step_idx > 0 and turn_budget < 16:
                        termination_reason = "context_limit"
                        break
                    if turn_budget <= 0:
                        raise ValueError(
                            f"首轮可用生成预算不足 (task_id={task_id}, "
                            f"prompt_tokens={len(runtime_token_ids)}, max_model_len={self.max_model_len})"
                        )

                    turn_sampling = dict(sampling_params)
                    turn_sampling["max_new_tokens"] = turn_budget
                    if self._stop_token_ids:
                        existing_stops = turn_sampling.get("stop_token_ids") or []
                        turn_sampling["stop_token_ids"] = list(
                            set(existing_stops + self._stop_token_ids)
                        )

                    with simple_timer("generate_sequences", metrics_dict):
                        output = await self.server_manager.generate(
                            request_id=request_id,
                            prompt_ids=runtime_token_ids,
                            sampling_params=turn_sampling,
                        )
                    assistant_turns += 1
                    if output.extra_fields:
                        if not extra_fields:
                            extra_fields.update(output.extra_fields)
                        else:
                            max_gs = output.extra_fields.get("max_global_steps", None)
                            if max_gs is not None:
                                extra_fields["max_global_steps"] = max_gs
                            for key in (
                                "spec_num_draft_tokens",
                                "spec_num_accepted_tokens",
                                "spec_num_verify_steps",
                            ):
                                if key in output.extra_fields and key in extra_fields:
                                    extra_fields[key] = int(extra_fields[key]) + int(
                                        output.extra_fields[key]
                                    )

                    if output.log_probs is not None and response_logprobs is None:
                        response_logprobs = [0.0] * len(response_mask)
                    elif output.log_probs is None and response_logprobs is not None and len(output.token_ids) > 0:
                        assistant_lps = [0.0] * len(output.token_ids)
                    else:
                        assistant_lps = output.log_probs

                    merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                        runtime_token_ids,
                        output.token_ids,
                        response_mask,
                        response_logprobs,
                        assistant_logprobs=assistant_lps,
                    )
                    runtime_token_ids = merge_result.token_ids

                    hit_turn_len_cap = len(output.token_ids) >= turn_budget
                    if len(response_mask) >= self.response_length:
                        termination_reason = "context_limit"
                        break

                    assistant_content, func_calls = await self.tool_parser.extract_tool_calls(
                        output.token_ids, tools=self._tool_schemas
                    )
                    assistant_msg = self._build_assistant_message(
                        assistant_content, func_calls, step_idx
                    )

                    try:
                        parsed_calls = parse_tool_calls(assistant_msg, observation_state)
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

                    with simple_timer("tool_calls", metrics_dict):
                        env_res = await self.loop.run_in_executor(None, shop_env.interact, action)

                    observation_state = env_res.get("observation_state") or {}
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

                    if step_idx + 1 >= self.max_steps:
                        termination_reason = "max_steps"
                        break

                    messages.append(assistant_msg)
                    previous_messages = list(messages)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(observation_state, ensure_ascii=False),
                        }
                    )
                    user_turns += 1

                    ctx_merge, ctx_mask, ctx_logprobs = await self.ct_merge_context_msg(
                        previous_messages,
                        messages,
                        runtime_token_ids,
                        response_mask,
                        response_logprobs,
                        tools=SHOP_TOOL_SCHEMAS,
                    )
                    # 若追加观察后超出总预算，则停留在本轮 assistant token 末尾结束轨迹，
                    # 防止 response_ids 被截断时停在只有 mask=0 的观察半截。
                    if (
                        len(ctx_mask) > self.response_length
                        or len(ctx_merge.token_ids) + 16 > self.max_model_len
                    ):
                        termination_reason = "context_limit"
                        break

                    runtime_token_ids = ctx_merge.token_ids
                    response_mask = ctx_mask
                    response_logprobs = ctx_logprobs
        finally:
            if shop_env.env_idx is not None:
                try:
                    await self.loop.run_in_executor(None, shop_env.release)
                except Exception as release_exc:
                    logger.warning(
                        "释放 ShopEnv 槽位失败 (task_id=%s, env_idx=%s): %s",
                        task_id,
                        shop_env.env_idx,
                        release_exc,
                    )

        response_len = len(response_mask)
        if response_len <= 0 or sum(response_mask) <= 0:
            raise RuntimeError(
                f"轨迹未产生可训练 assistant token (task_id={task_id}, termination={termination_reason})"
            )

        prompt_ids = runtime_token_ids[:-response_len]
        response_ids = runtime_token_ids[-response_len:]

        reward_score, reward_extras = self._extract_terminal_metrics(
            termination_reason, terminal_detail, purchase, task_id
        )
        reward_extras["num_steps"] = num_steps
        reward_extras["is_invalid_tool"] = int(termination_reason == "invalid_tool_call")
        reward_extras["is_context_limit"] = int(termination_reason == "context_limit")
        reward_extras["assistant_gen_tokens"] = int(sum(response_mask))

        _append_jsonl_record(
            self.log_dir,
            {
                "timestamp": time.time(),
                "elapsed_sec": round(time.monotonic() - started_at, 3),
                "global_step": global_steps,
                "split": split,
                "uid": uid,
                "session_id": session_id,
                "task_id": task_id,
                "termination_reason": termination_reason,
                "invalid_reason": invalid_reason,
                "num_steps": num_steps,
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(response_ids),
                "assistant_gen_tokens": int(sum(response_mask)),
                "reward_score": reward_score,
                **reward_extras,
                "action_trace": action_trace,
            },
        )

        agent_metrics = AgentLoopMetrics(
            generate_sequences=float(metrics_dict.get("generate_sequences", 0.0)),
            tool_calls=float(metrics_dict.get("tool_calls", 0.0)),
            compute_score=float(metrics_dict.get("compute_score", 0.0)),
        )
        extra_fields["reward_extra_info"] = reward_extras
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=reward_score,
            num_turns=user_turns + assistant_turns + 1,
            metrics=agent_metrics,
            extra_fields=extra_fields,
        )

"""Agent module: implements main logic for shopping agent.

两种运行模式：
- 旧评测模式（config 无 tools_mode）：模型输出 Action 字符串，直接交给环境；
- Teacher 采集模式（tools_mode: true）：模型通过原生 tool_calls 表达动作，
  经 tool_adapter 校验转换后执行；完整轨迹按步保存到批次目录。
"""

import argparse
import copy
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from tqdm import tqdm

from env import ShopEnv
from tool_adapter import SHOP_TOOL_SCHEMAS, ToolCallError, parse_tool_calls

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRY = 200
FAILED_CALL_MESSAGE = "failed call"

# Teacher 模式默认参数；来源：docs/v0 设计（shop-grpo-v0.md 第三节）。
TEACHER_TIMEOUT_SECONDS = 180.0
TEACHER_MAX_REQUEST_ATTEMPTS = 3
RETRYABLE_STATUS_CODES = {429, 500, 503}
FATAL_STATUS_CODES = {400, 401, 402, 422}


class TeacherFatalError(Exception):
    """鉴权、余额或请求格式等不可重试的 Teacher API 错误。"""


def _message_to_dict(message: Any) -> Dict[str, Any]:
    """把 SDK 消息对象转成干净 dict，保留 content、reasoning_content 和 tool_calls。"""
    data = message.model_dump() if hasattr(message, "model_dump") else dict(message)
    cleaned = {"role": data.get("role", "assistant")}
    for key in ("content", "reasoning_content", "tool_calls"):
        if data.get(key) is not None:
            cleaned[key] = data[key]
    return cleaned


def _completion_to_dict(completion: Any) -> Dict[str, Any]:
    """把完整 ChatCompletion 序列化，保留原始 message、finish_reason 与 usage。"""
    data = completion.model_dump() if hasattr(completion, "model_dump") else dict(completion)
    return data


class Agent:
    """Shopping agent class responsible for interacting with shopping environment."""

    def __init__(self, task_id: int, config: Dict[str, Any]) -> None:
        self.task_id = task_id
        self.config = config
        self.model_name = self.config["model_name"]
        self.tools_mode = bool(self.config.get("tools_mode", False))

        if self.tools_mode:
            api_key = self.config.get("model_key") or os.environ.get(
                self.config.get("api_key_env", "TEACHER_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "Teacher API key 未设置：请在 yaml 配置 model_key，"
                    "或设置环境变量 TEACHER_API_KEY"
                )
            self.model_key = api_key
            self.base_url = self.config["base_url"]
            self.max_tokens = self.config.get("max_tokens", 8192)
            self.reasoning_effort = self.config.get("reasoning_effort", "high")
            self.thinking = self.config.get("thinking", {"type": "enabled"})
            self.timeout_seconds = float(
                self.config.get("timeout_seconds", TEACHER_TIMEOUT_SECONDS)
            )
            self.max_request_attempts = int(
                self.config.get("max_request_attempts", TEACHER_MAX_REQUEST_ATTEMPTS)
            )
            if self.max_request_attempts < 1:
                raise ValueError(
                    f"max_request_attempts 必须 >= 1，收到 {self.max_request_attempts}"
                )
            self.tool_choice = self.config.get("tool_choice", "auto")
            # 每个 worker（任务）创建一次客户端，各轮复用；重试由采集层控制。
            self.client = OpenAI(
                api_key=self.model_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=0,
            )
        else:
            self.model_key = self.config.get("model_key")
            self.base_url = self.config.get("base_url")
            if self.model_key is None or self.base_url is None:
                raise ValueError("model_key and base_url are required")

        self.system_prompt = config["system_prompt"]

        self.shop_env: Optional[ShopEnv] = None
        self.task_complete = False
        self.conversation_log: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, str]] = []
        self.user_persona: Optional[Dict[str, Any]] = None
        self.env_idx: Optional[int] = None

        # Teacher 模式的轨迹状态
        self.batch_name: str = str(self.config.get("batch_name", self.model_name))
        self.max_steps = int(self.config.get("max_steps", 35))
        self.steps: List[Dict[str, Any]] = []
        self.instruction: Optional[str] = None
        self.initial_observation_state: Optional[Dict[str, Any]] = None
        self.usage_records: List[Dict[str, Any]] = []
        self.failure_reason: Optional[str] = None
        self.terminal: Dict[str, Any] = {}

    def set_shop_env(self, shop_env: ShopEnv) -> None:
        self.shop_env = shop_env

    # ------------------------------------------------------------------
    # 旧评测模式（Action 字符串）保留不动；Teacher 模式走下方新方法。
    # ------------------------------------------------------------------

    def reset(self) -> str:
        if self.shop_env is None:
            raise AttributeError("self.shop_env not initialized")

        env_result = self.shop_env.reset(self.task_id)
        self.env_idx = self.shop_env.env_idx

        if self.tools_mode:
            return self._reset_teacher(env_result)

        instruction = env_result["instruction"] + "\n\n搜索功能是否可用: True\n\n可点击的按钮: []"
        self.task_complete = False
        self.conversation_log = []

        if self.shop_env.if_persona and "user_persona" in env_result:
            self.user_persona = env_result["user_persona"]
            persona_str = json.dumps(self.user_persona, ensure_ascii=False)
            persona_content = f"\n用户的个人文档是：{persona_str}"
            self.messages = [
                {"role": "system", "content": self.system_prompt + persona_content}
            ]
        else:
            self.messages = [{"role": "system", "content": self.system_prompt}]

        return instruction

    def _reset_teacher(self, env_result: Dict[str, Any]) -> str:
        """记录初始状态并构造首条 user 消息：需求 + 公开观察。"""
        self.task_complete = False
        self.steps = []
        self.usage_records = []
        self.failure_reason = None
        self.terminal = {}
        self.instruction = env_result["instruction"]
        self.initial_observation_state = env_result.get("observation_state", {})

        user_content = (
            f"购物任务需求：\n{self.instruction}\n\n"
            f"当前观察（公开状态）：\n"
            f"{json.dumps(self.initial_observation_state, ensure_ascii=False)}"
        )
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        self._persist_teacher()
        return self.instruction

    def act(self, instruction: str) -> Tuple[bool, str]:
        if self.tools_mode:
            return self._act_teacher()
        return self._act_legacy(instruction)

    def _act_legacy(self, instruction: str) -> Tuple[bool, str]:
        if self.shop_env is None:
            raise AttributeError(
                "Agent not correctly connected to ShopEnv, please check the configuration."
            )

        self.messages.append({"role": "user", "content": instruction})

        llm_response = self.get_response_idealab(self.messages)
        llm_response = self._clean_response(llm_response)

        if llm_response == FAILED_CALL_MESSAGE:
            raise RuntimeError("LLM call failed")

        self.messages.append({"role": "assistant", "content": llm_response})
        action = llm_response

        env_response = self.shop_env.interact(action)
        if "error" in env_response:
            print(action)
            raise ValueError(f"环境报错: {env_response['error']}")

        observation = env_response.get("instruction", "")

        if env_response.get("done", False) or env_response.get("over", False):
            reward = env_response.get("reward", 0)
            reward_detail = env_response.get("reward_detail", {})
            goal = env_response.get("goal", {})
            purchase = env_response.get("purchase", {})

            self.save_to_json(reward, reward_detail, goal, purchase)
            self.task_complete = True

        return self.task_complete, observation

    def _act_teacher(self) -> Tuple[bool, str]:
        """Teacher 模式单步：请求模型 -> 校验转换 -> 执行 -> 回传公开观察。

        每步完成后立即写盘；发生任何异常时轨迹已保留。
        """
        if self.shop_env is None:
            raise AttributeError("Agent not correctly connected to ShopEnv")

        observation_before = self._current_observation_state()
        step_record: Dict[str, Any] = {
            "step": len(self.steps),
            "observation_before": observation_before,
        }
        self.steps.append(step_record)

        try:
            completion, attempts = self.call_teacher(self.messages)
        except Exception as exc:
            # 失败轮次同样落入轨迹：重试记录来自异常携带的 attempts；
            # 每个失败请求都没有 usage，逐个计为未知，不漏计也不记成 0。
            # attempts 为空列表（无属性或未发请求）时如实留空，不伪造记录。
            failed_attempts = getattr(exc, "attempts", None)
            if failed_attempts is None:
                failed_attempts = []
            step_record["request_attempts"] = failed_attempts
            self.usage_records.extend({"unknown": True} for _ in failed_attempts)
            self.failure_reason = (
                f"teacher_fatal: {exc}"
                if isinstance(exc, TeacherFatalError)
                else f"teacher_failed: {exc}"
            )
            self._persist_teacher()
            raise

        step_record["request_attempts"] = attempts
        raw = _completion_to_dict(completion)
        step_record["assistant_raw"] = raw
        usage = raw.get("usage")
        step_record["usage"] = usage
        # 成功响应缺 usage 字段也计未知；前面重试失败的请求同样未知。
        unknown_before = sum(1 for a in attempts if not a.get("ok"))
        self.usage_records.extend({"unknown": True} for _ in range(unknown_before))
        if usage is not None:
            self.usage_records.append(usage)
        else:
            self.usage_records.append({"unknown": True})

        choice = (raw.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        assistant_msg = choice.get("message") or {}
        step_record["finish_reason"] = finish_reason

        if finish_reason == "length":
            # 截断响应中的工具调用可能不完整，不执行，标记失败。
            self.failure_reason = "finish_length_truncated"
            self._persist_teacher()
            return self._finish_teacher()

        assistant_dict = {
            "role": "assistant",
            "content": assistant_msg.get("content"),
        }
        if assistant_msg.get("reasoning_content") is not None:
            assistant_dict["reasoning_content"] = assistant_msg["reasoning_content"]
        if assistant_msg.get("tool_calls"):
            assistant_dict["tool_calls"] = assistant_msg["tool_calls"]
        # assistant 消息（含思考与 tool_calls）按协议完整回传。
        self.messages.append(assistant_dict)

        try:
            calls = parse_tool_calls(assistant_msg, observation_before)
        except ToolCallError as exc:
            self.failure_reason = f"invalid_tool_call: {exc}"
            self._persist_teacher()
            return self._finish_teacher()

        if not calls:
            self.failure_reason = "no_tool_call"
            self._persist_teacher()
            return self._finish_teacher()

        tool_call_id, action, arguments, tool_name = calls[0]
        step_record.update(
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "action": action,
            }
        )

        env_result = self.shop_env.interact(action)
        # 完整环境返回（含 goal、reward 等内部字段）只进原始审计记录。
        step_record["env_result"] = env_result

        observation_state = env_result.get("observation_state", {})
        step_record["observation_after"] = observation_state
        # Teacher 上下文只回传公开 observation_state，隐藏答案不进入对话。
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(observation_state, ensure_ascii=False),
            }
        )

        done = bool(env_result.get("done", False))
        over = bool(env_result.get("over", False))
        step_record["done"] = done
        step_record["over"] = over

        if done or over:
            self.terminal = {
                "done": done,
                "over": over,
                "reward": env_result.get("reward"),
                "reward_detail": env_result.get("reward_detail", {}),
                "purchase": env_result.get("purchase", {}),
                "goal": env_result.get("goal", {}),
                "termination_reason": env_result.get("termination_reason"),
                "reward_valid": env_result.get("reward_valid"),
            }
            self.task_complete = True

        self._persist_teacher()
        return self.task_complete, env_result.get("instruction", "")

    def _finish_teacher(self) -> Tuple[bool, str]:
        self.task_complete = True
        return True, ""

    def _current_observation_state(self) -> Dict[str, Any]:
        if self.steps:
            return self.steps[-1].get("observation_after") or self.initial_observation_state or {}
        return self.initial_observation_state or {}

    def call_teacher(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """调用 Teacher，返回完整 completion 与每次尝试的记录。

        重试策略：总共最多 3 次尝试；429/500/503 与网络错误在 1s、2s 后重试
        （遵从 Retry-After）；400/401/402/422 直接抛 TeacherFatalError。
        """
        attempts: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None

        def _fail(exc: Exception) -> Exception:
            # 所有失败路径都带上完整 attempts，调用方把重试记录落入轨迹。
            fatal = TeacherFatalError(str(exc))
            fatal.attempts = list(attempts)
            return fatal

        for attempt in range(1, self.max_request_attempts + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=SHOP_TOOL_SCHEMAS,
                    tool_choice=self.tool_choice,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                    stream=False,
                    extra_body={"thinking": self.thinking},
                )
                attempts.append({"attempt": attempt, "ok": True})
                return completion, attempts
            except Exception as exc:  # noqa: BLE001 - 按状态码分类重试
                status = getattr(exc, "status_code", None)
                attempts.append(
                    {
                        "attempt": attempt,
                        "ok": False,
                        "status_code": status,
                        "error": str(exc),
                    }
                )
                last_error = exc
                if status in FATAL_STATUS_CODES:
                    raise _fail(exc) from exc
                if attempt == self.max_request_attempts:
                    break
                if status in RETRYABLE_STATUS_CODES or status is None:
                    retry_after = getattr(exc, "response", None)
                    delay = None
                    if retry_after is not None:
                        header = getattr(retry_after, "headers", {}).get("Retry-After")
                        if header:
                            try:
                                delay = float(header)
                            except (TypeError, ValueError):
                                delay = None
                    time.sleep(delay if delay is not None else float(attempt))
                else:
                    raise _fail(exc) from exc

        final_error = RuntimeError(
            f"Teacher 请求 {self.max_request_attempts} 次均失败: {last_error}"
        )
        final_error.attempts = list(attempts)
        raise final_error

    def _clean_response(self, response: str) -> str:
        if "</think>" in response:
            response = response.split("</think>")[-1].strip("\n")
        return response

    def get_response_idealab(
        self, messages: List[Dict[str, str]], max_try: int = DEFAULT_MAX_RETRY
    ) -> str:
        if self.model_key is None:
            raise ValueError("model_key not set, cannot call API")
        if self.base_url is None:
            raise ValueError("base_url not set, cannot call API")

        client = OpenAI(
            api_key=self.model_key,
            base_url=self.base_url,
        )

        for attempt in range(max_try):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=DEFAULT_TEMPERATURE,
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"LLM call failed (attempt {attempt + 1}/{max_try}): {e}")
                if attempt == max_try - 1:
                    traceback.print_exc()
                continue

        return FAILED_CALL_MESSAGE

    def build_trajectory(self, release_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """组装单题完整轨迹文件内容。"""
        record: Dict[str, Any] = {
            "task_id": self.task_id,
            "batch_name": self.batch_name,
            "model": {
                "model_name": self.model_name,
                "base_url": self.base_url,
                "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort,
                "thinking": self.thinking,
                "tool_choice": self.tool_choice,
            },
            "instruction": self.instruction,
            "initial_observation_state": self.initial_observation_state,
            "tools": SHOP_TOOL_SCHEMAS,
            "status": "incomplete",
            "failure_reason": self.failure_reason,
            "conversation": self.messages,
            "steps": self.steps,
            "terminal": self.terminal,
            "release_result": release_result,
            "token_usage_summary": self._sum_usage(),
        }
        return record

    def _sum_usage(self) -> Dict[str, Any]:
        known = [u for u in self.usage_records if not u.get("unknown")]
        unknown = len(self.usage_records) - len(known)
        return {
            "requests": len(self.usage_records),
            "requests_with_unknown_usage": unknown,
            "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in known),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in known),
            "reasoning_tokens": sum(
                (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                for u in known
            ),
        }

    def _teacher_output_path(self) -> str:
        # 按采集批次保存：改提示词/工具/参数后换批次名，避免与旧数据混用。
        return os.path.join(self.config["output_path"], self.batch_name)

    def _persist_teacher(self) -> None:
        os.makedirs(self._teacher_output_path(), exist_ok=True)
        file_path = os.path.join(self._teacher_output_path(), f"{self.task_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.build_trajectory(), f, ensure_ascii=False, indent=2)

    def save_to_json(
        self,
        reward: float,
        reward_detail: Dict[str, Any],
        goal: Dict[str, Any],
        purchase: Dict[str, Any],
    ) -> None:
        log_data = {
            "task_id": self.task_id,
            "reward": reward,
            "reward_detail": reward_detail,
            "goal": goal,
            "purchase": purchase,
            "conversation": self.messages,
        }

        file_path = os.path.join(
            self.config["output_path"], self.config["model_name"]
        )
        os.makedirs(file_path, exist_ok=True)

        filename = os.path.join(file_path, f"{self.task_id}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)

        print(f"[LOG] Saved to file: {filename}")


def get_finished_task(out_path: str) -> List[int]:
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    json_files = []
    for filename in os.listdir(out_path):
        if filename.endswith(".json"):
            task_id_str = filename.rsplit(".", 1)[0]
            try:
                json_files.append(int(task_id_str))
            except ValueError:
                print(
                    f"Warning: Unable to parse task ID '{task_id_str}', skipping file"
                )

    return sorted(json_files)


def get_finished_teacher_tasks(batch_path: str) -> Dict[int, str]:
    """读取批次目录中每个任务的状态（不能仅凭文件存在判断完成）。"""
    finished: Dict[int, str] = {}
    if not os.path.isdir(batch_path):
        return finished
    for filename in os.listdir(batch_path):
        if not filename.endswith(".json"):
            continue
        try:
            task_id = int(filename.rsplit(".", 1)[0])
        except ValueError:
            continue
        try:
            with open(os.path.join(batch_path, filename), encoding="utf-8") as f:
                record = json.load(f)
            status = record.get("status", "incomplete")
        except (OSError, json.JSONDecodeError):
            status = "incomplete"
        finished[task_id] = status
    return finished


def run_task(task_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """运行单题并返回轨迹结果；异常也返回已保存的结果。

    只有取得槽位后才释放环境；释放结果写入轨迹。
    """
    agent: Optional[Agent] = None
    shop_env: Optional[ShopEnv] = None
    release_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    try:
        shop_env = ShopEnv(config["env_config"])
        agent = Agent(task_id=task_id, config=config["agent_config"])
        agent.set_shop_env(shop_env)
        observation = agent.reset()

        steps = 0
        while True:
            done, observation = agent.act(observation)
            steps += 1
            if done:
                break
            if agent.tools_mode and steps >= agent.max_steps:
                agent.failure_reason = "max_steps_exceeded"
                agent._persist_teacher()
                break
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        if agent is not None and agent.failure_reason is None:
            agent.failure_reason = f"exception: {error}"
        print(f"Error (task {task_id}): {error}")
        traceback.print_exc()
    finally:
        if agent is not None and agent.shop_env is not None and agent.env_idx is not None:
            try:
                release_result = agent.shop_env.release()
            except Exception as release_exc:
                release_result = {"error": f"{type(release_exc).__name__}: {release_exc}"}

    if agent is None:
        return {"task_id": task_id, "status": "failed", "failure_reason": f"init: {error}"}

    result = agent.build_trajectory(release_result)
    if agent.failure_reason is None and agent.terminal.get("done"):
        result["status"] = "finished"
    elif agent.failure_reason is None:
        result["status"] = "incomplete"
        result["failure_reason"] = "released_before_done"
    else:
        result["status"] = "failed"
    if error is not None:
        result["exception"] = error
    # 终态写盘一次，保证释放结果和最终状态落盘。
    if agent.tools_mode:
        record = copy.deepcopy(result)
        os.makedirs(agent._teacher_output_path(), exist_ok=True)
        with open(
            os.path.join(agent._teacher_output_path(), f"{task_id}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    return result


def run_tasks_multithreaded(
    todo_tasks: List[int], config: Dict[str, Any], max_workers: int = 4
) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_task, task_id, config): task_id
            for task_id in todo_tasks
        }

        for future in tqdm(
            as_completed(future_to_task),
            total=len(todo_tasks),
            desc="执行任务",
        ):
            task_id = future_to_task[future]
            try:
                future.result()
                print(f"**Task {task_id} completed**")
            except Exception as e:
                print(f"**Task {task_id} failed**: {e}")


def run_tasks_singlethreaded(
    todo_tasks: List[int], config: Dict[str, Any]
) -> None:
    for task_id in tqdm(todo_tasks, desc="执行任务"):
        print(f"**Begin task {task_id}**")
        run_task(task_id, config)


def main() -> None:
    """旧评测入口：按 yaml 的 task_nums 顺序执行（注意 task_id 从 0 开始，
    会进入官方评测任务；Teacher 采集请使用 collect_teacher.py）。"""
    parser = argparse.ArgumentParser(
        description="Batch process tasks from YAML config files."
    )
    parser.add_argument(
        "--yaml_name", required=True, help="YAML config file name"
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Whether to use multithreading to execute",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of worker threads in multithread mode",
    )
    args = parser.parse_args()

    config_file = args.yaml_name
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found '{config_file}'")
        return
    except yaml.YAMLError as e:
        print(f"Error: YAML parsing failed: {e}")
        return

    agent_config = config.get("agent_config")
    if agent_config is None:
        print("Error: 'agent_config' missing in config file")
        return

    output_path = os.path.join(
        agent_config["output_path"], agent_config["model_name"]
    )
    finished_tasks = get_finished_task(output_path)
    all_tasks = list(range(agent_config.get("task_nums", 0)))
    todo_tasks = [i for i in all_tasks if i not in finished_tasks]

    print(
        f"Total tasks: {len(all_tasks)}, "
        f"Completed: {len(finished_tasks)}, "
        f"Pending: {len(todo_tasks)}"
    )

    if args.multithread:
        print(f"Using multithread mode, max workers: {args.max_workers}")
        run_tasks_multithreaded(todo_tasks, config, args.max_workers)
    else:
        print("Using single-thread mode")
        run_tasks_singlethreaded(todo_tasks, config)


if __name__ == "__main__":
    main()

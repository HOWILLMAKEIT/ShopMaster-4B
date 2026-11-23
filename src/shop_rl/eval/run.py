"""基于 verl 的统一评测入口（`trainer.val_only=true`）。

在严格 v2 协议（`paper_strict`、可配置上下文、贪婪 `val_kwargs.temperature=0.0`）下复用
`verl.trainer.main_ppo` 评测纯 HF 模型或 RL LoRA checkpoint，并将轨迹与汇总写入
指定输出目录。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from shop_rl.eval.summarize import summarize_run

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/rl/vanilla_grpo_a800_80g.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data/rl"
DEFAULT_AGENT_LOOPS_CONFIG = REPO_ROOT / "configs/rl/agent_loops.yaml"
DEFAULT_VERL_PYTHON = pathlib.Path("/opt/verl/.venv/bin/python")
LOCAL_VENV_RL_PYTHON = REPO_ROOT / "venv-rl/bin/python"

SPLIT_TO_PARQUET: Dict[str, str] = {
    "smoke20": "val_smoke20.parquet",
    "dev100": "val_dev100.parquet",
    "dev474": "val_dev474.parquet",
    "holdout915": "eval_holdout915.parquet",
}


def _resolve_default_verl_python() -> str:
    if DEFAULT_VERL_PYTHON.is_file():
        return str(DEFAULT_VERL_PYTHON)
    if LOCAL_VENV_RL_PYTHON.is_file():
        return str(LOCAL_VENV_RL_PYTHON)
    return sys.executable


def _resolve_train_parquet(data_dir: pathlib.Path, explicit: Optional[pathlib.Path]) -> pathlib.Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    smoke = data_dir / "smoke_train8.parquet"
    if smoke.is_file():
        return smoke.resolve()
    return (data_dir / "train.parquet").resolve()


def _validate_checkpoint_path(checkpoint: pathlib.Path, *, dry_run: bool) -> pathlib.Path:
    resolved = checkpoint.expanduser().resolve()
    name = resolved.name
    if "global_step_" not in name:
        raise ValueError(
            f"--checkpoint 必须指向 global_step_<N> 目录，收到: {checkpoint}"
        )
    step_suffix = name.split("global_step_")[-1]
    if not step_suffix.isdigit():
        raise ValueError(
            f"--checkpoint 目录名末尾必须为整数步数（global_step_<N>），收到: {checkpoint}"
        )
    if not dry_run and not resolved.is_dir():
        raise FileNotFoundError(f"checkpoint 目录不存在: {resolved}")
    return resolved


def build_verl_eval_command(
    *,
    verl_python: str,
    config: pathlib.Path,
    model_path: str,
    val_parquet: pathlib.Path,
    train_parquet: pathlib.Path,
    output_dir: pathlib.Path,
    label: str,
    agent_loops_config: pathlib.Path,
    checkpoint: Optional[pathlib.Path] = None,
    keep_lora: bool = False,
    enable_thinking: bool = False,
    max_model_len: Optional[int] = None,
    overrides: Sequence[str] = (),
) -> List[str]:
    """构造 `verl.trainer.main_ppo` 的完整命令行参数列表。"""
    config_abs = config.expanduser().resolve()
    out_abs = output_dir.expanduser().resolve()

    if max_model_len is not None and int(max_model_len) <= 2048:
        raise ValueError(f"--max-model-len 必须大于 2048（prompt 长度），收到: {max_model_len}")

    cmd: List[str] = [
        verl_python,
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        str(config_abs.parent),
        "--config-name",
        config_abs.stem,
        "trainer.val_before_train=true",
        "trainer.val_only=true",
        "trainer.logger=[console,file]",
        f"trainer.experiment_name={label}",
        f"trainer.default_local_dir={out_abs / 'checkpoints'}",
        f"trainer.validation_data_dir={out_abs / 'validation'}",
        f"data.val_files={val_parquet}",
        f"data.train_files={train_parquet}",
        f"data.apply_chat_template_kwargs.enable_thinking={'true' if enable_thinking else 'false'}",
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_loops_config}",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.0",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=false",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        # val_only 不执行训练与 KL 计算；关闭 use_kl_loss 避免 lora_rank=0 时额外初始化 ref 权重副本。
        "actor_rollout_ref.actor.use_kl_loss=false",
    ]

    if max_model_len is not None:
        total_len = int(max_model_len)
        cmd.extend(
            [
                f"actor_rollout_ref.rollout.max_model_len={total_len}",
                f"actor_rollout_ref.rollout.engine_kwargs.sglang.context_length={total_len}",
                f"data.max_response_length={total_len - 2048}",
            ]
        )

    if checkpoint is not None:
        ckpt_abs = checkpoint.expanduser().resolve()
        cmd.extend(
            [
                "trainer.resume_mode=resume_path",
                f"trainer.resume_from_path={ckpt_abs}",
                "actor_rollout_ref.actor.checkpoint.load_contents=[model]",
            ]
        )
    else:
        cmd.append("trainer.resume_mode=disable")
        if not keep_lora:
            cmd.extend(
                [
                    "actor_rollout_ref.model.lora_rank=0",
                    "actor_rollout_ref.model.lora.merge=false",
                ]
            )

    cmd.extend(overrides)
    return cmd


def build_eval_plan(
    *,
    output_dir: pathlib.Path,
    model_path: Optional[str] = None,
    checkpoint: Optional[pathlib.Path] = None,
    keep_lora: bool = False,
    enable_thinking: bool = False,
    max_model_len: Optional[int] = None,
    config: pathlib.Path = DEFAULT_CONFIG,
    split: str = "dev474",
    data_dir: pathlib.Path = DEFAULT_DATA_DIR,
    val_parquet: Optional[pathlib.Path] = None,
    train_parquet: Optional[pathlib.Path] = None,
    label: Optional[str] = None,
    env_url: Optional[str] = None,
    env_slots: Optional[int] = None,
    max_steps: Optional[int] = None,
    turn_max_new_tokens: Optional[int] = None,
    verl_python: Optional[str] = None,
    agent_loops_config: pathlib.Path = DEFAULT_AGENT_LOOPS_CONFIG,
    overrides: Sequence[str] = (),
    dry_run: bool = False,
    base_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """解析路径、环境变量与 Hydra 覆盖参数，返回纯数据结构的评测执行计划。"""
    env_source = os.environ if base_env is None else base_env

    resolved_model_raw = model_path or env_source.get("SHOP_RL_SFT_MODEL")
    if not resolved_model_raw:
        raise ValueError("必须通过 --model-path 或环境变量 SHOP_RL_SFT_MODEL 指定模型路径")
    resolved_model_path = str(pathlib.Path(resolved_model_raw).expanduser().resolve())

    if split not in SPLIT_TO_PARQUET:
        raise ValueError(f"不支持的 split: {split!r}，可选: {sorted(SPLIT_TO_PARQUET)}")

    config_abs = config.expanduser().resolve()
    data_dir_abs = data_dir.expanduser().resolve()
    output_abs = output_dir.expanduser().resolve()
    rollouts_dir = output_abs / "rollouts"
    validation_dir = output_abs / "validation"
    checkpoints_dir = output_abs / "checkpoints"
    agent_loops_abs = agent_loops_config.expanduser().resolve()

    val_parquet_abs = (
        val_parquet.expanduser().resolve()
        if val_parquet is not None
        else (data_dir_abs / SPLIT_TO_PARQUET[split]).resolve()
    )
    train_parquet_abs = _resolve_train_parquet(data_dir_abs, train_parquet)

    ckpt_abs: Optional[pathlib.Path] = None
    if checkpoint is not None:
        ckpt_abs = _validate_checkpoint_path(checkpoint, dry_run=dry_run)

    if not dry_run:
        if not config_abs.is_file():
            raise FileNotFoundError(f"verl 配置文件不存在: {config_abs}")
        if not agent_loops_abs.is_file():
            raise FileNotFoundError(f"agent_loops 配置不存在: {agent_loops_abs}")
        if not pathlib.Path(resolved_model_path).exists():
            raise FileNotFoundError(f"模型路径不存在: {resolved_model_path}")
        if not val_parquet_abs.is_file():
            raise FileNotFoundError(f"评测 Parquet 不存在: {val_parquet_abs}")
        if not train_parquet_abs.is_file():
            raise FileNotFoundError(f"占位训练 Parquet 不存在: {train_parquet_abs}")

    resolved_label = label or output_abs.name
    resolved_env_url = (
        env_url or env_source.get("SHOP_ENV_BASE_URL") or "http://127.0.0.1:5705"
    ).rstrip("/")
    resolved_env_slots = int(
        env_slots if env_slots is not None else (env_source.get("SHOP_RL_ENV_SLOTS") or "32")
    )
    resolved_max_steps = int(
        max_steps if max_steps is not None else (env_source.get("SHOP_RL_MAX_STEPS") or "35")
    )
    default_turn_tokens = "8192" if enable_thinking else "512"
    resolved_turn_tokens = int(
        turn_max_new_tokens
        if turn_max_new_tokens is not None
        else (env_source.get("SHOP_RL_TURN_MAX_NEW_TOKENS") or default_turn_tokens)
    )
    resolved_verl_python = verl_python or _resolve_default_verl_python()

    src_dir = str(REPO_ROOT / "src")
    existing_pypath = env_source.get("PYTHONPATH", "")
    pypath_parts = [p for p in existing_pypath.split(os.pathsep) if p]
    if not pypath_parts or pypath_parts[0] != src_dir:
        resolved_pypath = os.pathsep.join([src_dir, *pypath_parts])
    else:
        resolved_pypath = existing_pypath

    eval_env: Dict[str, str] = {
        "SHOP_RL_LOG_DIR": str(rollouts_dir),
        "SHOP_ENV_BASE_URL": resolved_env_url,
        "SHOP_RL_ENV_SLOTS": str(resolved_env_slots),
        "SHOP_RL_MAX_STEPS": str(resolved_max_steps),
        "SHOP_RL_TURN_MAX_NEW_TOKENS": str(resolved_turn_tokens),
        "VERL_FILE_LOGGER_PATH": str(output_abs / "verl_metrics.jsonl"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": resolved_pypath,
    }

    command = build_verl_eval_command(
        verl_python=resolved_verl_python,
        config=config_abs,
        model_path=resolved_model_path,
        val_parquet=val_parquet_abs,
        train_parquet=train_parquet_abs,
        output_dir=output_abs,
        label=resolved_label,
        agent_loops_config=agent_loops_abs,
        checkpoint=ckpt_abs,
        keep_lora=keep_lora,
        enable_thinking=enable_thinking,
        max_model_len=max_model_len,
        overrides=overrides,
    )

    return {
        "label": resolved_label,
        "mode": "checkpoint" if ckpt_abs is not None else "hf_model",
        "split": split,
        "config": str(config_abs),
        "agent_loops_config": str(agent_loops_abs),
        "model_path": resolved_model_path,
        "checkpoint": str(ckpt_abs) if ckpt_abs is not None else None,
        "keep_lora": bool(keep_lora),
        "enable_thinking": bool(enable_thinking),
        "max_model_len": int(max_model_len) if max_model_len is not None else None,
        "val_parquet": str(val_parquet_abs),
        "train_parquet": str(train_parquet_abs),
        "output_dir": str(output_abs),
        "rollouts_dir": str(rollouts_dir),
        "validation_dir": str(validation_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "verl_python": resolved_verl_python,
        "command": command,
        "env": eval_env,
    }


def _write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def execute_eval_plan(
    plan: Mapping[str, Any],
    *,
    base_env: Optional[Mapping[str, str]] = None,
) -> int:
    """创建输出目录、持久化 eval_meta.json、执行 verl 并在成功后写出 eval_summary.json。"""
    output_dir = pathlib.Path(plan["output_dir"])
    for subdir in (plan["rollouts_dir"], plan["validation_dir"], plan["checkpoints_dir"]):
        pathlib.Path(subdir).mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "eval_meta.json"
    started_mono = time.monotonic()
    meta: Dict[str, Any] = {
        **plan,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "elapsed_sec": None,
        "returncode": None,
    }
    _write_json(meta_path, meta)

    proc_env = dict(os.environ if base_env is None else base_env)
    proc_env.update(plan["env"])

    returncode = 1
    try:
        proc = subprocess.run(
            list(plan["command"]),
            env=proc_env,
            cwd=str(REPO_ROOT),
            check=False,
        )
        returncode = int(proc.returncode)
    except KeyboardInterrupt:
        returncode = 130
        meta["status"] = "interrupted"
        raise
    finally:
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta["elapsed_sec"] = round(time.monotonic() - started_mono, 3)
        meta["returncode"] = returncode
        if meta["status"] == "running":
            meta["status"] = "complete" if returncode == 0 else "failed"
        _write_json(meta_path, meta)

    if returncode == 0:
        val_parquet = pathlib.Path(plan["val_parquet"])
        summary = summarize_run(
            output_dir,
            label=str(plan["label"]),
            expected_tasks_path=val_parquet if val_parquet.is_file() else None,
        )
        _write_json(output_dir / "eval_summary.json", summary)

    return returncode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 verl (trainer.val_only=true) 按统一 v2 协议评测 HF 模型或 RL LoRA checkpoint。"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG,
        help="verl 基础配置 YAML（默认 configs/rl/vanilla_grpo_a800_80g.yaml）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="HF 基座/SFT 模型目录（未传时回退至环境变量 SHOP_RL_SFT_MODEL）",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="可选：RL LoRA checkpoint 目录（.../global_step_<N>）",
    )
    parser.add_argument(
        "--keep-lora",
        action="store_true",
        help="无 --checkpoint 时保留配置中的 LoRA 设置（默认关闭 lora_rank=0）",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="开启聊天模板思考模式（默认关闭；未指定单轮上限时自动提升至 8192）",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="可选：统一覆盖模型最大上下文长度（同步更新 max_model_len、sglang context_length 与 max_response_length）",
    )
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_TO_PARQUET.keys()),
        default="dev474",
        help="预设评测切分（默认 dev474）",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=DEFAULT_DATA_DIR,
        help="Parquet 数据目录（默认 data/rl）",
    )
    parser.add_argument(
        "--val-parquet",
        type=pathlib.Path,
        default=None,
        help="可选：直接覆盖 --split 对应的评测 Parquet 路径",
    )
    parser.add_argument(
        "--train-parquet",
        type=pathlib.Path,
        default=None,
        help="可选：verl DataLoader 初始化所需占位训练 Parquet（默认 smoke_train8.parquet 否则 train.parquet）",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="评测输出目录（自动创建 rollouts/ 与 validation/ 子目录）",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="评测标签与 experiment_name（默认使用 output_dir 目录名）",
    )
    parser.add_argument(
        "--env-url",
        type=str,
        default=None,
        help="ShopSimulator HTTP 服务地址（默认 SHOP_ENV_BASE_URL 或 http://127.0.0.1:5705）",
    )
    parser.add_argument(
        "--env-slots",
        type=int,
        default=None,
        help="环境并发槽位数（默认 SHOP_RL_ENV_SLOTS 或 32）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="单题最大交互步数（默认 SHOP_RL_MAX_STEPS 或 35）",
    )
    parser.add_argument(
        "--turn-max-new-tokens",
        type=int,
        default=None,
        help="单轮最大生成 token 数（默认 SHOP_RL_TURN_MAX_NEW_TOKENS，关思考 512 / 开思考 8192）",
    )
    parser.add_argument(
        "--verl-python",
        type=str,
        default=None,
        help="运行 verl 的 Python 解释器（默认 /opt/verl/.venv/bin/python 或 venv-rl/bin/python，否则 sys.executable）",
    )
    parser.add_argument(
        "--agent-loops-config",
        type=pathlib.Path,
        default=DEFAULT_AGENT_LOOPS_CONFIG,
        help="verl agent_loops 配置文件（默认 configs/rl/agent_loops.yaml）",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="可重复：追加传递给 verl 的额外 Hydra 覆盖参数",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印解析后的 JSON 命令与环境变量并退出（不创建目录，不需要 GPU）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        plan = build_eval_plan(
            output_dir=args.output_dir,
            model_path=args.model_path,
            checkpoint=args.checkpoint,
            keep_lora=args.keep_lora,
            enable_thinking=args.enable_thinking,
            max_model_len=args.max_model_len,
            config=args.config,
            split=args.split,
            data_dir=args.data_dir,
            val_parquet=args.val_parquet,
            train_parquet=args.train_parquet,
            label=args.label,
            env_url=args.env_url,
            env_slots=args.env_slots,
            max_steps=args.max_steps,
            turn_max_new_tokens=args.turn_max_new_tokens,
            verl_python=args.verl_python,
            agent_loops_config=args.agent_loops_config,
            overrides=args.override,
            dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    return execute_eval_plan(plan)


if __name__ == "__main__":
    raise SystemExit(main())

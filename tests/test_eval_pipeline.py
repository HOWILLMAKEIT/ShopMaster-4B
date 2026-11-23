"""统一 verl 评测管线单元测试：覆盖 prepare_tasks、eval.run、eval.summarize 与 CLI 路由。"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from shop_rl.eval.run import build_eval_plan, build_verl_eval_command
from shop_rl.eval.summarize import (
    build_comparison_report,
    compare_runs,
    load_task_ids,
    mcnemar_exact_p,
    summarize_records,
    wilson_interval,
)
from shop_rl.rl.prepare_tasks import SHOP_SYSTEM_PROMPT, load_system_prompt, prepare_rl_parquets

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{SRC_DIR}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(SRC_DIR)
    )
    if extra:
        env.update(extra)
    return env


def _make_eval_rec(
    *,
    task_id: int,
    paper_strict_score: float | None = 0.0,
    paper_strict_success: int = 0,
    reward_valid: int = 1,
    purchased: int = 0,
    r_type: float = 0.0,
    r_att: float = 0.0,
    r_option: float = 0.0,
    r_price: float = 0.0,
    termination_reason: str = "early_abstain",
    num_steps: int = 6,
    prompt_tokens: int = 1000,
    response_tokens: int = 800,
    assistant_gen_tokens: int = 200,
    elapsed_sec: float = 5.0,
) -> dict:
    return {
        "global_step": 0,
        "split": "val_dev474",
        "uid": f"u-{task_id}",
        "task_id": task_id,
        "reward_score": paper_strict_score,
        "paper_strict_score": paper_strict_score,
        "paper_strict_success": paper_strict_success,
        "reward_valid": reward_valid,
        "purchased": purchased,
        "r_type": r_type,
        "r_att": r_att,
        "r_option": r_option,
        "r_price": r_price,
        "termination_reason": termination_reason,
        "num_steps": num_steps,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "assistant_gen_tokens": assistant_gen_tokens,
        "elapsed_sec": elapsed_sec,
    }


def _read_parquet_task_ids(path: pathlib.Path) -> list[int]:
    table = pq.read_table(path, columns=["extra_info"])
    return [int(row["task_id"]) for row in table.column("extra_info").to_pylist()]


class TestPrepareTasks(unittest.TestCase):
    def test_load_system_prompt_independent_of_legacy_yaml(self):
        legacy_yaml = REPO_ROOT / "env/single_eval/configs/standard/eval_local_b_ep2.yaml"
        self.assertFalse(legacy_yaml.exists(), "legacy eval_local_b_ep2.yaml 应已清理")
        prompt = load_system_prompt()
        self.assertEqual(prompt, SHOP_SYSTEM_PROMPT)
        self.assertIn("Buy Now", prompt)
        self.assertIn("finish_without_purchase", prompt)
        self.assertTrue(
            hashlib.sha256(prompt.encode("utf-8")).hexdigest().startswith("5dad99f0")
        )

    def test_prepare_rl_parquets_real_data_exports_cleaned_splits(self):
        data_tasks = REPO_ROOT / "data/tasks"
        for obsolete in ("eval_dev_200.jsonl", "eval_dev_500_added_300.jsonl", "eval_dev_500.jsonl"):
            self.assertFalse((data_tasks / obsolete).exists(), f"{obsolete} 应已清理")

        eval_std_path = data_tasks / "eval_standard_task_ids.jsonl"
        val_path = data_tasks / "eval_dev_474.jsonl"
        holdout_path = data_tasks / "eval_holdout_915.jsonl"
        train_path = data_tasks / "grpo_tasks.jsonl"
        sft_path = data_tasks / "sft_candidate_tasks.jsonl"

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp)
            manifest = prepare_rl_parquets(
                train_jsonl=train_path,
                val_jsonl=val_path,
                holdout_jsonl=holdout_path,
                output_dir=out_dir,
                eval_standard_jsonl=eval_std_path,
                sft_candidate_jsonl=sft_path,
            )

            holdout_parquet = out_dir / "eval_holdout915.parquet"
            val474_parquet = out_dir / "val_dev474.parquet"
            val100_parquet = out_dir / "val_dev100.parquet"
            smoke20_parquet = out_dir / "val_smoke20.parquet"
            train_parquet = out_dir / "train.parquet"

            self.assertTrue(holdout_parquet.is_file())
            self.assertTrue(val474_parquet.is_file())
            self.assertTrue(smoke20_parquet.is_file())
            self.assertIn("eval_holdout915.parquet", manifest["files"])
            self.assertIn("val_dev474.parquet", manifest["files"])
            self.assertIn("val_smoke20.parquet", manifest["files"])
            self.assertIn("holdout_source", manifest)
            self.assertIn("eval_standard_source", manifest)

            holdout_ids = _read_parquet_task_ids(holdout_parquet)
            val474_ids = _read_parquet_task_ids(val474_parquet)
            val100_ids = _read_parquet_task_ids(val100_parquet)
            smoke20_ids = _read_parquet_task_ids(smoke20_parquet)
            train_ids = _read_parquet_task_ids(train_parquet)
            official_ids = set(load_task_ids(eval_std_path))
            sft_ids = set(load_task_ids(sft_path))

            self.assertEqual(len(holdout_ids), 915)
            self.assertEqual(len(set(holdout_ids)), 915)
            self.assertEqual(len(val474_ids), 474)
            self.assertEqual(len(val100_ids), 100)
            self.assertEqual(len(smoke20_ids), 20)
            self.assertEqual(smoke20_ids, val474_ids[:20])
            self.assertEqual(len(train_ids), 15550)
            self.assertEqual(manifest["files"]["eval_holdout915.parquet"]["split"], "eval_holdout915")
            self.assertGreaterEqual(min(train_ids), 1459)

            holdout_set = set(holdout_ids)
            val474_set = set(val474_ids)
            train_set = set(train_ids)

            self.assertFalse(holdout_set & val474_set)
            self.assertFalse(holdout_set & train_set)
            self.assertFalse(val474_set & train_set)
            self.assertFalse(holdout_set & sft_ids)
            self.assertTrue((holdout_set | val474_set) < official_ids)
            self.assertEqual(len(holdout_set | val474_set), 1389)

            # 验证静态价格不可核验拦截图拦截生效（task_id=51 为已知 multiple_effective_price_axes）
            bad_val = out_dir / "bad_val.jsonl"
            bad_train = out_dir / "bad_train.jsonl"
            bad_val.write_text('{"task_id": 51}\n', encoding="utf-8")
            bad_train.write_text('{"task_id": 1459}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "价格不可核验"):
                prepare_rl_parquets(
                    train_jsonl=bad_train,
                    val_jsonl=bad_val,
                    holdout_jsonl=None,
                    output_dir=out_dir / "bad_out",
                    val_limit=1,
                    smoke20_limit=1,
                    smoke_train_limit=1,
                    smoke_val_limit=1,
                    eval_standard_jsonl=None,
                    sft_candidate_jsonl=None,
                )


class TestEvalRun(unittest.TestCase):
    def test_plain_hf_model_plan_thinking_maxlen_and_keep_lora(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "smoke_train8.parquet").write_bytes(b"")
            out_dir = root / "eval_sft"

            plan = build_eval_plan(
                output_dir=out_dir,
                model_path=str(root / "models/sft_b_ep2"),
                config=REPO_ROOT / "configs/rl/eval_4090_24g.yaml",
                split="holdout915",
                data_dir=data_dir,
                overrides=["actor_rollout_ref.rollout.gpu_memory_utilization=0.65"],
                dry_run=True,
                base_env={},
            )

            cmd = plan["command"]
            self.assertEqual(plan["mode"], "hf_model")
            self.assertEqual(plan["label"], "eval_sft")
            self.assertFalse(plan["enable_thinking"])
            self.assertIsNone(plan["max_model_len"])
            self.assertIn("trainer.val_before_train=true", cmd)
            self.assertIn("trainer.val_only=true", cmd)
            self.assertIn("trainer.logger=[console,file]", cmd)
            self.assertIn("trainer.resume_mode=disable", cmd)
            self.assertIn("data.apply_chat_template_kwargs.enable_thinking=false", cmd)
            self.assertIn("actor_rollout_ref.rollout.val_kwargs.temperature=0.0", cmd)
            self.assertIn("actor_rollout_ref.rollout.val_kwargs.do_sample=false", cmd)
            self.assertIn("actor_rollout_ref.rollout.val_kwargs.n=1", cmd)
            self.assertIn("actor_rollout_ref.model.lora_rank=0", cmd)
            self.assertIn("actor_rollout_ref.model.lora.merge=false", cmd)
            self.assertIn("actor_rollout_ref.actor.use_kl_loss=false", cmd)
            self.assertIn(f"data.val_files={data_dir.resolve() / 'eval_holdout915.parquet'}", cmd)
            self.assertIn(f"data.train_files={data_dir.resolve() / 'smoke_train8.parquet'}", cmd)
            self.assertEqual(cmd[-1], "actor_rollout_ref.rollout.gpu_memory_utilization=0.65")

            env = plan["env"]
            self.assertEqual(env["SHOP_RL_LOG_DIR"], str((out_dir / "rollouts").resolve()))
            self.assertEqual(env["VERL_FILE_LOGGER_PATH"], str((out_dir / "verl_metrics.jsonl").resolve()))
            self.assertEqual(env["SHOP_ENV_BASE_URL"], "http://127.0.0.1:5705")
            self.assertEqual(env["SHOP_RL_ENV_SLOTS"], "32")
            self.assertEqual(env["SHOP_RL_MAX_STEPS"], "35")
            self.assertEqual(env["SHOP_RL_TURN_MAX_NEW_TOKENS"], "512")
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            self.assertTrue(env["PYTHONPATH"].startswith(str(SRC_DIR)))

            # 开思考 + 32K 上下文 + keep_lora
            plan_think = build_eval_plan(
                output_dir=out_dir,
                model_path=str(root / "models/sft_a_ep2"),
                keep_lora=True,
                enable_thinking=True,
                max_model_len=32768,
                split="smoke20",
                data_dir=data_dir,
                dry_run=True,
                base_env={},
            )
            self.assertTrue(plan_think["enable_thinking"])
            self.assertEqual(plan_think["max_model_len"], 32768)
            self.assertEqual(plan_think["env"]["SHOP_RL_TURN_MAX_NEW_TOKENS"], "8192")
            self.assertNotIn("actor_rollout_ref.model.lora_rank=0", plan_think["command"])
            self.assertIn("data.apply_chat_template_kwargs.enable_thinking=true", plan_think["command"])
            self.assertIn("actor_rollout_ref.rollout.max_model_len=32768", plan_think["command"])
            self.assertIn("actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=32768", plan_think["command"])
            self.assertIn("data.max_response_length=30720", plan_think["command"])
            self.assertIn(f"data.val_files={data_dir.resolve() / 'val_smoke20.parquet'}", plan_think["command"])

    def test_checkpoint_plan_and_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            ckpt = root / "checkpoints/global_step_100"
            out_dir = root / "eval_step100"

            plan = build_eval_plan(
                output_dir=out_dir,
                checkpoint=ckpt,
                data_dir=data_dir,
                dry_run=True,
                base_env={
                    "SHOP_RL_SFT_MODEL": str(root / "models/sft_b_ep2"),
                    "SHOP_ENV_BASE_URL": "http://127.0.0.1:5799",
                    "SHOP_RL_ENV_SLOTS": "16",
                },
            )
            cmd = plan["command"]
            self.assertEqual(plan["mode"], "checkpoint")
            self.assertEqual(plan["split"], "dev474")
            self.assertIn("trainer.resume_mode=resume_path", cmd)
            self.assertIn(f"trainer.resume_from_path={ckpt.resolve()}", cmd)
            self.assertIn("actor_rollout_ref.actor.checkpoint.load_contents=[model]", cmd)
            self.assertNotIn("actor_rollout_ref.model.lora_rank=0", cmd)
            self.assertIn(f"data.val_files={data_dir.resolve() / 'val_dev474.parquet'}", cmd)
            # 无 smoke_train8.parquet 时回退到 train.parquet
            self.assertIn(f"data.train_files={data_dir.resolve() / 'train.parquet'}", cmd)
            self.assertEqual(plan["env"]["SHOP_ENV_BASE_URL"], "http://127.0.0.1:5799")
            self.assertEqual(plan["env"]["SHOP_RL_ENV_SLOTS"], "16")

            # 非法 checkpoint 命名即使 dry_run 也必须报错
            with self.assertRaises(ValueError):
                build_eval_plan(
                    output_dir=out_dir,
                    model_path=str(root / "models/sft"),
                    checkpoint=root / "checkpoints/step_100",
                    dry_run=True,
                )
            with self.assertRaises(ValueError):
                build_eval_plan(
                    output_dir=out_dir,
                    model_path=str(root / "models/sft"),
                    checkpoint=root / "checkpoints/global_step_100/actor",
                    dry_run=True,
                )
            # 未指定模型路径且无 SHOP_RL_SFT_MODEL 必须报错
            with self.assertRaises(ValueError):
                build_eval_plan(output_dir=out_dir, dry_run=True, base_env={})
            # 非法 --max-model-len 必须报错
            with self.assertRaises(ValueError):
                build_eval_plan(
                    output_dir=out_dir,
                    model_path=str(root / "models/sft"),
                    max_model_len=2048,
                    dry_run=True,
                )
            # 非 dry-run 时 checkpoint 目录不存在必须报错
            with self.assertRaises(FileNotFoundError):
                build_eval_plan(
                    output_dir=out_dir,
                    model_path=str(root / "models/sft"),
                    checkpoint=ckpt,
                    dry_run=False,
                )

    def test_build_verl_eval_command_pure_helper(self):
        cmd = build_verl_eval_command(
            verl_python="/usr/bin/python3",
            config=REPO_ROOT / "configs/rl/vanilla_grpo_a800_80g.yaml",
            model_path="/models/base",
            val_parquet=pathlib.Path("/data/val_dev474.parquet"),
            train_parquet=pathlib.Path("/data/smoke_train8.parquet"),
            output_dir=pathlib.Path("/out/run1"),
            label="run1",
            agent_loops_config=REPO_ROOT / "configs/rl/agent_loops.yaml",
        )
        self.assertEqual(cmd[:3], ["/usr/bin/python3", "-m", "verl.trainer.main_ppo"])
        self.assertIn("--config-name", cmd)
        self.assertEqual(cmd[cmd.index("--config-name") + 1], "vanilla_grpo_a800_80g")

    def test_build_eval_plan_discovers_venv_cuda12_ld_library_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "smoke_train8.parquet").write_bytes(b"")

            venv_dir = root / "fake_venv"
            (venv_dir / "bin").mkdir(parents=True)
            venv_py = venv_dir / "bin/python"
            venv_py.symlink_to(sys.executable)

            nvidia_root = venv_dir / "lib/python3.10/site-packages/nvidia"
            for rel in (
                "cuda_runtime/lib/libcudart.so.12",
                "cublas/lib/libcublas.so.12",
                "cu13/lib/libcudart.so.13",
            ):
                p = nvidia_root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"")

            plan = build_eval_plan(
                output_dir=root / "eval_out",
                model_path=str(root / "models/sft_a_ep2"),
                verl_python=str(venv_py),
                data_dir=data_dir,
                dry_run=True,
                base_env={"LD_LIBRARY_PATH": "/custom/lib"},
            )
            expected_cudart = os.path.abspath(str(nvidia_root / "cuda_runtime/lib"))
            expected_cublas = os.path.abspath(str(nvidia_root / "cublas/lib"))
            excluded_cu13 = os.path.abspath(str(nvidia_root / "cu13/lib"))

            ld_parts = plan["env"]["LD_LIBRARY_PATH"].split(os.pathsep)
            self.assertEqual(ld_parts[0], expected_cudart)
            self.assertIn(expected_cublas, ld_parts)
            self.assertEqual(ld_parts[-1], "/custom/lib")
            self.assertNotIn(excluded_cu13, ld_parts)

    def test_cli_dry_run_and_fake_verl_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            out_dry = root / "dry_out"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "shop_rl.eval.run",
                    "--config",
                    str(REPO_ROOT / "configs/rl/eval_4090_24g.yaml"),
                    "--model-path",
                    str(root / "fake_model"),
                    "--split",
                    "holdout915",
                    "--enable-thinking",
                    "--max-model-len",
                    "32768",
                    "--output-dir",
                    str(out_dry),
                    "--dry-run",
                ],
                env=_subprocess_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["mode"], "hf_model")
            self.assertEqual(payload["split"], "holdout915")
            self.assertTrue(payload["enable_thinking"])
            self.assertEqual(payload["max_model_len"], 32768)
            self.assertEqual(payload["env"]["SHOP_RL_TURN_MAX_NEW_TOKENS"], "8192")
            self.assertFalse(out_dry.exists(), "--dry-run 不应创建输出目录")

            # 使用本地模拟 verl 解释器验证真实子进程落盘 eval_meta.json 与 eval_summary.json
            model_dir = root / "model"
            model_dir.mkdir()
            val_pq = root / "val.parquet"
            train_pq = root / "train.parquet"
            rows = [{"extra_info": {"task_id": tid, "split": "val"}} for tid in (1, 2)]
            pq.write_table(pa.Table.from_pylist(rows), val_pq)
            pq.write_table(pa.Table.from_pylist(rows), train_pq)

            fake_py = root / "fake_verl_python.py"
            fake_py.write_text(
                f"#!{sys.executable}\n"
                "import json, os, pathlib\n"
                "log_dir = pathlib.Path(os.environ['SHOP_RL_LOG_DIR'])\n"
                "log_dir.mkdir(parents=True, exist_ok=True)\n"
                "recs = [\n"
                "  {'task_id': 1, 'reward_valid': 1, 'reward_score': 1.0, 'paper_strict_score': 1.0, 'paper_strict_success': 1, 'purchased': 1, 'r_type': 1.0, 'r_att': 1.0, 'r_option': 1.0, 'r_price': 1.0, 'termination_reason': 'purchase', 'num_steps': 5, 'prompt_tokens': 500, 'response_tokens': 300, 'assistant_gen_tokens': 100, 'elapsed_sec': 2.0},\n"
                "  {'task_id': 2, 'reward_valid': 1, 'reward_score': 0.0, 'paper_strict_score': 0.0, 'paper_strict_success': 0, 'purchased': 0, 'r_type': 0.0, 'r_att': 0.0, 'r_option': 0.0, 'r_price': 0.0, 'termination_reason': 'early_abstain', 'num_steps': 8, 'prompt_tokens': 600, 'response_tokens': 400, 'assistant_gen_tokens': 150, 'elapsed_sec': 3.0},\n"
                "]\n"
                "(log_dir / 'rollouts-999.jsonl').write_text('\\n'.join(json.dumps(r) for r in recs) + '\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            fake_py.chmod(fake_py.stat().st_mode | stat.S_IXUSR)

            out_exec = root / "exec_out"
            proc2 = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "shop_rl.eval.run",
                    "--model-path",
                    str(model_dir),
                    "--val-parquet",
                    str(val_pq),
                    "--train-parquet",
                    str(train_pq),
                    "--output-dir",
                    str(out_exec),
                    "--verl-python",
                    str(fake_py),
                ],
                env=_subprocess_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc2.returncode, 0, msg=proc2.stderr)
            meta = json.loads((out_exec / "eval_meta.json").read_text(encoding="utf-8"))
            summary = json.loads((out_exec / "eval_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["returncode"], 0)
            self.assertEqual(summary["expected_tasks"], 2)
            self.assertEqual(summary["paper_strict_success_count"], 1)
            self.assertAlmostEqual(summary["paper_strict_success_rate"], 0.5)


class TestEvalSummarize(unittest.TestCase):
    def test_wilson_and_mcnemar_exact_values(self):
        self.assertIsNone(wilson_interval(0, 0))
        lo0, hi0 = wilson_interval(0, 10)
        self.assertGreaterEqual(lo0, 0.0)
        self.assertLessEqual(hi0, 1.0)
        self.assertAlmostEqual(lo0, 0.0, places=9)

        lo_mid, hi_mid = wilson_interval(5, 10)
        self.assertAlmostEqual(lo_mid + hi_mid, 1.0, places=9)
        self.assertTrue(0.0 < lo_mid < 0.5 < hi_mid < 1.0)

        lo1, hi1 = wilson_interval(10, 10)
        self.assertAlmostEqual(hi1, 1.0, places=9)

        # 手算精确双侧 McNemar p 值验证
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact_p(1, 9), 22 / 1024, places=12)
        self.assertAlmostEqual(mcnemar_exact_p(0, 6), 2 / 64, places=12)
        self.assertEqual(mcnemar_exact_p(5, 5), 1.0)
        # 大 discordant 不溢出
        p_large = mcnemar_exact_p(600, 600)
        self.assertEqual(p_large, 1.0)

    def test_per_run_metrics_dedup_invalid_missing_and_subscores(self):
        records = [
            # task 1 重复两次：先失败后成功，去重保留最后一条
            _make_eval_rec(task_id=1, paper_strict_score=0.0, paper_strict_success=0, termination_reason="early_abstain"),
            _make_eval_rec(
                task_id=1,
                paper_strict_score=1.0,
                paper_strict_success=1,
                purchased=1,
                r_type=1.0,
                r_att=1.0,
                r_option=1.0,
                r_price=1.0,
                termination_reason="purchase",
                num_steps=4,
                prompt_tokens=1000,
                response_tokens=500,
                assistant_gen_tokens=100,
                elapsed_sec=4.0,
            ),
            # task 2 已购但超预算（r_price=0）
            _make_eval_rec(
                task_id=2,
                paper_strict_score=0.0,
                paper_strict_success=0,
                purchased=1,
                r_type=1.0,
                r_att=0.5,
                r_option=1.0,
                r_price=0.0,
                termination_reason="purchase",
                num_steps=10,
                prompt_tokens=1200,
                response_tokens=1800,
                assistant_gen_tokens=300,
                elapsed_sec=10.0,
            ),
            # task 3 不可核验（invalid）
            _make_eval_rec(
                task_id=3,
                paper_strict_score=None,
                paper_strict_success=0,
                reward_valid=0,
                purchased=1,
                termination_reason="purchase",
                num_steps=8,
                prompt_tokens=1000,
                response_tokens=1000,
                assistant_gen_tokens=200,
                elapsed_sec=8.0,
            ),
            # task 4 弃购失败
            _make_eval_rec(
                task_id=4,
                paper_strict_score=0.0,
                paper_strict_success=0,
                termination_reason="invalid_tool_call",
                num_steps=2,
                prompt_tokens=900,
                response_tokens=100,
                assistant_gen_tokens=50,
                elapsed_sec=2.0,
            ),
            # task 99 不在 expected_task_ids 中
            _make_eval_rec(task_id=99, paper_strict_score=1.0, paper_strict_success=1, purchased=1, termination_reason="purchase"),
        ]
        # expected 为 [1, 2, 3, 4, 5] -> task 5 缺失
        summary = summarize_records(records, label="m1", expected_task_ids=[1, 2, 3, 4, 5])

        self.assertEqual(summary["expected_tasks"], 5)
        self.assertEqual(summary["duplicate_count"], 1)
        self.assertEqual(summary["unexpected_count"], 1)
        self.assertEqual(summary["total_records"], 4)
        self.assertEqual(summary["valid_count"], 3)
        self.assertEqual(summary["invalid_count"], 1)
        self.assertEqual(summary["missing_count"], 1)
        self.assertAlmostEqual(summary["valid_coverage_rate"], 3 / 5)

        self.assertEqual(summary["paper_strict_success_count"], 1)
        self.assertAlmostEqual(summary["paper_strict_success_rate"], 1 / 3)
        self.assertAlmostEqual(summary["full_denominator_success_rate"], 1 / 5)
        self.assertAlmostEqual(summary["paper_strict_score_mean"], 1 / 3)
        self.assertAlmostEqual(summary["purchase_rate"], 2 / 3)

        sub = summary["purchased_subscores"]
        self.assertAlmostEqual(sub["r_type"], 1.0)
        self.assertAlmostEqual(sub["r_att"], 0.75)
        self.assertAlmostEqual(sub["r_option"], 1.0)
        self.assertAlmostEqual(sub["r_price"], 0.5)
        self.assertAlmostEqual(sub["over_budget_rate"], 0.5)

        term_rates = summary["termination"]["rates"]
        self.assertAlmostEqual(term_rates["purchase"], 3 / 4)
        self.assertAlmostEqual(term_rates["invalid_tool"], 1 / 4)
        self.assertAlmostEqual(summary["cost"]["num_steps_mean"], (4 + 10 + 8 + 2) / 4)

    def test_paired_comparison_subsets_and_cli(self):
        # 12 题：2 both_success, 1 first_only, 9 second_only -> McNemar p = 22/1024
        recs_a = []
        recs_b = []
        for tid in range(1, 13):
            succ_a = tid in (1, 2, 3)
            succ_b = tid in (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            recs_a.append(
                _make_eval_rec(
                    task_id=tid,
                    paper_strict_score=1.0 if succ_a else 0.0,
                    paper_strict_success=int(succ_a),
                    purchased=int(succ_a),
                    termination_reason="purchase" if succ_a else "early_abstain",
                )
            )
            recs_b.append(
                _make_eval_rec(
                    task_id=tid,
                    paper_strict_score=1.0 if succ_b else 0.25,
                    paper_strict_success=int(succ_b),
                    purchased=int(succ_b),
                    termination_reason="purchase" if succ_b else "early_abstain",
                )
            )
        # 额外放一条在 B 中 invalid 的 task 13，确认不会计入配对交集
        recs_a.append(_make_eval_rec(task_id=13, paper_strict_score=1.0, paper_strict_success=1))
        recs_b.append(_make_eval_rec(task_id=13, paper_strict_score=None, reward_valid=0))

        pair = compare_runs("sft", recs_a, "rl", recs_b)
        self.assertEqual(pair["n_paired"], 12)
        self.assertEqual(pair["both_success"], 2)
        self.assertEqual(pair["both_fail"], 0)
        self.assertEqual(pair["first_only"], 1)
        self.assertEqual(pair["second_only"], 9)
        self.assertAlmostEqual(pair["delta_success_rate"], (11 - 3) / 12)
        self.assertAlmostEqual(pair["mcnemar_p_value"], 22 / 1024, places=12)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dir_a = root / "sft"
            dir_b = root / "rl"
            (dir_a / "rollouts").mkdir(parents=True)
            (dir_b / "rollouts").mkdir(parents=True)
            (dir_a / "rollouts/rollouts-1.jsonl").write_text(
                "\n".join(json.dumps(r) for r in recs_a) + "\n", encoding="utf-8"
            )
            (dir_b / "rollouts/rollouts-1.jsonl").write_text(
                "\n".join(json.dumps(r) for r in recs_b) + "\n", encoding="utf-8"
            )

            expected_jsonl = root / "expected.jsonl"
            expected_jsonl.write_text(
                "\n".join(json.dumps({"task_id": tid}) for tid in range(1, 14)) + "\n",
                encoding="utf-8",
            )
            subset_pq = root / "subset_first6.parquet"
            pq.write_table(
                pa.Table.from_pylist([{"extra_info": {"task_id": tid}} for tid in range(1, 7)]),
                subset_pq,
            )
            out_json = root / "summary.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "shop_rl.eval.summarize",
                    "--run",
                    f"sft={dir_a}",
                    "--run",
                    f"rl={dir_b}",
                    "--expected-tasks",
                    str(expected_jsonl),
                    "--subset",
                    f"first6={subset_pq}",
                    "--baseline",
                    "sft",
                    "--out-json",
                    str(out_json),
                ],
                env=_subprocess_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("sft -> rl", proc.stdout)
            self.assertIn("Subset: first6", proc.stdout)

            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(report["expected_tasks"], 13)
            self.assertEqual(report["pairs"]["sft_vs_rl"]["n_paired"], 12)
            self.assertAlmostEqual(report["pairs"]["sft_vs_rl"]["mcnemar_p_value"], 22 / 1024, places=12)
            self.assertEqual(report["subsets"]["first6"]["expected_tasks"], 6)
            self.assertEqual(report["subsets"]["first6"]["pairs"]["sft_vs_rl"]["n_paired"], 6)
            self.assertEqual(report["subsets"]["first6"]["pairs"]["sft_vs_rl"]["first_only"], 1)
            self.assertEqual(report["subsets"]["first6"]["pairs"]["sft_vs_rl"]["second_only"], 3)

            # 位置参数 + 直接放在目录根的 rollouts-*.jsonl + 无 --baseline 时全两两比较
            dir_c = root / "rl_v2"
            dir_c.mkdir()
            (dir_c / "rollouts-1.jsonl").write_text(
                "\n".join(json.dumps(r) for r in recs_b) + "\n", encoding="utf-8"
            )
            out_all = root / "summary_all.json"
            proc_all = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "shop_rl.eval.summarize",
                    str(dir_a),
                    str(dir_b),
                    str(dir_c),
                    "--out-json",
                    str(out_all),
                ],
                env=_subprocess_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_all.returncode, 0, msg=proc_all.stderr)
            report_all = json.loads(out_all.read_text(encoding="utf-8"))
            self.assertEqual(
                set(report_all["pairs"].keys()),
                {"sft_vs_rl", "sft_vs_rl_v2", "rl_vs_rl_v2"},
            )
            self.assertEqual(report_all["runs"]["rl_v2"]["valid_count"], 12)

            # build_comparison_report 直接调用与 baseline 校验
            with self.assertRaises(ValueError):
                build_comparison_report([("sft", dir_a), ("rl", dir_b)], baseline="unknown")


class TestShopRLCLI(unittest.TestCase):
    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "shop_rl.cli", *args],
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_help_and_eval_subcommands(self):
        p_help = self._run_cli("--help")
        self.assertEqual(p_help.returncode, 0, msg=p_help.stderr)
        self.assertIn("shop_rl eval <run|summarize>", p_help.stdout)
        self.assertNotIn("shop_rl report", p_help.stdout)

        p_run = self._run_cli("eval", "run", "--help")
        self.assertEqual(p_run.returncode, 0, msg=p_run.stderr)
        self.assertIn("--dry-run", p_run.stdout)

        p_sum = self._run_cli("eval", "summarize", "--help")
        self.assertEqual(p_sum.returncode, 0, msg=p_sum.stderr)
        self.assertIn("--baseline", p_sum.stdout)

    def test_removed_commands_exit_code_2(self):
        for bad_args in (
            ("report", "curves"),
            ("report", "dev500"),
            ("eval", "lane"),
            ("eval", "summarize-dev500"),
            ("eval", "orchestrate"),
        ):
            proc = self._run_cli(*bad_args)
            self.assertEqual(proc.returncode, 2, msg=f"期望 {bad_args} 返回退出码 2，实际 {proc.returncode}")


if __name__ == "__main__":
    unittest.main()

"""shop_rl.rl.monitor 单元测试：覆盖 GRPO 组分类、已购子分、终止统计、断点容错与 CLI。"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from shop_rl.rl.monitor import aggregate_all_steps, load_rollout_records, percentile


def _make_rec(
    *,
    global_step: int = 1,
    split: str = "train",
    uid: str = "u1",
    reward_score: float | None = 0.0,
    reward_valid: int = 1,
    purchased: int = 0,
    paper_strict_success: int = 0,
    r_type: float = 0.0,
    r_att: float = 0.0,
    r_option: float = 0.0,
    r_price: float = 0.0,
    termination_reason: str = "early_abstain",
    num_steps: int = 5,
    prompt_tokens: int = 800,
    response_tokens: int = 1200,
    assistant_gen_tokens: int = 200,
    elapsed_sec: float = 10.0,
):
    return {
        "global_step": global_step,
        "split": split,
        "uid": uid,
        "task_id": 2000,
        "reward_score": reward_score,
        "reward_valid": reward_valid,
        "purchased": purchased,
        "paper_strict_score": reward_score if reward_valid else None,
        "paper_strict_success": paper_strict_success,
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


class TestRLMonitor(unittest.TestCase):
    def test_group_classification_and_invalid_exclusion(self):
        records = [
            # Group 1 (has_variance): [1.0, 0.0]
            _make_rec(uid="g_var", reward_score=1.0, purchased=1, paper_strict_success=1, r_type=1.0, r_att=1.0, r_option=1.0, r_price=1.0, termination_reason="purchase"),
            _make_rec(uid="g_var", reward_score=0.0, termination_reason="early_abstain"),
            # Group 2 (all_zero): [0.0, 0.0]
            _make_rec(uid="g_zero", reward_score=0.0, termination_reason="invalid_tool_call"),
            _make_rec(uid="g_zero", reward_score=0.0, termination_reason="context_limit"),
            # Group 3 (all_one): [1.0, 1.0]
            _make_rec(uid="g_one", reward_score=1.0, purchased=1, paper_strict_success=1, r_type=1.0, r_att=1.0, r_option=1.0, r_price=1.0, termination_reason="purchase"),
            _make_rec(uid="g_one", reward_score=1.0, purchased=1, paper_strict_success=1, r_type=1.0, r_att=1.0, r_option=1.0, r_price=1.0, termination_reason="purchase"),
            # Group 4 (same_fractional): [0.5, 0.5]
            _make_rec(uid="g_frac", reward_score=0.5, purchased=1, r_type=1.0, r_att=0.5, r_option=1.0, r_price=1.0, termination_reason="purchase"),
            _make_rec(uid="g_frac", reward_score=0.5, purchased=1, r_type=1.0, r_att=1.0, r_option=0.5, r_price=1.0, termination_reason="purchase"),
            # Group 5 (incomplete: size 1 != 2)
            _make_rec(uid="g_short", reward_score=0.0, termination_reason="max_steps"),
            # Group 6 (contains invalid reward_valid=0 -> excluded from complete groups & valid means)
            _make_rec(uid="g_inv", reward_score=None, reward_valid=0, purchased=1, termination_reason="purchase"),
            _make_rec(uid="g_inv", reward_score=0.0, termination_reason="no_tool_call"),
        ]

        rows = aggregate_all_steps(records, group_size=2)
        self.assertEqual(len(rows), 1)
        row = rows[0]

        self.assertEqual(row["global_step"], 1)
        self.assertEqual(row["split"], "train")
        self.assertEqual(row["num_trajectories"], 11)
        self.assertEqual(row["num_valid"], 10)
        self.assertEqual(row["num_invalid"], 1)
        self.assertAlmostEqual(row["valid_coverage_rate"], 10 / 11)

        # Complete groups: 4 (g_var, g_zero, g_one, g_frac) out of 6 total uids
        self.assertEqual(row["num_groups_total"], 6)
        self.assertEqual(row["num_groups_complete"], 4)
        self.assertEqual(row["num_groups_incomplete"], 2)
        self.assertAlmostEqual(row["group_has_variance_ratio"], 0.25)
        self.assertAlmostEqual(row["group_all_zero_ratio"], 0.25)
        self.assertAlmostEqual(row["group_all_one_ratio"], 0.25)
        self.assertAlmostEqual(row["group_same_fractional_ratio"], 0.25)
        self.assertAlmostEqual(row["group_reward_std_mean"], 0.5 / 4)

        # Valid reward shares: scores = [1, 0, 0, 0, 1, 1, 0.5, 0.5, 0, 0] -> sum=4.0 / 10
        self.assertAlmostEqual(row["reward_mean"], 0.4)
        self.assertAlmostEqual(row["reward_zero_ratio"], 0.5)
        self.assertAlmostEqual(row["reward_fractional_ratio"], 0.2)
        self.assertAlmostEqual(row["reward_one_ratio"], 0.3)
        self.assertAlmostEqual(row["purchase_rate"], 0.5)
        self.assertAlmostEqual(row["paper_strict_success_rate"], 0.3)

        # Purchased-only sub-scores over the 5 valid purchases
        self.assertAlmostEqual(row["purchased_r_type"], 1.0)
        self.assertAlmostEqual(row["purchased_r_att"], (1 + 1 + 1 + 0.5 + 1.0) / 5)
        self.assertAlmostEqual(row["purchased_r_option"], (1 + 1 + 1 + 1.0 + 0.5) / 5)
        self.assertAlmostEqual(row["purchased_r_price"], 1.0)
        self.assertAlmostEqual(row["purchased_over_budget_rate"], 0.0)

        # Terminations across all 11 trajectories
        self.assertAlmostEqual(row["term_purchase_rate"], 6 / 11)
        self.assertAlmostEqual(row["term_abstain_rate"], 1 / 11)
        self.assertAlmostEqual(row["term_invalid_tool_rate"], 1 / 11)
        self.assertAlmostEqual(row["term_context_limit_rate"], 1 / 11)
        self.assertAlmostEqual(row["term_max_steps_rate"], 1 / 11)
        self.assertAlmostEqual(row["term_no_tool_call_rate"], 1 / 11)

    def test_zero_purchases_sets_purchased_subscores_none_and_over_budget(self):
        records = [
            _make_rec(uid="u1", reward_score=0.0, termination_reason="repeat_loop", num_steps=10, prompt_tokens=1000, response_tokens=500, assistant_gen_tokens=100, elapsed_sec=4.0),
            _make_rec(uid="u1", reward_score=0.0, termination_reason="finish_length_truncated", num_steps=20, prompt_tokens=1000, response_tokens=3000, assistant_gen_tokens=500, elapsed_sec=12.0),
        ]
        rows = aggregate_all_steps(records, group_size=2)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIsNone(row["purchased_r_type"])
        self.assertIsNone(row["purchased_over_budget_rate"])
        self.assertAlmostEqual(row["term_repeat_loop_rate"], 0.5)
        self.assertAlmostEqual(row["term_length_truncated_rate"], 0.5)
        self.assertAlmostEqual(row["num_steps_p95"], percentile([10, 20], 95.0))
        self.assertAlmostEqual(row["total_tokens_p95"], percentile([1500, 4000], 95.0))

        # Verify over-budget rate counts valid purchases with r_price == 0
        rec_over = _make_rec(uid="u2", reward_score=0.0, purchased=1, r_type=1.0, r_att=1.0, r_option=1.0, r_price=0.0, termination_reason="purchase")
        rows2 = aggregate_all_steps([rec_over, _make_rec(uid="u2", reward_score=0.0)], group_size=2)
        self.assertAlmostEqual(rows2[0]["purchased_over_budget_rate"], 1.0)
        self.assertAlmostEqual(rows2[0]["purchased_r_price"], 0.0)

    def test_load_multifile_partial_tail_split_separation_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = pathlib.Path(tmp)
            p1 = log_dir / "rollouts-101.jsonl"
            p2 = log_dir / "rollouts-102.jsonl"
            p1.write_text(
                "\n".join(
                    [
                        json.dumps(_make_rec(global_step=0, split="val_dev100", uid="v1", reward_score=1.0, purchased=1, paper_strict_success=1, termination_reason="purchase")),
                        json.dumps(_make_rec(global_step=1, split="train", uid="t1", reward_score=0.0)),
                        "",
                        '{"global_step": 1, "incomplete_trailing_line',  # half-written trailing line
                    ]
                ),
                encoding="utf-8",
            )
            p2.write_text(
                "\n".join(
                    [
                        json.dumps(_make_rec(global_step=1, split="train", uid="t1", reward_score=1.0, purchased=1, paper_strict_success=1, termination_reason="purchase")),
                        json.dumps(_make_rec(global_step=1, split="val", uid="v2", reward_score=0.0)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = load_rollout_records(log_dir)
            self.assertEqual(len(loaded), 4)

            out_jsonl = log_dir / "shop_metrics.jsonl"
            env = dict(os.environ)
            src_dir = str(pathlib.Path(__file__).resolve().parents[1] / "src")
            env["PYTHONPATH"] = f"{src_dir}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_dir
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "shop_rl.cli",
                    "rl",
                    "monitor",
                    "--log-dir",
                    str(log_dir),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--group-size",
                    "2",
                    "--no-tb",
                    "--once",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            out_rows = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([(r["global_step"], r["split"]) for r in out_rows], [(0, "val"), (1, "train"), (1, "val")])
            train_row = out_rows[1]
            self.assertEqual(train_row["num_groups_complete"], 1)
            self.assertAlmostEqual(train_row["group_has_variance_ratio"], 1.0)
            self.assertAlmostEqual(out_rows[0]["reward_mean"], 1.0)
            self.assertAlmostEqual(out_rows[2]["reward_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from web_agent_site.engine.config import load_config, validate_config
from web_agent_site.engine.paper_strict import (
    PAPER_STRICT_VERSION,
    _category_reward,
    score_purchase,
    terminal_reward_detail,
)


class PaperStrictRewardTest(unittest.TestCase):
    def setUp(self):
        self.product = {
            "asin": "alternative",
            "category": "家电›冰箱",
            "query": "",
            "title": "宿舍冰箱",
            "Title": "宿舍冰箱",
            "Attributes": ["静音"],
            "BulletPoints": [],
            "Description": "",
        }
        self.goal = {
            "asin": "target",
            "category": "家电›冰箱",
            "query": "",
            "name": "宿舍冰箱",
            "attributes": ["静音", "节能"],
            "goal_options": ["白色", "60升"],
            "price_upper": 60,
        }

    def test_partial_match_multiplies_fractions_at_budget_boundary(self):
        scores = score_purchase(
            self.product, self.goal, {"颜色": "白色"}, {"status": "pass", "price": 60},
        )
        self.assertEqual(
            [scores[key] for key in ("r_type", "r_att", "r_option", "r_price")],
            [1.0, 0.5, 0.5, 1.0],
        )
        self.assertEqual(scores["paper_strict_score"], 0.25)
        self.assertFalse(scores["paper_strict_success"])
        self.assertNotIn("price", self.product)

    def test_over_budget_and_unverifiable_price_differ(self):
        for resolution, expected in [
            ({"status": "pass", "price": 61}, 0.0),
            ({"status": "unverifiable", "price": None}, None),
        ]:
            with self.subTest(resolution=resolution):
                result = score_purchase(self.product, self.goal, {"颜色": "白色"}, resolution)
                self.assertEqual(result["r_price"], expected)
                self.assertEqual(result["paper_strict_score"], expected)
                detail = terminal_reward_detail("max_steps", result)
                self.assertEqual(detail["reward_valid"], expected is not None)

    def test_no_budget_or_option_requirement_and_alternative_asin(self):
        goal = {**self.goal, "price_upper": None, "goal_options": [], "attributes": ["静音"]}
        scores = score_purchase(
            self.product, goal, {}, {"status": "unverifiable", "price": None},
        )
        self.assertEqual(scores["paper_strict_score"], 1.0)
        self.assertTrue(scores["paper_strict_success"])

    def test_category_thresholds_and_missing_query(self):
        product = {**self.product, "category": "宠物›食品"}
        # Control keyword overlap so the appendix's exact thresholds are reproducible.
        for shared, expected in ((0, 0.0), (1, 0.1), (2, 0.5), (4, 0.5), (5, 1.0)):
            with self.subTest(shared=shared):
                with patch(
                    "web_agent_site.engine.paper_strict._title_keywords",
                    side_effect=(set(range(20)), set(range(shared))),
                ):
                    self.assertEqual(_category_reward(product, self.goal), expected)

    def test_nonpurchase_zero_preserves_termination_reason(self):
        for reason in ("repeat_loop", "abstain", "max_steps"):
            with self.subTest(reason=reason):
                detail = terminal_reward_detail(reason)
                self.assertEqual(detail["reward_version"], PAPER_STRICT_VERSION)
                self.assertEqual(detail["terminal_utility"], 0.0)
                self.assertEqual(detail["termination_reason"], reason)
                self.assertNotIn("reward_v3", detail)

    def test_bundled_strict_mode_contract(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "environment.json"
        self.assertEqual(load_config(path)["reward"]["mode"], "paper_strict")
        config = json.loads(path.read_text(encoding="utf-8"))
        del config["reward"]["mode"]
        with self.assertRaisesRegex(ValueError, "reward mode"):
            validate_config(config)
        config["reward"]["mode"] = "reward_v3"
        with self.assertRaisesRegex(ValueError, "reward mode"):
            validate_config(config)

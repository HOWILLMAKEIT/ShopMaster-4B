import unittest

from web_agent_site.engine.constraints import explicit_budget_from_instruction


class ExplicitBudgetTest(unittest.TestCase):
    def test_explicit_budget_is_used_as_written(self):
        self.assertEqual(
            explicit_budget_from_instruction("预算在1000元以下。"),
            1000.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格不超过 2199 元"),
            2199.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格在70元左右"),
            77.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格在130-140元之间"),
            140.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格30元到40元之间"),
            40.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("预算在1万元左右"),
            11000.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("预算1万2以内"),
            12000.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("预算不超过2,500元"),
            2500.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("买一台微波炉，别超过3千"),
            3000.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("500元以内的电饭煲"),
            500.0,
        )

    def test_open_ended_and_missing_budget_return_none(self):
        self.assertIsNone(explicit_budget_from_instruction("预算4k+"))
        self.assertIsNone(explicit_budget_from_instruction("买一个舒适的枕头"))
        self.assertIsNone(explicit_budget_from_instruction(""))
        self.assertIsNone(explicit_budget_from_instruction(None))

    def test_lower_bound_markers_are_not_treated_as_upper_bounds(self):
        for text in (
            "价格60元起",
            "预算60元以上",
            "价格100元起步",
            "价格200元起价",
            "价格500元往上",
        ):
            with self.subTest(text=text):
                self.assertIsNone(explicit_budget_from_instruction(text))
        self.assertEqual(
            explicit_budget_from_instruction("价格60元起，不超过200元"),
            200.0,
        )

    def test_non_price_unit_traps_are_ignored(self):
        for text in (
            "功率不超过3千瓦",
            "容量不超过2万毫安",
            "速率不超过1千兆",
            "功率不超过2kw",
        ):
            with self.subTest(text=text):
                self.assertIsNone(explicit_budget_from_instruction(text))
        self.assertEqual(
            explicit_budget_from_instruction("功率不超过3千瓦，预算不超过500元"),
            500.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("功率不超过3千瓦，别超过2千"),
            2000.0,
        )


if __name__ == "__main__":
    unittest.main()

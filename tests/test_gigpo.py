"""无折扣、按轨迹去重 GiGPO 锚点与优势估计器单元测试。"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from shop_rl.rl.gigpo import (
    canonical_anchor_key,
    compute_anchor_state_id,
    compute_gigpo_advantage,
)


def _make_obs(page_type: str, **extra):
    obs = {"page_type": page_type}
    obs.update(extra)
    return obs


class AnchorStateTests(unittest.TestCase):
    def test_skip_home_and_non_content_pages(self):
        for page_type in ("search_home", "terminal", "unknown", ""):
            obs = _make_obs(page_type)
            self.assertIsNone(canonical_anchor_key(obs))
            self.assertEqual(compute_anchor_state_id(obs), 0)
        self.assertIsNone(canonical_anchor_key(None))
        self.assertEqual(compute_anchor_state_id(None), 0)

    def test_search_product_subpage_canonicalization_and_determinism(self):
        s1 = _make_obs(
            "search_results",
            normalized_query="red running shoes",
            query="Red Running Shoes",
            page=2,
            total_results=50,
        )
        s2 = _make_obs(
            "search_results",
            normalized_query="red running shoes",
            query="different display",
            page=2,
            total_results=10,
        )
        self.assertEqual(canonical_anchor_key(s1), ("search", "red running shoes", 2))
        self.assertEqual(compute_anchor_state_id(s1), compute_anchor_state_id(s2))
        self.assertGreater(compute_anchor_state_id(s1), 0)

        p1 = _make_obs(
            "product_detail",
            product={"asin": "B001XYZ", "title": "Title A", "price": 19.99},
            selected_options={"size": "M", "color": "Black"},
            available_options={"size": ["M", "L"]},
        )
        p2 = _make_obs(
            "product_detail",
            product={"asin": "B001XYZ", "title": "Other", "price": 99.99},
            selected_options={"color": "Black", "size": "M"},
        )
        self.assertEqual(
            canonical_anchor_key(p1),
            ("item", "B001XYZ", (("color", "Black"), ("size", "M"))),
        )
        self.assertEqual(compute_anchor_state_id(p1), compute_anchor_state_id(p2))

        sub1 = _make_obs(
            "information_subpage",
            product={"asin": "B001XYZ"},
            subpage="Features",
            selected_options={"color": "Black"},
            content="foo",
        )
        sub2 = _make_obs(
            "information_subpage",
            product={"asin": "B001XYZ"},
            subpage="features",
            selected_options={"color": "Black"},
            content="bar",
        )
        self.assertEqual(
            canonical_anchor_key(sub1),
            ("subpage", "B001XYZ", "features", (("color", "Black"),)),
        )
        self.assertEqual(compute_anchor_state_id(sub1), compute_anchor_state_id(sub2))

    def test_selected_options_prevent_state_aliasing(self):
        unselected = _make_obs(
            "product_detail",
            product={"asin": "B001XYZ"},
            selected_options={},
        )
        selected_red = _make_obs(
            "product_detail",
            product={"asin": "B001XYZ"},
            selected_options={"color": "red"},
        )
        selected_blue = _make_obs(
            "product_detail",
            product={"asin": "B001XYZ"},
            selected_options={"color": "blue"},
        )
        sub_empty = _make_obs(
            "information_subpage",
            product={"asin": "B001XYZ"},
            subpage="features",
            selected_options={},
        )
        ids = {
            compute_anchor_state_id(unselected),
            compute_anchor_state_id(selected_red),
            compute_anchor_state_id(selected_blue),
            compute_anchor_state_id(sub_empty),
        }
        self.assertEqual(len(ids), 4)
        self.assertNotIn(0, ids)


class GiGPOAdvantageTests(unittest.TestCase):
    def test_exact_four_trajectory_math_and_omega_zero_fallback(self):
        # 4 trajectories in one prompt group u1: rewards [1.0, 1.0, 0.0, 0.0]
        # Token layout (len=6):
        #   t=0: home (mask=1, anchor=0) -> A^S must stay 0
        #   t=1: obs  (mask=0, anchor=0) -> final A must be 0
        #   t=2: unique search anchor per traj (mask=1) -> |G_S|==1 -> A^S must stay 0
        #   t=3: obs  (mask=0, anchor=999 corrupted) -> mask=0 overrides to final A=0
        #   t=4: shared item anchor 555 (mask=1) -> |G_S|==4, [1,1,0,0] -> non-zero A^S
        #   t=5: shared terminal buy on same anchor 555 in trajs 0,2 only -> still weighted properly
        h_shared = 555
        mask = torch.tensor(
            [
                [1, 0, 1, 0, 1, 1],
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1],
                [1, 0, 1, 0, 1, 0],
            ],
            dtype=torch.int64,
        )
        anchors = torch.tensor(
            [
                [0, 0, 101, 999, h_shared, h_shared],
                [0, 0, 102, 999, h_shared, 0],
                [0, 0, 103, 999, h_shared, h_shared],
                [0, 0, 104, 999, h_shared, 0],
            ],
            dtype=torch.int64,
        )
        rewards = torch.zeros((4, 6), dtype=torch.float32)
        rewards[0, 5] = 1.0
        rewards[1, 4] = 1.0
        uids = np.array(["u1", "u1", "u1", "u1"], dtype=object)

        grp = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        mu = 0.5
        sigma = float(torch.std(grp).item())
        eps = 1e-6
        ae_pos = (1.0 - mu) / (sigma + eps)
        ae_neg = (0.0 - mu) / (sigma + eps)
        omega = 0.5

        adv, ret = compute_gigpo_advantage(
            rewards,
            mask,
            uids,
            anchor_state_ids=anchors,
            epsilon=eps,
            norm_adv_by_std_in_grpo=True,
            omega=omega,
        )
        self.assertTrue(torch.allclose(adv, ret))

        # Check home (t=0) & unique search (t=2) get pure A^E
        for t in (0, 2):
            self.assertAlmostEqual(adv[0, t].item(), ae_pos, places=5)
            self.assertAlmostEqual(adv[1, t].item(), ae_pos, places=5)
            self.assertAlmostEqual(adv[2, t].item(), ae_neg, places=5)
            self.assertAlmostEqual(adv[3, t].item(), ae_neg, places=5)

        # Check obs / masked tokens are identically zero (even with non-zero anchor at t=3)
        for row in range(4):
            self.assertEqual(adv[row, 1].item(), 0.0)
            self.assertEqual(adv[row, 3].item(), 0.0)
        self.assertEqual(adv[1, 5].item(), 0.0)
        self.assertEqual(adv[3, 5].item(), 0.0)

        # Check shared anchor at t=4 (and t=5 where mask=1) gets A^E + omega * A^S = (1 + omega) * A^E here
        expected_pos = ae_pos * (1.0 + omega)
        expected_neg = ae_neg * (1.0 + omega)
        self.assertAlmostEqual(adv[0, 4].item(), expected_pos, places=5)
        self.assertAlmostEqual(adv[0, 5].item(), expected_pos, places=5)
        self.assertAlmostEqual(adv[1, 4].item(), expected_pos, places=5)
        self.assertAlmostEqual(adv[2, 4].item(), expected_neg, places=5)
        self.assertAlmostEqual(adv[2, 5].item(), expected_neg, places=5)
        self.assertAlmostEqual(adv[3, 4].item(), expected_neg, places=5)

        # Check omega=0 falls back exactly to pure GRPO A^E
        adv_zero, _ = compute_gigpo_advantage(
            rewards,
            mask,
            uids,
            anchor_state_ids=anchors,
            epsilon=eps,
            norm_adv_by_std_in_grpo=True,
            omega=0.0,
        )
        for row, expected in ((0, ae_pos), (1, ae_pos), (2, ae_neg), (3, ae_neg)):
            self.assertAlmostEqual(adv_zero[row, 4].item(), expected, places=5)

    def test_trajectory_deduplication_resists_repeat_loop(self):
        # 4 trajectories in group u1 with rewards [1.0, 0.0, 0.0, 1.0].
        # Trajectory 1 visits anchor 777 ONCE; Trajectory 1' visits 777 FIVE times in a loop.
        # Deduped G_S for anchor 777 is {0, 1, 2} -> rewards [1.0, 0.0, 0.0], NOT inflated by 5 copies of 0.0.
        h = 777
        mask_single = torch.ones((4, 5), dtype=torch.int64)
        anchors_single = torch.tensor(
            [
                [h, 0, 0, 0, 0],
                [h, 0, 0, 0, 0],
                [h, 0, 0, 0, 0],
                [888, 0, 0, 0, 0],
            ],
            dtype=torch.int64,
        )
        anchors_loop = torch.tensor(
            [
                [h, 0, 0, 0, 0],
                [h, h, h, h, h],
                [h, 0, 0, 0, 0],
                [888, 0, 0, 0, 0],
            ],
            dtype=torch.int64,
        )
        rewards = torch.zeros((4, 5), dtype=torch.float32)
        rewards[0, 4] = 1.0
        rewards[3, 4] = 1.0
        uids = np.array(["u1", "u1", "u1", "u1"], dtype=object)

        adv_single, _ = compute_gigpo_advantage(
            rewards, mask_single, uids, anchor_state_ids=anchors_single, omega=0.5
        )
        adv_loop, _ = compute_gigpo_advantage(
            rewards, mask_single, uids, anchor_state_ids=anchors_loop, omega=0.5
        )

        # Trajectory 0's anchor token and Trajectory 1's anchor tokens must receive identical A(i,t)
        self.assertAlmostEqual(adv_single[0, 0].item(), adv_loop[0, 0].item(), places=6)
        self.assertAlmostEqual(adv_single[2, 0].item(), adv_loop[2, 0].item(), places=6)
        for t in range(5):
            self.assertAlmostEqual(adv_single[1, 0].item(), adv_loop[1, t].item(), places=6)

        # Verify exact numerical values from deduped G_S=[1.0, 0.0, 0.0]
        grp_e = torch.tensor([1.0, 0.0, 0.0, 1.0])
        ae0 = (1.0 - 0.5) / (float(torch.std(grp_e).item()) + 1e-6)
        grp_s = torch.tensor([1.0, 0.0, 0.0])
        as0 = (1.0 - float(torch.mean(grp_s).item())) / (float(torch.std(grp_s).item()) + 1e-6)
        self.assertAlmostEqual(adv_loop[0, 0].item(), ae0 + 0.5 * as0, places=5)

    def test_zero_variance_anchor_group_and_singleton_suppress_step_advantage(self):
        # Prompt group u1 has mixed scores [1, 1, 0, 0] -> non-zero A^E.
        # Anchor 301 is visited ONLY by winners (traj 0 & 1 -> scores [1, 1], sigma=0) -> A^S == 0!
        # Anchor 302 is visited ONLY by traj 2 (|G_S|=1) -> A^S == 0!
        # Anchor 303 is visited by traj 1 (1.0) and traj 2 (0.0) -> A^S != 0!
        mask = torch.ones((4, 3), dtype=torch.int64)
        anchors = torch.tensor(
            [
                [301, 0, 0],
                [301, 303, 0],
                [302, 303, 0],
                [0, 0, 0],
            ],
            dtype=torch.int64,
        )
        rewards = torch.zeros((4, 3), dtype=torch.float32)
        rewards[0, 2] = 1.0
        rewards[1, 2] = 1.0
        uids = np.array(["u1", "u1", "u1", "u1"], dtype=object)

        adv, _ = compute_gigpo_advantage(
            rewards, mask, uids, anchor_state_ids=anchors, omega=0.5
        )
        grp = torch.tensor([1.0, 1.0, 0.0, 0.0])
        sigma_e = float(torch.std(grp).item())
        ae_pos = (1.0 - 0.5) / (sigma_e + 1e-6)
        ae_neg = (0.0 - 0.5) / (sigma_e + 1e-6)

        # Anchor 301 has sigma=0 -> stays equal to pure A^E
        self.assertAlmostEqual(adv[0, 0].item(), ae_pos, places=5)
        self.assertAlmostEqual(adv[1, 0].item(), ae_pos, places=5)
        # Anchor 302 has |G_S|=1 -> stays equal to pure A^E
        self.assertAlmostEqual(adv[2, 0].item(), ae_neg, places=5)
        # Anchor 303 has G_S=[1.0, 0.0] -> receives non-zero A^S
        sigma_303 = math.sqrt(0.5)
        as_pos = (1.0 - 0.5) / (sigma_303 + 1e-6)
        as_neg = (0.0 - 0.5) / (sigma_303 + 1e-6)
        self.assertAlmostEqual(adv[1, 1].item(), ae_pos + 0.5 * as_pos, places=5)
        self.assertAlmostEqual(adv[2, 1].item(), ae_neg + 0.5 * as_neg, places=5)

    def test_unnormalized_norm_adv_false(self):
        # Cross-uid isolation: u1 vs u2 must not mix anchor sets even if both use anchor 555
        mask = torch.ones((4, 2), dtype=torch.int64)
        anchors = torch.tensor(
            [
                [555, 0],
                [555, 0],
                [555, 0],
                [555, 0],
            ],
            dtype=torch.int64,
        )
        rewards = torch.zeros((4, 2), dtype=torch.float32)
        rewards[0, 1] = 1.0
        rewards[2, 1] = 1.0
        rewards[3, 1] = 1.0
        uids = np.array(["u1", "u1", "u2", "u2"], dtype=object)

        adv, _ = compute_gigpo_advantage(
            rewards,
            mask,
            uids,
            anchor_state_ids=anchors,
            norm_adv_by_std_in_grpo=False,
            omega=0.5,
        )
        # u1: [1, 0], mu=0.5 -> A^E=[+0.5, -0.5], A^S=[+0.5, -0.5] -> A(t=0)=[+0.75, -0.75]
        self.assertAlmostEqual(adv[0, 0].item(), 0.75, places=5)
        self.assertAlmostEqual(adv[1, 0].item(), -0.75, places=5)
        # u2: [1, 1], sigma=0 -> A=0
        self.assertAlmostEqual(adv[2, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(adv[3, 0].item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

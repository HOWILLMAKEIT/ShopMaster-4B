"""无折扣轨迹去重锚点 GiGPO 优势估计器与公开状态锚点编码。

实现 `A(i, t) = A^E(tau_i) + omega * A^S(i, t)`：
- 全部锚点仅由环境公开 `observation_state` 生成，不读取任何任务答案字段；
- `gamma = 1.0` 无折扣，避免直接购买比“核验子页后再买”获得更高的折现回报；
- 同一轨迹内对相同锚点去重，防止 repeat_loop 反复访问把 `G_S(u, h)` 均值/方差带偏；
- 跳过 `search_home`（避免 Turn 0 被 `(1 + omega)` 倍加权），并仅在 `|G_S| >= 2` 且 `sigma > 1e-6`
  时施加步级信用。
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import logging
import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_DEFAULT_GIGPO_OMEGA = 0.5


def _noop_register_adv_est(_name: str):
    def _decorator(fn):
        return fn

    return _decorator


try:
    from verl.trainer.ppo.core_algos import (
        ADV_ESTIMATOR_REGISTRY as _ADV_REGISTRY,
        register_adv_est as _verl_register_adv_est,
    )
except ImportError:  # 允许无 verl 的本地单元测试导入纯 Python 锚点工具
    _ADV_REGISTRY = None
    register_adv_est = _noop_register_adv_est
else:
    if "gigpo" in _ADV_REGISTRY:
        register_adv_est = _noop_register_adv_est
    else:
        register_adv_est = _verl_register_adv_est


def _normalize_options(raw_options: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw_options, Mapping):
        return ()
    return tuple(
        sorted((str(k).strip(), str(v).strip()) for k, v in raw_options.items())
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def canonical_anchor_key(observation_state: Optional[Mapping[str, Any]]) -> Optional[tuple]:
    """将环境公开 `observation_state` 映射为无状态混叠的规范化锚点键。

    跳过 `search_home` / `terminal` / `unknown`（返回 `None`），避免把 100% 汇合的
    Turn 0 首页无意放大 `(1 + omega)` 倍。
    """
    if not isinstance(observation_state, Mapping):
        return None
    page_type = str(observation_state.get("page_type") or "").strip()
    if page_type in ("", "search_home", "terminal", "unknown"):
        return None

    if page_type == "search_results":
        norm_query = str(
            observation_state.get("normalized_query")
            or observation_state.get("query")
            or ""
        ).strip()
        if not norm_query:
            return None
        try:
            page = int(observation_state.get("page") or 1)
        except (TypeError, ValueError):
            page = 1
        return ("search", norm_query, page)

    if page_type == "product_detail":
        product = observation_state.get("product")
        asin = str(product.get("asin") or "").strip() if isinstance(product, Mapping) else ""
        if not asin:
            return None
        opts = _normalize_options(observation_state.get("selected_options"))
        return ("item", asin, opts)

    if page_type == "information_subpage":
        product = observation_state.get("product")
        asin = str(product.get("asin") or "").strip() if isinstance(product, Mapping) else ""
        if not asin:
            return None
        subpage = str(observation_state.get("subpage") or "information").strip().casefold()
        opts = _normalize_options(observation_state.get("selected_options"))
        return ("subpage", asin, subpage, opts)

    return None


def compute_anchor_state_id(observation_state: Optional[Mapping[str, Any]]) -> int:
    """将公开状态规范化锚点哈希为正的非零 63-bit 整数；跳过或空状态返回 `0`。"""
    key = canonical_anchor_key(observation_state)
    if key is None:
        return 0
    payload = json.dumps(
        _to_jsonable(key), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    anchor_id = int.from_bytes(digest, "big") & ((1 << 63) - 1)
    return anchor_id if anchor_id != 0 else 1


def resolve_gigpo_omega(omega: Optional[float] = None, config: Optional[Any] = None) -> float:
    """按显式参数 -> config 属性 -> `SHOP_RL_GIGPO_OMEGA` -> 默认 0.5 解析权重。"""
    if omega is not None:
        return float(omega)
    if config is not None:
        cfg_val = (
            config.get("gigpo_omega", None)
            if isinstance(config, Mapping)
            else getattr(config, "gigpo_omega", None)
        )
        if cfg_val is not None:
            return float(cfg_val)
    raw_env = os.getenv("SHOP_RL_GIGPO_OMEGA", "").strip()
    if raw_env:
        return float(raw_env)
    return _DEFAULT_GIGPO_OMEGA


def _resolve_norm_adv_by_std(
    norm_adv_by_std_in_grpo: Optional[bool], config: Optional[Any]
) -> bool:
    if norm_adv_by_std_in_grpo is not None:
        return bool(norm_adv_by_std_in_grpo)
    if config is not None:
        val = (
            config.get("norm_adv_by_std_in_grpo", True)
            if isinstance(config, Mapping)
            else getattr(config, "norm_adv_by_std_in_grpo", True)
        )
        return True if val is None else bool(val)
    return True


def _normalize_uid(raw_uid: Any) -> Any:
    if hasattr(raw_uid, "item"):
        try:
            return raw_uid.item()
        except Exception:
            pass
    return raw_uid


@register_adv_est("gigpo")
def compute_gigpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | Sequence[Any],
    anchor_state_ids: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: Optional[bool] = None,
    omega: Optional[float] = None,
    config: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算无折扣、按轨迹去重的 GiGPO 混合优势 `(advantages, returns)`。

    Args:
        token_level_rewards: `(bsz, resp_len)` 终局奖励张量（最后有效位置或逐 token 之和为 `R^(i)`）。
        response_mask: `(bsz, resp_len)` 仅 assistant 生成 token 为 1。
        index: 长度为 `bsz` 的 prompt `uid` 分组数组。
        anchor_state_ids: 可选 `(bsz, resp_len)` `int64` 锚点张量，`0` 代表跳过（观测/首页/padding）。
        epsilon: 防除零平滑项（默认 `1e-6`）。
        norm_adv_by_std_in_grpo: 是否除以组标准差；`None` 时从 `config` 读取。
        omega: 步级优势权重；`None` 时读取 `SHOP_RL_GIGPO_OMEGA`（默认 `0.5`）。
        config: `verl` `AlgoConfig` 对象（可选）。

    Returns:
        `(advantages, returns)`，形状均为 `(bsz, resp_len)`，`float32`。
    """
    scores = token_level_rewards.sum(dim=-1).to(dtype=torch.float32)
    mask_f = response_mask.to(dtype=torch.float32)
    resolved_omega = resolve_gigpo_omega(omega=omega, config=config)
    use_std = _resolve_norm_adv_by_std(norm_adv_by_std_in_grpo, config)

    with torch.no_grad():
        bsz = scores.shape[0]
        if len(index) != bsz:
            raise ValueError(f"index 长度 ({len(index)}) 与 batch size ({bsz}) 不一致")

        uids = [_normalize_uid(index[i]) for i in range(bsz)]
        scores_cpu = scores.detach().cpu().tolist()

        # 1) 轨迹级 GRPO 优势 A^E（与 verl core_algos.compute_grpo_outcome_advantage 完全一致）
        uid_to_scores: dict[Any, list[float]] = defaultdict(list)
        for i in range(bsz):
            uid_to_scores[uids[i]].append(float(scores_cpu[i]))

        uid_mean: dict[Any, float] = {}
        uid_std: dict[Any, float] = {}
        for uid, grp in uid_to_scores.items():
            if len(grp) == 1:
                uid_mean[uid] = 0.0
                uid_std[uid] = 1.0
            else:
                grp_t = torch.tensor(grp, dtype=torch.float32)
                uid_mean[uid] = float(torch.mean(grp_t).item())
                uid_std[uid] = float(torch.std(grp_t).item())

        ep_adv_cpu = [0.0] * bsz
        for i in range(bsz):
            u = uids[i]
            if use_std:
                ep_adv_cpu[i] = (float(scores_cpu[i]) - uid_mean[u]) / (uid_std[u] + epsilon)
            else:
                ep_adv_cpu[i] = float(scores_cpu[i]) - uid_mean[u]

        ep_adv = torch.tensor(ep_adv_cpu, dtype=torch.float32, device=token_level_rewards.device)
        step_adv = torch.zeros_like(token_level_rewards, dtype=torch.float32)

        active_anchor_count = 0
        group_sizes: list[int] = []
        nonzero_as_vals: list[float] = []

        # 2) 步级去重锚点优势 A^S(i, t)（仅在 omega != 0 且传入 anchor_state_ids 时计算）
        if anchor_state_ids is not None and resolved_omega != 0.0:
            if anchor_state_ids.shape != response_mask.shape:
                raise ValueError(
                    f"anchor_state_ids 形状 {tuple(anchor_state_ids.shape)} "
                    f"与 response_mask {tuple(response_mask.shape)} 不一致"
                )

            anchor_np = anchor_state_ids.detach().cpu().numpy()
            mask_np = response_mask.detach().cpu().numpy() > 0

            # 每条轨迹 i 去重采集其有效 assistant token 上的非零锚点
            uid_anchor_trajs: dict[tuple[Any, int], list[int]] = defaultdict(list)
            row_unique_anchors: list[list[int]] = []
            for i in range(bsz):
                valid_ids = anchor_np[i][mask_np[i]]
                nonzero_ids = valid_ids[valid_ids > 0]
                uniq = np.unique(nonzero_ids).tolist() if nonzero_ids.size > 0 else []
                row_unique_anchors.append(uniq)
                u = uids[i]
                for aid in uniq:
                    uid_anchor_trajs[(u, int(aid))].append(i)

            # 对每个 (uid, anchor_id) 计算 |G_S| >= 2 且 sigma > 1e-6 的步级优势
            traj_anchor_adv: dict[tuple[int, int], float] = {}
            for (_u, aid), traj_indices in uid_anchor_trajs.items():
                g_size = len(traj_indices)
                if g_size < 2:
                    continue
                grp_scores = torch.tensor(
                    [float(scores_cpu[idx]) for idx in traj_indices],
                    dtype=torch.float32,
                )
                sigma = float(torch.std(grp_scores).item())
                if sigma <= 1e-6:
                    continue
                mu = float(torch.mean(grp_scores).item())
                active_anchor_count += 1
                group_sizes.append(g_size)
                denom = (sigma + epsilon) if use_std else 1.0
                for idx in traj_indices:
                    val = (float(scores_cpu[idx]) - mu) / denom
                    traj_anchor_adv[(idx, aid)] = val
                    nonzero_as_vals.append(abs(val))

            # 将已激活的锚点优势回填到 step_adv_np（再单次拷贝回 GPU/目标设备）
            if traj_anchor_adv:
                step_adv_np = np.zeros(anchor_np.shape, dtype=np.float32)
                for i in range(bsz):
                    for aid in row_unique_anchors[i]:
                        val = traj_anchor_adv.get((i, aid))
                        if val is not None and val != 0.0:
                            step_adv_np[i][anchor_np[i] == aid] = val
                step_adv = torch.from_numpy(step_adv_np).to(
                    device=token_level_rewards.device, dtype=torch.float32
                )

        advantages = (ep_adv.unsqueeze(-1) + float(resolved_omega) * step_adv) * mask_f

        total_mask_tokens = float(mask_f.sum().item())
        covered_frac = (
            float(((step_adv != 0.0) & (mask_f > 0.0)).float().sum().item()) / total_mask_tokens
            if total_mask_tokens > 0.0
            else 0.0
        )
        mean_abs_ae = (
            float(sum(abs(v) for v in ep_adv_cpu) / len(ep_adv_cpu)) if ep_adv_cpu else 0.0
        )
        mean_abs_as = (
            float(sum(nonzero_as_vals) / len(nonzero_as_vals)) if nonzero_as_vals else 0.0
        )
        mean_group_size = float(sum(group_sizes) / len(group_sizes)) if group_sizes else 0.0
        summary_msg = (
            f"[GiGPO] bsz={bsz} groups={len(uid_to_scores)} active_anchors={active_anchor_count} "
            f"mean_group_size={mean_group_size:.2f} covered_token_frac={covered_frac:.3f} "
            f"mean_abs_AE={mean_abs_ae:.4f} mean_abs_AS={mean_abs_as:.4f} omega={resolved_omega:.3f}"
        )
        print(summary_msg, flush=True)
        logger.debug("%s", summary_msg)

    return advantages, advantages

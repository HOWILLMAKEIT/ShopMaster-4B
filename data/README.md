# data

官方任务数据与由它派生的 task_id 清单。环境商品全集在 `env/shop_env/data/`，不在此目录。

| 文件 | 内容 | 行数 | SHA256 |
| --- | --- | ---: | --- |
| `fine_items_train_standard.jsonl` | 官方训练任务（standard，不带用户背景信息），每行一个商品条目含一个 `instructions` | 21,962 | `16001864a43c392d58116bf280c6784511997bc422b8c43ddf59a4e08c514587` |
| `fine_items_eval_standard.jsonl` | 官方评测任务（standard），同上结构 | 1,459 | `ef73aa40386d0d3c722704769b677276abbcb26cf2196adb165581e9a26ed63a` |
| `tasks/train_standard_task_ids.jsonl` | 官方训练任务的 task_id 清单 | 21,962 | `1e91f10e89a9b7b2ef90ce3ea789ab6675698612bc2297e32985a7c4007d3be6` |
| `tasks/eval_standard_task_ids.jsonl` | 官方评测任务的 task_id 清单 | 1,459 | `d4190059db0bcccdee5553e74d1697d05a4216be45e9604ef1bf1ed6deaa53a4` |
| `tasks/sft_candidate_tasks.jsonl` | 从官方训练池分层的 SFT 候选集 | 6,000 | `df8c4366b73710746308b8ce6976857712a19abee05163d54f0265b8163ca36a` |
| `tasks/eval_dev_474.jsonl` | 清洗后的开发集（原 `eval_dev_500` 剔除 26 道价格不可核验题） | 474 | `4df6732d8dd1618bd6d7a1d681c5c6785a6b0b5d237be6acedddcda1b9e70df6` |
| `tasks/eval_holdout_915.jsonl` | 清洗后的冻结最终留出集（原 959 题剔除 44 道价格不可核验题） | 915 | `b3fe9e538e4d4c277e47f86194ea2980505aa72afc28cff9beb6bfbeb018bcaa` |
| `tasks/grpo_tasks.jsonl` | 清洗后的 RL 训练集（原 15,951 题剔除 401 道价格不可核验题） | 15,550 | `7f1c8d766e781e65c1963726893fb0b80945e5cc0a6df2f413b93461fbaf5b24` |

## 来源与清洗规则

- 官方原始任务下载自 [wpei/ShopSimulator](https://huggingface.co/datasets/wpei/ShopSimulator)（commit `ab34b289`，下载于 2026-11-13），`align_task_ids.py` 按商品全集 `items_eval_train.json` 顺序展开对齐为 eval 0–1458（1,459 题）与 train 1459–23420（21,962 题）。
- `eval_dev_474.jsonl` 来自原 `seed=42` 抽取的 500 题开发集，经 `variant_price.resolve_variant_price` 与 `explicit_budget_from_instruction` 静态核验后剔除 26 道不可核验题（21 道 `multiple_effective_price_axes` + 5 道 `effective_price_axis_unselected`），得到 474 题开发集。
- `eval_holdout_915.jsonl` 由其余 959 道官方评测题剔除 44 道价格不可核验题（35 道多有效价格轴 + 9 道未指定价格轴）得到，保持零开发污染并在最终评测前全程冻结。
- `grpo_tasks.jsonl` 由原 15,951 道 RL 候选训练题剔除 401 道价格不可核验题（299 道多有效价格轴 + 102 道未指定价格轴）得到。

## 使用约束

- 训练（SFT 数据构造、RL 采样）只使用 `task_id >= 1459` 的任务；`task_id <= 1458` 保留作评测。
- `grpo_tasks.jsonl` 与 `eval_dev_474.jsonl`、`eval_holdout_915.jsonl`、`sft_candidate_tasks.jsonl` 两两零交集；`data/sft/success_tasks.jsonl`（1,015 题）经静态核验包含 0 道价格不可核验题。
- 不修改本目录的官方原始文件；派生 Parquet 由 `shop_rl rl prepare-tasks` 生成至 `data/rl/`。

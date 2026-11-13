# data

官方任务数据与由它派生的 task_id 清单。环境商品全集在 `env/shop_env/data/`，不在此目录。

| 文件 | 内容 | 行数 | SHA256 |
| --- | --- | ---: | --- |
| `fine_items_train_standard.jsonl` | 官方训练任务（standard，不带用户背景信息），每行一个商品条目含一个 `instructions` | 21,962 | `16001864a43c392d58116bf280c6784511997bc422b8c43ddf59a4e08c514587` |
| `fine_items_eval_standard.jsonl` | 官方评测任务（standard），同上结构 | 1,459 | `ef73aa40386d0d3c722704769b677276abbcb26cf2196adb165581e9a26ed63a` |
| `tasks/train_standard_task_ids.jsonl` | 训练任务的 task_id 清单 | 21,962 | `1e91f10e89a9b7b2ef90ce3ea789ab6675698612bc2297e32985a7c4007d3be6` |
| `tasks/eval_standard_task_ids.jsonl` | 评测任务的 task_id 清单 | 1,459 | `d4190059db0bcccdee5553e74d1697d05a4216be45e9604ef1bf1ed6deaa53a4` |

## 来源

- 前两个文件下载自 [wpei/ShopSimulator](https://huggingface.co/datasets/wpei/ShopSimulator)（官方 HF 数据集，commit `ab34b289`，下载于 2026-11-13）。
- task_id 清单由 `scripts/align_task_ids.py` 生成，重跑即可复现。

## task_id 对应关系

环境 `get_existed_goals` 按商品全集 `env/shop_env/data/items_eval_train.json`（23,421 条）的顺序展开 goals，本项目数据每条恰好一个 instruction 且 attributes 非空，因此第 i 条商品对应 task_id=i：

- 官方评测任务 = task_id 0–1458（1,459 题）
- 官方训练任务 = task_id 1459–23420（21,962 题）

此对应由脚本对 `get_existed_goals` 的静态模拟得出（推断）；尚未在运行中的环境用 `reset(task_id)` 核对返回的 instruction 与本目录文件一致，核对列入 v0-001。

## 使用约束

- 训练（SFT 数据构造、RL 采样）只使用 task_id ≥ 1459 的任务；task_id ≤ 1458 保留作评测。
- 不修改本目录的官方文件；派生数据放独立目录并记录生成方式。

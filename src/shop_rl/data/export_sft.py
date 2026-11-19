#!/usr/bin/env python
"""导出 v1 SFT 训练数据（策略 A 带思考，策略 B 无思考）。

输入：data/sft/success_tasks.jsonl 与其指向的轨迹文件（gong-4080 项目根运行）。
处理：官方 chat_template 渲染整条对话并 tokenize；按带思考长度分层抽样
（L1 ≤6K / L2 6–8.2K 各 125 条，L3 8.2–12K / L4 12–16.4K 全取，seed 42）得
388 条训练集；剩余池分层抽约 40 条验证集。labels 与分桶 mask（思考段 /
行动段 = content + tool_call + im_end）按每轮 assistant 的渲染前缀定位：
生成提示止于 <think>，思考正文到 </think> 前，之后为行动段。最后一条
assistant 之后的尾部观察不产生学习信号，导出时丢弃（分层口径仍按含
尾部观察的全长，与长度审计一致）。
输出：data/sft/v1/sft_{A,B}.pkl 与 data/sft/v1/manifest_{A,B}.json。
运行：HF_ENDPOINT=https://hf-mirror.com .venv/bin/python shop_rl.data.export_sft [--strategy B]
"""
import argparse
import json
import os
import pickle
import random
import statistics

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(key, None)

from transformers import AutoTokenizer

# (层名, 上界 token, 抽取配额)；配额 None 表示全取。分层口径 = 含尾部观察的全长。
BANDS = [("L1", 6000, 125), ("L2", 8192, 125), ("L3", 12000, None), ("L4", 16384, None)]
VAL_TOTAL = 40


def build_messages(trajectory, drop_trailing):
    msgs = [dict(m) for m in trajectory["conversation"]]
    if drop_trailing:
        while msgs and msgs[-1]["role"] != "assistant":
            msgs.pop()
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                fn = dict(call["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                calls.append({**call, "function": fn})
            m["tool_calls"] = calls
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strategy", choices=["A", "B"], default="A",
                    help="A=带思考（reasoning_content 进样本）；B=无思考（空标记，正文+工具调用进 loss）")
    ap.add_argument("--out-dir", default="data/sft/v1")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    strategy = args.strategy

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    close_think_id = tok.convert_tokens_to_ids("</think>")
    assert close_think_id is not None and close_think_id >= 0

    def render(msgs, tools, gen_prompt, no_think):
        text = tok.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=gen_prompt,
            **({"enable_thinking": False} if no_think else {}))
        return tok(text, add_special_tokens=False)["input_ids"]

    # 1) 全量渲染计长（含尾部观察，与长度审计同口径）。分层口径固定为
    #    带思考长度（no_think=False），保证 A/B 抽到完全相同的任务集合。
    pool = []  # (task_id, length, trajectory)
    with open("data/sft/success_tasks.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            with open(rec["file"]) as tf:
                traj = json.load(tf)
            length = len(render(build_messages(traj, drop_trailing=False), traj["tools"], False, False))
            pool.append((traj["task_id"], length, traj))
    pool.sort(key=lambda x: x[0])

    in_pool = [p for p in pool if p[1] <= 16384]
    assert len(in_pool) == 988, f"≤16,384 应为 988，实际 {len(in_pool)}"

    def band_of(length):
        for name, upper, _ in BANDS:
            if length <= upper:
                return name
        return None

    # 2) 分层抽样：训练集 + 验证集
    train_ids, val_ids = [], []
    for name, upper, quota in BANDS:
        items = [p for p in in_pool if band_of(p[1]) == name]
        picked = rng.sample(items, quota) if quota else items
        picked_ids = {t for t, _, _ in picked}
        train_ids += [t for t, _, _ in picked]
        rest = [t for t, _, _ in items if t not in picked_ids]
        n_val = max(1, round(VAL_TOTAL * len(rest) / 600))
        val_ids += rng.sample(rest, min(n_val, len(rest)))
    train_ids.sort(), val_ids.sort()
    id_set = {"train": set(train_ids), "val": set(val_ids)}

    # 3) 逐样本生成 ids / labels / 分桶 mask
    # 边界全部用原子特殊 token（<|im_start|>/<think>/</think>/<|im_end|>）在完整
    # 渲染序列里定位，不做分段 tokenize（BPE 跨边界合并会产生对不齐）。
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    think_open_id = tok.convert_tokens_to_ids("<think>")

    def is_nl(tid):
        return tok.convert_ids_to_tokens(tid) in ("\n", "Ċ")

    anomalies = []
    samples = {"train": [], "val": []}
    for task_id, length, traj in pool:
        split = next((s for s, ids in id_set.items() if task_id in ids), None)
        if split is None:
            continue
        msgs = build_messages(traj, drop_trailing=True)
        if strategy == "B":  # 无思考渲染：去掉 reasoning_content，模板插空思考标记
            for m in msgs:
                if m["role"] == "assistant":
                    m.pop("reasoning_content", None)
        full_ids = render(msgs, traj["tools"], False, strategy == "B")
        labels = [-100] * len(full_ids)
        think_mask = [False] * len(full_ids)
        act_mask = [False] * len(full_ids)
        starts = [j for j, t in enumerate(full_ids) if t == im_start_id]
        if len(starts) != len(msgs):  # 模板合并了连续 tool 消息等，定位失准
            anomalies.append({"task_id": task_id, "turn": -1, "kind": "msg-align"})
            continue
        for k, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            p = starts[k]
            nxt = starts[k + 1] if k + 1 < len(starts) else len(full_ids)
            e = nxt - 1
            while e > p and full_ids[e] != im_end_id:
                e -= 1
            if think_open_id in full_ids[p : p + 8]:
                q = p + full_ids[p : p + 8].index(think_open_id) + 1
                if is_nl(full_ids[q]):
                    q += 1
            else:  # 无思考段（罕见）：整段归行动段
                q = p + 2
                anomalies.append({"task_id": task_id, "turn": k, "kind": "no-think"})
            window = full_ids[q : e + 1]
            if close_think_id in window:
                p_close = q + window.index(close_think_id)
                if strategy == "B":  # 空思考标记整体 mask，loss 从其后正文开始
                    cut = p_close + 1
                    while cut <= e and is_nl(full_ids[cut]):
                        cut += 1
                    q = cut  # 思考桶为空；行动桶从正文起
                else:
                    cut = p_close - 1 if p_close > q and is_nl(full_ids[p_close - 1]) else p_close
            else:
                cut = q
            for j in range(q, e + 1):  # e 为 <|im_end|>，参与 loss
                labels[j] = full_ids[j]
                think_mask[j] = j < cut
                act_mask[j] = j >= cut
        samples[split].append({
            "task_id": task_id,
            "band": band_of(length),
            "audit_length": length,
            "input_ids": full_ids,
            "labels": labels,
            "think_mask": think_mask,
            "act_mask": act_mask,
        })

    # 4) 抽样断言：A 思考段在前、行动段以 </think> 开头；B 无思考、正文直接开始
    for smp in rng.sample(samples["train"], 5):
        ids, tk, ac = smp["input_ids"], smp["think_mask"], smp["act_mask"]
        first_a = ac.index(True)
        head = tok.decode(ids[first_a : first_a + 6])
        if strategy == "A":
            assert tk.index(True) < first_a, f"task {smp['task_id']} 分桶顺序异常"
            assert "</think>" in head, f"task {smp['task_id']} 行动段应以 </think> 开头，实际：{head!r}"
        else:
            assert not any(tk), f"task {smp['task_id']} 策略 B 不应有思考桶"
            assert "</think>" not in head, f"task {smp['task_id']} 策略 B 行动段应以正文开头，实际：{head!r}"

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"sft_{strategy}.pkl"), "wb") as f:
        pickle.dump(samples, f, protocol=4)

    def stat(split):
        xs = samples[split]
        loss_tok = [sum(s["think_mask"]) for s in xs]
        act_tok = [sum(s["act_mask"]) for s in xs]
        return {
            "n": len(xs),
            "bands": {b: sum(1 for s in xs if s["band"] == b) for b, _, _ in BANDS},
            "seq_tokens": {"total": sum(len(s["input_ids"]) for s in xs),
                           "min": min(len(s["input_ids"]) for s in xs),
                           "median": statistics.median(len(s["input_ids"]) for s in xs),
                           "max": max(len(s["input_ids"]) for s in xs)},
            "loss_tokens": {"think_total": sum(loss_tok), "think_mean": round(statistics.mean(loss_tok)),
                            "act_total": sum(act_tok), "act_mean": round(statistics.mean(act_tok))},
        }

    manifest = {
        "seed": args.seed, "strategy": strategy, "bands": {b: {"upper": u, "quota": q} for b, u, q in BANDS},
        "train": stat("train"), "val": stat("val"), "anomalies": anomalies[:20], "n_anomalies": len(anomalies),
    }
    with open(os.path.join(args.out_dir, f"manifest_{strategy}.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

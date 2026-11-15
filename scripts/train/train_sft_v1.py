#!/usr/bin/env python
"""v1 LoRA SFT 训练脚本（策略 A/B 数据，两卡 4090）。

用法（gong-4090 项目根 /mnt/data1/qiuwenjie/shop-rl）：
  torchrun --nproc_per_node=2 scripts/train/train_sft_v1.py --data data/sft/sft_A.pkl \
      --model-dir models/Qwen3.5-4B --out outputs/v1/sft-a
  v1-001 吞吐/显存冒烟：加 --limit-longest 12 --max-updates 6 --epochs 1

省显存提吞吐的三个手段（相对"固定 batch=1 + 全位置 logits"的朴素实现）：
1. 梯度检查点（use_reentrant=False）。
2. 按 token 预算组批：样本按长度排序后贪心装 ≤ --token-budget 的 micro-batch，
   短序列自动多条同批（GPU 利用率远高于单条 5k 序列），批顺序再随机。
3. lm_head 只算 loss 位置：序列中仅 assistant 段参与 loss（约 1/4），把
   hidden states 按 labels!=-100 选中后与输出头做矩阵乘，选出的位置再分块
   并用 checkpoint 重计算——避免 16k×248k 全 logits（约 8GB）。
同步方式：两卡各自持有半份 micro-batch，LoRA 梯度经 all_reduce 求和（base
参数全部冻结，梯度体量小，无需 DDP bucket）。

loss 与指标：每样本先在其 loss token 上平均（方法设计 loss 公式），再对
batch 内样本平均；同时记录思考段 / 行动段（content+tool_call+im_end）分桶
CE、lr、梯度范数、吞吐与显存峰值，逐条写 metrics.jsonl，训练结束汇总
metrics.json。验证集每 epoch 在 rank0 上评估。每 epoch 存 LoRA adapter。
"""
import argparse
import json
import math
import os
import pickle
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

CE_CHUNK = 512  # 分块交叉熵的行数（GPU1 与他人共享，压低瞬时显存）


# ---------- 数据与组批 ----------

def load_samples(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["train"], data["val"]


def make_batches(samples, token_budget, seed, epoch):
    """按长度排序后贪心装批，批顺序随机。返回 [idx,...] 列表的列表。"""
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    jitter = torch.rand(len(samples), generator=g).tolist()
    order = sorted(range(len(samples)), key=lambda i: (len(samples[i]["input_ids"]), jitter[i]))
    batches, cur, cur_tok = [], [], 0
    for i in order:
        n = len(samples[i]["input_ids"])
        if cur and cur_tok + n > token_budget:
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.append(i)
        cur_tok += n
    if cur:
        batches.append(cur)
    perm = torch.randperm(len(batches), generator=g).tolist()
    return [batches[p] for p in perm]


def collate(samples, idxs, pad_id=0):
    xs = [samples[i] for i in idxs]
    T = max(len(x["input_ids"]) for x in xs)
    B = len(xs)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    lab = torch.full((B, T), -100, dtype=torch.long)
    th = torch.zeros((B, T), dtype=torch.bool)
    ac = torch.zeros((B, T), dtype=torch.bool)
    for r, x in enumerate(xs):
        t = len(x["input_ids"])
        ids[r, :t] = torch.tensor(x["input_ids"])
        att[r, :t] = 1
        lab[r, :t] = torch.tensor(x["labels"])
        th[r, :t] = torch.tensor(x["think_mask"])
        ac[r, :t] = torch.tensor(x["act_mask"])
    return ids, att, lab, th, ac


# ---------- 分块交叉熵（只算 loss 位置，重计算省显存） ----------

def ce_rows(h_rows, w_head, y_rows):
    logits = (h_rows @ w_head.t()).float()
    return F.cross_entropy(logits, y_rows, reduction="none")


def gathered_ce(h, labels, think, act, w_head, lm_loss=True):
    """h: [B,T,H]（已含全部位置），labels/think/act: [B,T]。
    返回 (loss, n_rows, think_ce, act_ce)；后三者 detached。"""
    B = h.shape[0]
    tgt = labels[:, 1:]
    sel = tgt != -100
    hs = h[:, :-1][sel]
    ys = tgt[sel]
    bidx = sel.nonzero(as_tuple=False)[:, 0]
    ths = think[:, 1:][sel]
    acs = act[:, 1:][sel]
    ces = []
    for s in range(0, hs.shape[0], CE_CHUNK):
        e = min(s + CE_CHUNK, hs.shape[0])
        if lm_loss:
            ces.append(ckpt(ce_rows, hs[s:e], w_head, ys[s:e], use_reentrant=False))
        else:
            with torch.no_grad():
                ces.append(ce_rows(hs[s:e], w_head, ys[s:e]))
    ce = torch.cat(ces)                                   # [N] 带梯度
    counts = torch.bincount(bidx, minlength=B).float().clamp(min=1)
    sample_ce = torch.zeros(B, device=h.device, dtype=ce.dtype).index_add_(0, bidx, ce) / counts
    loss = sample_ce.mean()
    with torch.no_grad():
        think_ce = ce[ths].mean().item() if ths.any() else float("nan")
        act_ce = ce[acs].mean().item() if acs.any() else float("nan")
    return loss, ce.shape[0], think_ce, act_ce


# ---------- 模型 ----------

def build_model(model_dir, device):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    except ValueError:  # 多模态 config 可能不映射到 CausalLM，退回条件生成类
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    inner = getattr(model, "model", model)          # 文本主干（Qwen3_5TextModel，无视觉塔）
    # 实测结构：24 层 linear_attn（DeltaNet）+ 8 层 self_attn + 32 层 mlp，全取
    lora_target = [n for n, _ in model.named_modules()
                   if n.startswith("model.layers.") and n.endswith((
                       "q_proj", "k_proj", "v_proj", "o_proj",
                       "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj",
                       "gate_proj", "up_proj", "down_proj"))]
    assert lora_target, "未找到语言骨干线性层，请核对模块命名"
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=lora_target, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_tr:,} / {n_all:,} ({100 * n_tr / n_all:.2f}%)")
    base = model.base_model.model               # 去掉 PEFT 壳拿原模型
    return model, base, trainable


def grad_sync_and_clip(trainable, world_size, max_norm):
    if world_size > 1:
        for p in trainable:
            if p.grad is not None:
                g = p.grad.float()
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
                p.grad.copy_(g.to(p.grad.dtype))
    return torch.nn.utils.clip_grad_norm_(trainable, max_norm)


# ---------- 主流程 ----------

def evaluate(base, w_head, val, token_budget, device):
    inner = getattr(base, "model", base)   # 文本骨干，输出 last_hidden_state
    was_training = base.training
    base.eval()
    totals = {"loss": 0.0, "think": 0.0, "act": 0.0, "n": 0, "tok": 0}
    with torch.no_grad():
        for idxs in make_batches(val, token_budget, seed=0, epoch=-1):
            ids, att, lab, th, ac = collate(val, idxs)
            h = inner(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
            loss, n, tce, ace = gathered_ce(h, lab.to(device), th.to(device), ac.to(device),
                                            w_head.to(device), lm_loss=False)
            b = len(idxs)
            totals["loss"] += loss.item() * b
            totals["think"] += tce * b
            totals["act"] += ace * b
            totals["n"] += b
            totals["tok"] += n
    if was_training:
        base.train()
    n = max(totals["n"], 1)
    return {"val_loss": totals["loss"] / n, "val_think_ce": totals["think"] / n,
            "val_act_ce": totals["act"] / n, "val_samples": totals["n"],
            "val_loss_tokens": totals["tok"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft/sft_A.pkl")
    ap.add_argument("--model-dir", default="models/Qwen3.5-4B")
    ap.add_argument("--out", default="outputs/v1/sft-a")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--token-budget", type=int, default=14336)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-updates", type=int, default=None)
    ap.add_argument("--limit-longest", type=int, default=None,
                    help="v1-001 冒烟：只取最长的 K 条样本")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0

    train, val = load_samples(args.data)
    if args.limit_longest:
        train = sorted(train, key=lambda x: -len(x["input_ids"]))[: args.limit_longest]
        val = val[:8]

    model, base, trainable = build_model(args.model_dir, device)
    w_head = base.get_output_embeddings().weight.detach()   # [V,H]，冻结
    inner = getattr(base, "model", base)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.999), eps=1e-8)

    # 预排 micro-batch，两卡交错取批
    epoch_batches = [make_batches(train, args.token_budget, args.seed, e) for e in range(args.epochs)]
    n_local = min(len(batches) for batches in epoch_batches) // world
    total_updates = math.ceil(n_local / args.accum) * args.epochs
    if args.max_updates:
        total_updates = min(total_updates, args.max_updates)
    warmup = max(1, int(total_updates * args.warmup_ratio))

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    log_f = open(metrics_path, "a") if is_main else None

    def set_lr(step):
        if step <= warmup:
            lr = args.lr * step / warmup
        else:
            p = (step - warmup) / max(1, total_updates - warmup)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        return lr

    if is_main:
        print(json.dumps({"config": {**vars(args), "world": world,
                                     "n_train": len(train), "n_val": len(val),
                                     "total_updates": total_updates}}))

    step, micro_count = 0, 0
    accum_loss = accum_tokens = 0.0
    t0 = time.time()
    tokens_seen, loss_tokens_seen = 0, 0
    stop = False
    model.train()
    for epoch in range(args.epochs):
        batches = epoch_batches[epoch][rank::world]
        n_micro = min(len(batches), n_local)
        for mb in range(n_micro):
            idxs = batches[mb]
            ids, att, lab, th, ac = collate(train, idxs)
            ids, att = ids.to(device), att.to(device)
            lab, th, ac = lab.to(device), th.to(device), ac.to(device)
            h = inner(input_ids=ids, attention_mask=att).last_hidden_state
            loss, n_rows, tce, ace = gathered_ce(h, lab, th, ac, w_head)
            (loss / args.accum).backward()
            micro_count += 1
            accum_loss += loss.item()
            accum_tokens += n_rows
            tokens_seen += int(att.sum().item())
            loss_tokens_seen += n_rows

            if micro_count == args.accum or mb == n_micro - 1:
                step += 1
                micro_count = 0
                lr = set_lr(step)
                gnorm = grad_sync_and_clip(trainable, world, args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                dt = time.time() - t0
                if is_main:
                    rec = {"step": step, "epoch": epoch + 1, "loss": round(accum_loss / args.accum, 4),
                           "think_ce": round(tce, 4), "act_ce": round(ace, 4),
                           "lr": round(lr, 6), "grad_norm": round(float(gnorm), 3),
                           "tokens_per_s": round(tokens_seen / dt, 1),
                           "loss_tokens_per_s": round(loss_tokens_seen / dt, 1),
                           "samples": len(idxs),
                           "mem_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           "elapsed_s": round(dt, 1)}
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    print(rec, flush=True)
                accum_loss = accum_tokens = 0
                if step >= total_updates:
                    stop = True
                    break

        if is_main:  # epoch 收尾：先存后验，不因到达步数上限而跳过
            model.save_pretrained(os.path.join(args.out, f"adapter-ep{epoch + 1}"))
            vm = evaluate(base, w_head, val, args.token_budget, device)
            vm.update({"epoch": epoch + 1, "step": step, "event": "val"})
            log_f.write(json.dumps(vm) + "\n")
            log_f.flush()
            print(vm, flush=True)

        if stop:
            break

    if is_main:
        summary = {"total_updates": step, "wall_s": round(time.time() - t0, 1),
                   "tokens": tokens_seen, "loss_tokens": loss_tokens_seen,
                   "mem_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "config": {**vars(args), "world": world}}
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("SUMMARY", summary)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

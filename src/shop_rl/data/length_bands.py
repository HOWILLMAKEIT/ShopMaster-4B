#!/usr/bin/env python
"""按官方 Qwen3.5-4B chat_template 渲染并统计 SFT 轨迹的长度分层。

输入：data/sft/success_tasks.jsonl（每行 {task_id, file}，file 指向轨迹 JSON，
在 gong-4080 项目根运行）。处理：把 conversation 中 assistant 的
tool_calls.arguments 从 JSON 字符串解析回 dict，用官方模板渲染整条对话
（不加生成提示），再 tokenize 计长。输出：各层条数与 token 合计，
以及与长度审计报告（sft-qwen35-length-audit-2026-11-14.json）的一致性校验。

运行（gong-4080 项目根）：
  HF_ENDPOINT=https://hf-mirror.com .venv/bin/python shop_rl.data.length_bands

依赖 transformers；服务器代理不在线时先 unset 代理变量再走镜像安装，
命令见 AGENTS.md 环境与命令表。
"""
import json
import os
import statistics

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(key, None)

from transformers import AutoTokenizer

BANDS = [
    ("L1", "<=6000", 6000),
    ("L2", "6001-8192", 8192),
    ("L3", "8193-12000", 12000),
    ("L4", "12001-16384", 16384),
    ("L5", ">16384", None),
]
AUDIT = {  # 审计报告的聚合值；2026-11-14 重跑时七项全部一致
    "count": 1015, "min": 4010, "median": 5548,
    "kept_8192": 850, "kept_12288": 941, "kept_16384": 988,
    "total_tokens": 6795527,
}


def message_list(trajectory):
    messages = []
    for m in trajectory["conversation"]:
        m = dict(m)
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                fn = dict(call["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
                calls.append({**call, "function": fn})
            m["tool_calls"] = calls
        messages.append(m)
    return messages


def rendered_length(tokenizer, trajectory):
    text = tokenizer.apply_chat_template(
        message_list(trajectory),
        tools=trajectory["tools"],
        tokenize=False,
        add_generation_prompt=False,
    )
    # transformers 5.x 的 tokenize=True 返回 BatchEncoding，因此先取文本再 tokenize。
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    lengths = []
    with open("data/sft/success_tasks.jsonl") as f:
        for line in f:
            record = json.loads(line)
            with open(record["file"]) as tf:
                trajectory = json.load(tf)
            lengths.append(rendered_length(tokenizer, trajectory))

    lengths.sort()
    n = len(lengths)
    kept = {t: sum(1 for x in lengths if x <= t) for t in (8192, 12288, 16384)}
    checks = {
        "count": (n, AUDIT["count"]),
        "min": (lengths[0], AUDIT["min"]),
        "median": (statistics.median(lengths), AUDIT["median"]),
        "kept_8192": (kept[8192], AUDIT["kept_8192"]),
        "kept_12288": (kept[12288], AUDIT["kept_12288"]),
        "kept_16384": (kept[16384], AUDIT["kept_16384"]),
        "total_tokens": (sum(lengths), AUDIT["total_tokens"]),
    }
    result = {
        "checks_vs_audit": {k: {"got": g, "audit": a, "match": g == a} for k, (g, a) in checks.items()},
        "bands": {},
    }
    lower = 0
    for name, label, upper in BANDS:
        in_band = [x for x in lengths if x > lower and (upper is None or x <= upper)]
        result["bands"][name] = {"range": label, "count": len(in_band), "tokens": sum(in_band)}
        lower = upper if upper else lower
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

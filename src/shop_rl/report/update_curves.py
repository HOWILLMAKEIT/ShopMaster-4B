#!/usr/bin/env python3
"""将 A/B 原始 metrics.jsonl 的本地快照合并到 v1 HTML 的交互曲线。

在仓库根运行：shop_rl report curves
输入 outputs/v1/sft-curves.json（键 sft-a/sft-b，值为逐行解析后的 metrics）
及 sft-comparison.json；更新 docs/v1/summary-v1.html，保留其他章节。
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=ROOT / 'outputs/v1/sft-curves.json')
    parser.add_argument('--summary', type=Path, default=ROOT / 'outputs/v1/sft-comparison.json')
    parser.add_argument('--report', type=Path, default=ROOT / 'docs/v1/summary-v1.html')
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    summary = json.loads(args.summary.read_text())
    series = []
    for key, name, color in [('sft-a', 'A（带思考）', '#0f6b5f'),
                             ('sft-b', 'B（无思考）', '#386cb0'),
                             ('sft-b2a', 'B2A（两阶段）', '#b3541e')]:
        rows = metrics[key]
        train = [r.copy() for r in rows if r.get('event') != 'val']
        validation = [r for r in rows if r.get('event') == 'val']
        for epoch in sorted({r['epoch'] for r in train}):
            group = [r for r in train if r['epoch'] == epoch]
            for i, row in enumerate(group, 1):
                row['progress'] = epoch - 1 + i / len(group)
        if key == 'sft-b2a':
            # 两阶段只在 500 题上评过；成功率曲线取其中复用的同一批 200 题子集
            v18 = json.loads((ROOT / 'outputs/eval/v1-008/b2a-timing-and-failures.json').read_text())
            evaluation = [{'epoch': ep, 'success': v18['models_500'][f'b2a-ep{ep}']['overall']['reused200_pct']}
                          for ep in (1, 2, 3)]
        else:
            prefix = 'eval-b-' if key == 'sft-b' else 'eval-'
            evaluation = [{'epoch': ep, 'success': next(b['success_pct'] for b in summary['evaluation']['batches']
                          if b['batch'] == f'{prefix}ep{ep}-dev200')} for ep in (1, 2, 3)]
        series.append({'name': name, 'color': color, 'train': train,
                       'validation': validation, 'evaluation': evaluation})
    data = json.dumps(series, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    charts = '''<div id="sft-comparison-curves">
<h3>A / B / B2A 学习曲线对照</h3>
<p>同一指标叠加三组曲线：A（带思考直训）、B（无思考）、B2A（B ep2 续训 A 数据，两阶段）。
训练进度默认按 epoch 对齐（A 每轮 45 步，B 每轮 41 步，B2A 每轮 45 步），也可切换为优化步。
实线为滑动平均，淡线为原始日志；悬浮同时读取各组数值及各自的步数。
B2A 的逐 epoch 成功率取自其 500 题评测中复用的同一批 200 题子集。</p>
<div class="curve-controls">
<label>横轴 <select id="curve-axis"><option value="progress">训练进度（epoch）</option><option value="step">优化步</option></select></label>
<label>平滑 <select id="curve-window"><option value="1">不平滑</option><option value="9" selected>9 步</option><option value="21">21 步</option></select></label>
</div>
<div class="charts" id="sft-main-charts"></div>
<details><summary>查看 A 的思考段 CE（B 无思考监督，不画成 0）</summary><div id="sft-thought-chart"></div></details>
<p class="sub">总 loss 的监督目标在 A/B 间不同，不能用曲线高低判断哪组更好。
行动 CE 为每个优化步最后一个 micro-batch 的日志值，A/B 的格式 token 掩码也不完全相同。
梯度范数为裁剪前数值。滑动平均仅使用当前 epoch 内的历史点，不跨轮混合。
原始曲线快照：<a href="../../outputs/v1/sft-curves.json">sft-curves.json</a>。</p>
</div>
'''
    text = args.report.read_text()
    # 新版报告管理自己的图表布局。旧版文件仍可通过 --report 重新生成。
    if 'id="curve-metric"' not in text:
        if '<div id="sft-comparison-curves">' in text:
            start = text.index('<div id="sft-comparison-curves">')
        else:
            start = text.index('<div class="charts">')
        end = text.index('<h3>训练记录的解释范围</h3>', start)
        text = text[:start] + charts + '\n' + text[end:]
    runtime_start = text.rfind('<script>')
    start = text.rfind('<script type="application/json" id="sft-curve-data">', 0, runtime_start)
    if start < 0:
        start = runtime_start
    end = text.index('</script>', runtime_start) + len('</script>')
    runtime = Path(__file__).with_name('sft_curves.js').read_text().rstrip()
    text = text[:start] + '<script type="application/json" id="sft-curve-data">' + data + '</script>\n<script>\n' + runtime + '\n</script>' + text[end:]
    text = text.replace('<span class="no">3</span>SFT 过程（策略 A：带思考）',
                        '<span class="no">3</span>SFT 过程与 A/B 学习曲线')
    if '.curve-controls {' not in text:
        text = text.replace('</style>', '''.curve-controls { display:flex; flex-wrap:wrap; gap:18px; margin:12px 0; }
.curve-controls select { font:inherit; color:var(--ink); background:var(--card); border:1px solid var(--line); border-radius:6px; padding:4px 6px; }
.chart .curve-foot { margin:4px 6px; font-size:12px; color:var(--muted); }
.chart .tip { max-width:calc(100% - 16px); }
.chart svg { touch-action:pan-y; }
</style>''', 1)
    args.report.write_text(text)
    print(f'Updated {args.report}: A/B combined curves with epoch/step axes.')


if __name__ == '__main__':
    main()

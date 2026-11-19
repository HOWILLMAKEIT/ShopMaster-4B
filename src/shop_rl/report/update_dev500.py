#!/usr/bin/env python3
"""Update the 500-task results section from the v1-006 artifact."""
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]


def render_results(result):
    models = result['models']
    labels = {
        'base': '原模型',
        'a-ep2': 'A ep2 · 带思考',
        'b-ep2': 'B ep2 · 无思考',
    }
    baseline = models['base']['all500']['success_pct']
    result_rows, detail_rows = [], []
    for key, label in labels.items():
        full, added, reused = (models[key][k] for k in ('all500', 'added300', 'reused200'))
        assert (full['count'], added['count'], reused['count']) == (500, 300, 200)
        assert full['success'] == added['success'] + reused['success']
        low, high = full['wilson_95_pct']
        delta = '—' if key == 'base' else f'{full["success_pct"] - baseline:+.1f}'
        row_class = ' class="highlight"' if key == 'b-ep2' else ''
        result_rows.append(
            f'<tr{row_class}><th scope="row">{label}</th>'
            f'<td class="num"><strong>{full["success_pct"]:.1f}%</strong> <small>{full["success"]}/500</small></td>'
            f'<td class="num">{delta}</td>'
            f'<td class="num">{added["success_pct"]:.1f}% <small>{added["success"]}/300</small></td>'
            f'<td class="num">{full["completion_tokens"]:,}</td></tr>'
        )
        detail_rows.append(
            f'<tr><th scope="row">{label}</th>'
            f'<td class="num">{reused["success_pct"]:.1f}% <small>{reused["success"]}/200</small></td>'
            f'<td class="num">{low:.1f}–{high:.1f}%</td>'
            f'<td class="num">{full["statuses"].get("finished", 0)}</td>'
            f'<td class="num">{full["failure_reasons"].get("invalid_tool_call", 0)}</td>'
            f'<td class="num">{full["error_types"].get("context_limit", 0)}</td>'
            f'<td class="num">{full["failure_reasons"].get("no_tool_call", 0)}</td>'
            f'<td class="num">{added["completion_tokens"]:,}</td>'
            f'<td class="num">{full["unknown_usage_requests"]}</td></tr>'
        )

    a, b = (models[key]['all500'] for key in ('a-ep2', 'b-ep2'))
    pairs = result['pairs']['a-ep2_vs_b-ep2']
    full_pair, new_pair = (pairs[key] for key in ('all500', 'added300'))
    checked = datetime.fromisoformat(result['checked_at']).astimezone(
        ZoneInfo('America/Los_Angeles')
    ).strftime('%Y-%m-%d %H:%M %Z')
    saved = 100 * (1 - b['completion_tokens'] / a['completion_tokens'])
    return f'''<!--dev500:start-->
<section id="dev500-report" aria-labelledby="result-title">
  <div class="section-head">
    <span class="section-no" aria-hidden="true">01</span><h2 id="result-title">500 题扩展评测</h2>
  </div>
  <p class="section-lead">评测在用于选择 checkpoint 的 200 题上增加 300 题。三组使用相同题目，原结果经校验后逐题复用。所有失败计入分母，另有 959 道官方题未纳入本轮。</p>
  <div class="table-scroll" tabindex="0" role="region" aria-label="扩展评测主结果">
    <table id="evaluation-dev500" class="result-table">
      <caption>表 1　严格购买成功率与生成用量</caption>
      <thead><tr><th scope="col">模型</th><th scope="col" class="num">全部 500 题</th>
      <th scope="col" class="num">较原模型（百分点）</th><th scope="col" class="num">新增 300 题</th>
      <th scope="col" class="num">500 题已知生成 token</th></tr></thead>
      <tbody>{''.join(result_rows)}</tbody>
    </table>
  </div>
  <p class="caption">已知生成 token 为接口返回的 completion token 合计，包含失败任务的已知用量。实际耗时和完整成本需要另行测量。</p>
  <div class="result-reading">
    <p><strong>成功率。</strong>B 比 A 多成功 {b['success'] - a['success']} 题（{b['success_pct'] - a['success_pct']:+.1f} 个百分点）。</p>
    <p><strong>生成用量。</strong>B 的已知生成 token 比 A 少 {saved:.1f}%。非法工具失败从 A 的 {a['failure_reasons'].get('invalid_tool_call', 0)} 题降至 B 的 {b['failure_reasons'].get('invalid_tool_call', 0)} 题。</p>
  </div>
  <details class="report-detail">
    <summary>分项结果与成对检验</summary>
    <div class="table-scroll" tabindex="0" role="region" aria-label="扩展评测分项指标">
      <table class="dense-table">
        <caption>补充表　复用题结果、失败分类与用量</caption>
        <thead><tr><th scope="col">模型</th><th scope="col" class="num">复用 200 题</th><th scope="col" class="num">500 题 Wilson 95% CI</th>
        <th scope="col" class="num">正常结束</th><th scope="col" class="num">非法工具</th><th scope="col" class="num">上下文超限</th>
        <th scope="col" class="num">无工具调用</th><th scope="col" class="num">新增题已知生成 token</th>
        <th scope="col" class="num">500 题未知用量请求</th></tr></thead>
        <tbody>{''.join(detail_rows)}</tbody>
      </table>
    </div>
    <p>精确双侧 McNemar 检验只比较 A、B 成败不同的题。全部 500 题中 A、B 分别独自成功 {full_pair['first_only']}、{full_pair['second_only']} 题（<i>p</i> = {full_pair['mcnemar_exact_two_sided_p']:.4f}）。仅看新增 300 题，分别是 {new_pair['first_only']}、{new_pair['second_only']} 题（<i>p</i> = {new_pair['mcnemar_exact_two_sided_p']:.4f}）。两项比较均为探索性分析，未做多重比较校正。</p>
    <p>正常结束包含终局未成功购买的任务。上下文超限指输入加请求的 8,192 个最大输出 token 超过服务的 32,768 上限。Wilson 区间描述单个成功率的不确定性。</p>
  </details>
  <p class="source-line">v1-006 · 核对于 {checked} · <a href="../../outputs/eval/dev500-results.json">逐题结果与汇总</a> · <a href="../../outputs/eval/dev500-manifest.json">固定任务与协议</a></p>
</section>
<!--dev500:end-->'''


def main():
    result = json.loads((ROOT / 'outputs/eval/dev500-results.json').read_text())
    path = ROOT / 'docs/v1/summary-v1.html'
    text, count = re.subn(
        r'<!--dev500:start-->.*?<!--dev500:end-->',
        lambda _: render_results(result),
        path.read_text(),
        flags=re.S,
    )
    if count != 1:
        raise ValueError('Expected one dev500 section in the report')
    path.write_text(text)
    print('Updated:', path)


if __name__ == '__main__':
    main()

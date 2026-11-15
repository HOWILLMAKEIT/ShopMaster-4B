#!/usr/bin/env python3
"""Embed the completed v1-006 results in the local SFT HTML report."""
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def main():
    result = json.loads((ROOT / 'outputs/eval/dev500-results.json').read_text())
    models = result['models']
    labels = {'base': 'Base（原模型）', 'a-ep2': 'SFT-A ep2（带思考）', 'b-ep2': 'SFT-B ep2（无思考）'}
    rows, costs, failures = [], [], []
    baseline = models['base']['all500']['success_pct']
    for key, label in labels.items():
        scores = models[key]
        all_scores, added, reused = (scores[k] for k in ('all500', 'added300', 'reused200'))
        assert (all_scores['count'], added['count'], reused['count']) == (500, 300, 200)
        assert all_scores['success'] == added['success'] + reused['success']
        low, high = all_scores['wilson_95_pct']
        errors = all_scores['failure_reasons']
        rows.append(f'<tr><td>{label}</td><td class="num"><b>{all_scores["success"]}/500<br>{all_scores["success_pct"]:.1f}%</b></td>'
                    f'<td class="num">{all_scores["success_pct"]-baseline:+.1f}</td><td class="num">{low:.1f}–{high:.1f}%</td>'
                    f'<td class="num">{added["success"]}/300<br>{added["success_pct"]:.1f}%</td>'
                    f'<td class="num">{reused["success"]}/200<br>{reused["success_pct"]:.1f}%</td></tr>')
        failures.append(f'<tr><td>{label}</td><td class="num">{all_scores["statuses"].get("finished",0)}</td>'
                        f'<td class="num">{errors.get("invalid_tool_call",0)}</td>'
                        f'<td class="num">{all_scores["error_types"].get("context_limit",0)}</td>'
                        f'<td class="num">{errors.get("no_tool_call",0)}</td></tr>')
        costs.append(f'<tr><td>{label}</td><td class="num">{all_scores["completion_tokens"]:,}</td>'
                     f'<td class="num">{added["completion_tokens"]:,}</td><td class="num">{all_scores["unknown_usage_requests"]}</td></tr>')
    a, b = models['a-ep2']['all500'], models['b-ep2']['all500']
    pairs = result['pairs']['a-ep2_vs_b-ep2']
    full_pair, new_pair = pairs['all500'], pairs['added300']
    checked = datetime.fromisoformat(result['checked_at']).astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M PST')
    saved = 100 * (1-b['completion_tokens']/a['completion_tokens'])
    section = f'''<!--dev500:start-->
<section id="dev500-report">
<h2><span class="no">8</span>500 题扩展评测（v1-006，已完成）</h2>
<p><b>三组均完成 500/500，最后一组结束于 2026-11-15 00:49 PST。</b>结果核对：{checked}。
固定任务集保留原 200 题，再从剩余 1,259 题中用 seed 42 抽取 300 题，三模型同集。
原 200 题逐题结果经校验后复用，原失败记录也保留；本轮新执行 900 条轨迹，另复用 600 条。</p>
<div class="table-scroll"><table id="evaluation-dev500">
<thead><tr><th>模型</th><th class="num">扩展 500 题</th><th class="num">较 Base<br>百分点</th><th class="num">Wilson 95% 区间</th><th class="num">新增 300 题</th><th class="num">复用 200 题</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<div class="table-scroll"><table>
<tr><th>模型（500 题）</th><th class="num">正常结束</th><th class="num">非法工具</th><th class="num">上下文边界</th><th class="num">无工具调用</th></tr>
{''.join(failures)}</table></div>
<p>所有失败均保留在分母内。上下文边界指输入 token 加请求的 8,192 个最大输出 token 超过服务的 32,768 上限，
本轮三组的 <code>teacher_fatal</code> 都被归入此类；这些记录不应统称为超时。
本轮沿用原有预算，没有中途扩大上下文或减少输出预算后重测失败题。</p>
<h3>A/B 的成对比较</h3>
<p>B 比 A 多成功 <b>{b['success']-a['success']} 题（{b['success_pct']-a['success_pct']:+.1f} 个百分点）</b>。
500 题中 A 独自成功 {full_pair['first_only']} 题、B 独自成功 {full_pair['second_only']} 题，
精确双侧 McNemar p={full_pair['mcnemar_exact_two_sided_p']:.4f}，未达到 0.05 阈值。
新增 300 题中 A 独自成功 {new_pair['first_only']} 题、B 独自成功 {new_pair['second_only']} 题，
p={new_pair['mcnemar_exact_two_sided_p']:.4f}，接近该阈值。</p>
<p class="sub">这是探索性比较，未对多个比较作校正。500 题含已用于选择 ep2 的原 200 题，属于扩展开发集分析。
新增 300 题未参与此前 checkpoint 选择；本次仍只有一个训练种子和每题一次完整评测。
Wilson 区间反映成功比例的不确定性，不覆盖所有训练与采样误差；未显著不等于证明等效。</p>
<h3>生成开销与当前建议</h3>
<div class="table-scroll"><table>
<tr><th>模型</th><th class="num">500 题已知 completion token</th><th class="num">新增 300 题已知 completion token</th><th class="num">500 题未知用量请求</th></tr>
{''.join(costs)}</table></div>
<p>B 的已知 completion token 比 A 少 <b>{saved:.1f}%</b>；非法工具调用为 B 的 {b['failure_reasons'].get('invalid_tool_call',0)} 题与 A 的 {a['failure_reasons'].get('invalid_tool_call',0)} 题，
上下文边界失败则分别为 {b['error_types'].get('context_limit',0)} 与 {a['error_types'].get('context_limit',0)} 题。这些是本轮同时观察到的差异，不能单独证明因果机制。</p>
<p><b>当前建议仍是单域购物优先以 B ep2 作候选，保留 A ep2。</b>依据是本轮较高的观测成功率与较少的生成 token，
不把 500 题的差异写成已确认的总体优势。训练思考监督与推理开关仍未分别消融。
未知请求没有按 0 token 计算，已知 token 合计不等于完整成本，也不代表实测推理耗时。</p>
<h3>运行与证据</h3>
<p>A/B 先在两张 4090 上并行，B 完成后 Base 接同卡，与 A 并行。
验证初始 8 并发的内存余量后，各组提升到 16 并发并使用独立的 16 槽环境；incomplete 任务恢复，已完成任务不重跑。
本次临时推理服务已关闭，两张 4090 已释放。</p>
<p><a href="../../outputs/eval/dev500-results.json">结果与逐题成败快照</a> ·
<a href="../../outputs/eval/dev500-manifest.json">固定任务与运行协议</a> ·
<a href="../../outputs/eval/dev500-reuse-audit.json">原 200 题复用校验</a>。
完整轨迹仍在 4080 <code>outputs/eval/eval-{{base,a-ep2,b-ep2}}-dev500/</code>，状态、日志与源码快照在
<code>outputs/eval/v1-006/</code>。其余 959 个官方评测任务未纳入本轮。</p>
</section>
<!--dev500:end-->
'''
    path = ROOT / 'docs/v1/summary-v1.html'
    text = path.read_text()
    if '<!--dev500:start-->' in text:
        text = re.sub(r'<!--dev500:start-->.*?<!--dev500:end-->\n?', section, text, flags=re.S)
    else:
        pos = text.index('\n</div>\n<script type="application/json"')
        text = text[:pos]+section+text[pos:]
    start = text.index('<div class="headline">')
    end = text.index('\n<h2>', start)
    text = text[:start]+f'''<div class="headline">
<b>500 题扩展评测已完成。</b>Base {baseline:.1f}%（{models['base']['all500']['success']}/500），
A ep2 {a['success_pct']:.1f}%（{a['success']}/500），B ep2 <b>{b['success_pct']:.1f}%（{b['success']}/500）</b>。
B 较 Base +{b['success_pct']-baseline:.1f} 个百分点，较 A +{b['success_pct']-a['success_pct']:.1f} 个百分点。
本轮复用原 200 题并新增 300 题；分组结果与成对检验见 <a href="#dev500-report">第 8 节</a>。
</div>
'''+text[end:]
    text = re.sub(r'<title>.*?</title>', '<title>shop-grpo v1：A/B 学习曲线与 500 题扩展评测</title>', text, count=1)
    text = text.replace('结果核对：2026-11-14 23:58 PST（UTC−8）', '500 题结果核对：'+checked)
    text = text.replace('<span class="no">4</span>评测效果：七轮均为 200/200 题', '<span class="no">4</span>原 200 题逐 epoch 评测')
    text = text.replace('<span class="no">6</span>证据与后续工作', '<span class="no">6</span>原 200 题阶段的证据')
    text = text.replace('本次只整理既有产物，没有启动新训练或新评测。', '该次核对只整理当时既有产物；后续 500 题实验见第 8 节。')
    text = text.replace('<h3>本轮如何选</h3>', '<h3>原 200 题结果下的初步建议</h3>')
    if '<div class="note">后续可选：' in text:
        start_note = text.index('<div class="note">后续可选：')
        end_note = text.index('</div>', start_note)+len('</div>')
        text = text[:start_note]+'''<div class="note">后续 500 题扩展评测已完成，结果见 <a href="#dev500-report">第 8 节</a>。
其余 959 题未纳入本轮。思考监督与推理开关仍需分别比较。</div>'''+text[end_note:]
    path.write_text(text)
    print('Updated:', path)


if __name__ == '__main__':
    main()

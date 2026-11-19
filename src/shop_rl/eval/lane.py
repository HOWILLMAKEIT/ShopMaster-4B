#!/usr/bin/env python3
"""Run one v1-006 evaluation lane on the 4080 host after its services are ready."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

BASE_4090 = '''
import json,os,signal,subprocess,time
from pathlib import Path
root=Path('/mnt/data1/qiuwenjie/shop-rl')
info=json.loads((root/'logs/dev500-b-ep2-serve.json').read_text())
pid=info['pid']
try: command=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
except FileNotFoundError: command=''
if 'qwen3.5-4b-sftb-mm-ep2' in command:
    rows=[x.split(None,2) for x in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]]
    selected={pid}
    for _ in range(5): selected.update(int(p) for p,pp,args in rows if int(pp) in selected)
    for p in selected:
        try: os.kill(p,signal.SIGTERM)
        except ProcessLookupError: pass
    time.sleep(3)
    for p in selected:
        try: os.kill(p,signal.SIGKILL)
        except ProcessLookupError: pass
env=os.environ.copy()
env.update(HOME='/mnt/data1/qiuwenjie/home-eval',XDG_CACHE_HOME='/mnt/data1/qiuwenjie/xdg-cache',TRITON_CACHE_DIR='/mnt/data1/qiuwenjie/triton-cache',HF_HOME='/mnt/data1/qiuwenjie/hf-home',CUDA_VISIBLE_DEVICES='1',PATH=str(root/'venv-eval/bin')+':'+env['PATH'])
args=[str(root/'venv-eval/bin/python'),'-m','sglang.launch_server','--model-path',str(root/'models/Qwen3.5-4B'),'--served-model-name','qwen3.5-4b','--host','0.0.0.0','--port','8102','--mem-fraction-static','0.85','--context-length','32768','--attention-backend','triton','--sampling-backend','pytorch','--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3']
with (root/'logs/dev500-base-serve.log').open('ab') as log:
    proc=subprocess.Popen(args,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
state={'pid':proc.pid,'gpu':1,'port':8102,'args':args}
(root/'logs/dev500-base-serve.json').write_text(json.dumps(state,indent=2))
print(json.dumps(state),flush=True)
'''


def wait_for(check, description, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            print('Ready:', description, flush=True)
            return
        time.sleep(3)
    raise TimeoutError(description)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True, choices=['base', 'a-ep2', 'b-ep2'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--queue-base-after-b', action='store_true', help='Reuse GPU 1 for Base as soon as B finishes.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'outputs/eval/dev500-manifest.json').read_text())
    lane = next(x for x in manifest['lanes'] if x['label'] == args.label)
    out = ROOT / 'outputs/eval/v1-006'
    out.mkdir(exist_ok=True)
    state_path = out / f'{args.label}-state.json'
    state = {'label': args.label, 'workers': args.workers, 'status': 'waiting',
             'started_at': datetime.now(timezone.utc).isoformat()}

    def save():
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temp.replace(state_path)

    save()
    try:
        if args.queue_base_after_b:
            if args.label != 'base':
                raise ValueError('--queue-base-after-b requires --label base')
            state['status'] = 'queued_after_b'
            save()

            def b_complete():
                path = out / 'b-ep2-state.json'
                if not path.exists():
                    return False
                status = json.loads(path.read_text())['status']
                if status == 'failed':
                    raise RuntimeError('B lane failed before GPU handoff')
                return status == 'complete'

            wait_for(b_complete, 'B evaluation completion', timeout=7200)
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                     'qiuwenjie@222.201.145.4', 'python3 -'],
                                    input=BASE_4090, text=True, capture_output=True, check=True)
            state['service'] = json.loads(result.stdout)
            save()
            subprocess.run(['ssh', '-fN', '-L', '8102:127.0.0.1:8102', '-o', 'BatchMode=yes',
                            '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=15',
                            'qiuwenjie@222.201.145.4'], check=True)
        def model_ready():
            try:
                with HTTP.open(f'http://127.0.0.1:{lane["service_port"]}/get_model_info', timeout=3) as r:
                    info = json.load(r)
                return info.get('model_path', '').rstrip('/').endswith(lane['model_dir'])
            except (OSError, ValueError):
                return False

        def environment_ready():
            try:
                HTTP.open(f'http://127.0.0.1:{lane["environment_port"]}/api/shop_agent', timeout=3)
            except urllib.error.HTTPError as exc:
                return exc.code == 405
            except OSError:
                return False
            return False

        wait_for(model_ready, f'{args.label} model endpoint')
        wait_for(environment_ready, f'{args.label} environment')
        env = os.environ.copy()
        for name in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'):
            env.pop(name, None)
        command = [str(ROOT / '.venv/bin/python'), '-I', 'env/single_eval/collect_teacher.py',
                   'collect', '--yaml', lane['config'], '--workers', str(args.workers), '--retry-incomplete']
        with (out / f'{args.label}-collect.log').open('ab') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', collector_pid=proc.pid)
            save()
            code = proc.wait()
        if code:
            raise RuntimeError(f'collector exited with {code}')
        sys.path.insert(0, str(ROOT / 'env'))
        from get_score import is_success
        directory = ROOT / 'outputs/eval' / lane['batch_name']
        records = [json.loads(p.read_text()) for p in directory.glob('*.json') if p.stem.isdigit()]
        expected = set(manifest['reused_task_ids']) | set(manifest['added_task_ids'])
        assert len(records) == len({r['task_id'] for r in records}) == 500
        assert {r['task_id'] for r in records} == expected
        assert all(r['status'] in ('finished', 'failed') for r in records)
        scores = {}
        for label, ids in [('all500', expected), ('added300', set(manifest['added_task_ids'])),
                           ('reused200', set(manifest['reused_task_ids']))]:
            selected = [r for r in records if r['task_id'] in ids]
            success = sum(is_success(r) for r in selected)
            scores[label] = {'count': len(selected), 'success': success, 'success_pct': 100 * success / len(selected),
                             'failure_reasons': dict(Counter((r.get('failure_reason') or 'none').split(':')[0] for r in selected))}
        state.update(status='complete', results=scores, completed_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(state, ensure_ascii=False), flush=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()

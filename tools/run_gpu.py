#!/usr/bin/env python3
"""Safe new-directory GPU39 experiment launcher (invokes unmodified experiment apps).

The GPU execution method/source is unchanged; this file is a convenience wrapper.
No GPU requests are sent except under an explicit 'run --confirm-gpu-use'.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ENGINES={
 'original-v2':ROOT/'source/spra_workflow_sched_v2',
 'native-v21':ROOT/'source/spra_workflow_sched_v2_1',
 'fixed-case':ROOT/'source/spra_workflow_accept_v1',
}
SUITES={'original-v2':('smoke','ablation','grid','confirm'),
        'native-v21':('smoke','agecheck','ablation','grid','confirm'),
        'fixed-case':('smoke','agecheck','ablation','grid','confirm','case-confirm')}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','gate','freeze','plan','run','report','export','register-case','case-report','environment'])
    p.add_argument('--engine',choices=sorted(ENGINES),default='native-v21')
    p.add_argument('--suite',default=None)
    p.add_argument('--runtime',type=Path,help='A genuine existing native vLLM runtime directory; NOT supplied in this archive')
    p.add_argument('--out',type=Path,help='New distinct experiment path; required for running cases')
    p.add_argument('--max-conditions',type=int)
    p.add_argument('--replicas',nargs='+',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--confirm-gpu-use',action='store_true',help='Required to send REAL model requests')
    p.add_argument('--case',action='append',help='Pre-register e.g. --case 4:2:3')
    p.add_argument('--basis',type=Path,default=ROOT/'evidence/original/checkpoint_review_20261012.zip')
    p.add_argument('--reason',type=str)
    a=p.parse_args()
    if a.command in {'register-case','case-report','environment'} and a.engine!='fixed-case':
        p.error('This command belongs to --engine fixed-case')
    if a.command=='run' and not a.confirm_gpu_use:
        p.error('Refusing REAL GPU work without --confirm-gpu-use')
    if a.command in {'run','gate'} and a.runtime is None:
        p.error('--runtime /absolute/path/to/real/native/runtime is required (not included in the archive)')
    if a.command in {'run','report','export','case-report'} and a.out is None:
        p.error('--out is required: use a fresh results directory for run')
    if a.command=='run' and not a.resume and a.out.exists():
        p.error(f'Output path already exists: {a.out}. Never overwrite it; use a new path or --resume.')
    if a.command=='register-case' and (not a.case or not a.reason):
        p.error('--case R:rate:block and --reason (why chosen) are required')
    engine=ENGINES[a.engine]
    suite=a.suite or ('case-confirm' if a.engine=='fixed-case' else 'grid')
    if suite not in SUITES[a.engine]:
        p.error(f'{suite} not supported by {a.engine}. Supported suites: {SUITES[a.engine]}')
    app_cmd={'register-case':'register-cases'}.get(a.command,a.command)
    cmd=[sys.executable,str(engine/'run.py'),app_cmd,'--repo',str(ROOT/'source')]
    if a.runtime:cmd.extend(['--runtime',str(a.runtime.resolve())])
    if app_cmd in {'plan','run'}:cmd.extend(['--suite',suite])
    if a.out:cmd.extend(['--out',str(a.out.resolve())])
    if a.replicas and a.command=='run':cmd.extend(['--replicas',*map(str,a.replicas)])
    if a.max_conditions and a.command=='run':cmd.extend(['--max-conditions',str(a.max_conditions)])
    if a.resume and a.command=='run':cmd.append('--resume')
    if a.confirm_gpu_use and a.command=='run':cmd.append('--confirm-gpu-use')
    if a.command=='register-case':
        for x in a.case:cmd.extend(['--case',x])
        cmd.extend(['--basis',str(a.basis.resolve()),'--reason',a.reason])
    print('RUNNING_ORIGINAL_APP='+json.dumps({'engine':a.engine,'command':a.command,'argv':cmd},ensure_ascii=False),flush=True)
    result=subprocess.run(cmd,cwd=engine)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()

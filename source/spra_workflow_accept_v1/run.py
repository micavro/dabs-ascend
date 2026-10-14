#!/usr/bin/env python3
"""Workflow-only native experiment extension for GPU39."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from wfsched.common import HERE,ROOT,RUNTIME,atomic
from wfsched.design import plan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','freeze','gate','plan','run','report','export','benchmark','register-cases','case-report','environment'])
    p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--runtime',type=Path,default=RUNTIME)
    p.add_argument('--suite',choices=['smoke','agecheck','ablation','grid','confirm','case-confirm','port-smoke','port-cases','port-load','port-scale','port-grid','port-ablation'],default='smoke')
    p.add_argument('--out',type=Path)
    p.add_argument('--replicas',type=int,nargs='+',choices=[1,2,3,4],help='filter a frozen run, not redefine its plan')
    p.add_argument('--max-conditions',type=int,help='cleanly pause after this many newly accepted conditions')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--confirm-gpu-use',action='store_true')
    p.add_argument('--case',action='append',default=[],help='fixed R:rate:block, e.g. 4:2:3; registration only')
    p.add_argument('--basis',type=Path,help='existing selection evidence file; hashed before new runs')
    p.add_argument('--reason',default='',help='why these cases were chosen')
    args=p.parse_args()
    if sys.version_info<(3,11):raise RuntimeError('Existing native executor uses asyncio.timeout; Python 3.11+ required')
    if args.max_conditions is not None and args.max_conditions<1:p.error('--max-conditions must be positive')
    if args.command=='register-cases':
        from wfsched.acceptance import register
        if not args.case or args.basis is None or not args.reason:p.error('registration requires --case, --basis and --reason')
        print(json.dumps(register(args.case,args.basis,args.reason),ensure_ascii=False,indent=2))
    elif args.command=='case-report':
        if args.out is None:p.error('--out must identify an existing run')
        from wfsched.accept_report import write_report
        print('CASE_REPORT='+str(write_report(args.out)))
    elif args.command=='environment':
        from wfsched.environment import collect
        print('ENVIRONMENT_FILE='+str(collect(HERE)))
    elif args.command=='plan':
        value=plan(args.suite)
        if args.out:
            if args.out.exists():raise FileExistsError('refuse to overwrite plan output')
            atomic(args.out,value)
        else:print(json.dumps(value,ensure_ascii=False,indent=2))
    elif args.command=='check':
        from wfsched.legacy import inspect,load_original,load_profile
        receipt=inspect(args.repo,require_data=True)
        original=load_original(args.repo);load_profile(original,args.repo)
        from wfsched.selector import choose_lite
        from wfsched.worker import process_plan
        print('WF_CHECK='+json.dumps({'source_check':receipt,'native_import_ok':True,
               'placeholder_backend':False,'new_generation_requests':0},ensure_ascii=False),flush=True)
    elif args.command=='freeze':
        from wfsched.legacy import freeze
        value=freeze(args.repo,args.out or HERE/'frozen/D0')
        print('WF_FREEZE='+json.dumps(value,ensure_ascii=False,indent=2))
    elif args.command=='gate':
        from wfsched.runner import gate_only
        asyncio.run(gate_only(args))
    elif args.command=='run':
        if not args.confirm_gpu_use:p.error('run sends real requests; require --confirm-gpu-use')
        if args.out is None:args.out=HERE/'runs'/(args.suite+'_v2_1')
        from wfsched.runner import run
        asyncio.run(run(args))
    elif args.command in ('report','export'):
        if args.out is None:p.error('--out must identify the existing run directory')
        from wfsched.runner import read_run,export_run
        from wfsched.report import write
        if args.command=='export':export_run(args.out)
        else:
            value,rows,status=read_run(args.out)
            write(args.out,rows,value['plan']['conditions'],status)
            print('WF_REPORT='+str(args.out/'analysis/REPORT.md'))
    elif args.command=='benchmark':
        from wfsched.bench import run_benchmark
        target=args.out or HERE/'review/LOCAL_CPU_BENCHMARK.json'
        if target.exists():raise FileExistsError('benchmark output exists; use a fresh output path')
        value=run_benchmark(args.repo,target)
        print(json.dumps({k:v for k,v in value.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print('STOP: '+str(exc),file=sys.stderr)
        raise SystemExit(2)

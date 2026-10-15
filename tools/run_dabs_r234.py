#!/usr/bin/env python3
"""Run the independent complete R2-R4 x lambda4/8/12/16 experiment, never modifying the original experiment app.

Place as tools/run_dabs_r234.py in DABS_GPU39_Repro_v1.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'source/spra_workflow_fullgrid_r234_sll_v1'
SUITES=('dabs-r234-balanced-64',)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('check','freeze','gate','plan','run','report','export'))
    p.add_argument('--suite',choices=SUITES,default='dabs-r234-balanced-64')
    p.add_argument('--runtime',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--replicas',type=int,nargs='+',choices=(2,3,4))
    p.add_argument('--max-conditions',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--confirm-gpu-use',action='store_true')
    args=p.parse_args()
    if not APP.is_dir():p.error('Missing source/spra_workflow_fullgrid_r234_sll_v1; unpack kit first')
    if args.command in ('gate','run') and args.runtime is None:
        p.error('--runtime is required for gate/run')
    if args.command in ('run','report','export') and args.out is None:
        p.error('--out is required')
    if args.command=='run' and not args.confirm_gpu_use:
        p.error('refusing REAL GPU work without --confirm-gpu-use')
    if args.command=='run' and args.out.exists() and not args.resume:
        p.error('refusing to overwrite an existing output; choose new --out or --resume')
    if args.command=='run' and args.max_conditions is not None and args.max_conditions<1:
        p.error('--max-conditions must be positive')
    if args.command=='run' and args.resume and not args.out.exists():
        p.error('refusing to resume missing output')
    cmd=[sys.executable,str(APP/'run.py'),args.command,'--repo',str(ROOT/'source')]
    if args.runtime is not None:cmd += ['--runtime',str(args.runtime.resolve())]
    if args.command in ('run','plan'):cmd+=['--suite',args.suite]
    if args.out:cmd+=['--out',str(args.out.resolve())]
    if args.command=='run':
        if args.replicas:cmd+=['--replicas',*map(str,args.replicas)]
        if args.max_conditions:cmd+=['--max-conditions',str(args.max_conditions)]
        if args.resume:cmd+=['--resume']
        cmd+=['--confirm-gpu-use']
    print('RUNNING_DABS_FULLGRID_APP='+json.dumps({'argv':cmd,'new_dir':str(APP)},ensure_ascii=False),flush=True)
    # Export source roots to the subprocess itself. The original V3 wrapper
    # incorrectly relied on the caller's shell to retain run_checks.sh paths.
    src=ROOT/'source'
    env=os.environ.copy()
    local_paths=[str(src/'spra_stage19_musique_joint_v1'),
                 str(src/'spra_stage19_typed_handoff_v1')]
    if env.get('PYTHONPATH'):
        local_paths.append(env['PYTHONPATH'])
    env['PYTHONPATH']=os.pathsep.join(local_paths)
    env['PYTHONDONTWRITEBYTECODE']='1'
    return subprocess.call(cmd,cwd=APP,env=env)

if __name__=='__main__':
    raise SystemExit(main())

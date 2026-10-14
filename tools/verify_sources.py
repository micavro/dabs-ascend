#!/usr/bin/env python3
"""Validate source and evidence data without invoking any GPU service.

Use before running GPU experiments on a fresh machine.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-native',action='store_true',help='Also run source/native import checks (NO inference)')
    a=p.parse_args()
    checks=json.loads((ROOT/'provenance/COLLECTION_AUDIT.json').read_text(encoding='utf-8'))
    # Port input identity and original dataset manifest.
    asset=json.loads((ROOT/'provenance/input_assets.json').read_text(encoding='utf-8'))
    ds=ROOT/'source/spra_stage19_musique_joint_v1/dataset/compiled'
    bad=[]
    for fname,h in asset['files'].items():
        f=ds/fname
        if not f.exists() or sha(f)!=h:bad.append(fname)
    if sha(ds/'manifest.json')!=asset['source_dataset_manifest_sha256']:bad.append('dataset manifest')
    # Historic profile manifests, code and data are checked by the original check command.
    for f in ['source/spra_workflow_sched_v2_1/plans/grid.json',
              'source/spra_workflow_sched_v2/plans/ablation.json',
              'source/spra_workflow_accept_v1/wfsched/selector.py',
              'evidence/original/checkpoint_review_20261012.zip']:
        if not (ROOT/f).is_file():bad.append(f)
    if bad:raise SystemExit('ERROR: incorrect/missing files: '+', '.join(bad))
    print('DATASET_AND_SOURCE_FILES=PASS')
    print('METHOD_TRUTH=RR is round-robin; DAG is Ours; original Preble reproduction is NOT included')
    if a.test_native:
        for engine in ['native-v21','fixed-case']:
            cmd=[sys.executable,str(ROOT/'tools/run_gpu.py'),'check','--engine',engine]
            print('RUN_OFFLINE='+str(cmd),flush=True)
            if subprocess.call(cmd)!=0:raise SystemExit(f'FAIL_SOURCE_CHECK={engine}')
        print('NATIVE_SOURCE_CHECK=PASS')

if __name__=='__main__':main()

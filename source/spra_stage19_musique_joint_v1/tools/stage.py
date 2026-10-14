#!/usr/bin/env python3
"""Explicit idle-only staging. Previous deployment and results remain intact."""
from pathlib import Path
import argparse,fcntl,hashlib,json,shutil,sys,time,uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint18v2.common import atomic,load

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['stage','restore']);p.add_argument('--runtime',type=Path,default=Path('/home/wmy/sci/spra_core_exp/joint_e1'));p.add_argument('--backup',type=Path);a=p.parse_args()
    launcher=a.runtime/'launcher.py'
    if not launcher.is_file():raise SystemExit('STOP: launcher missing; no files changed')
    with (a.runtime/'experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        content=launcher.read_text();current_dir='joint18v2' if 'joint18v2_hook' in content else 'joint18'
        for path in (a.runtime/current_dir).glob('*.status.json'):
            s=load(path)
            if any(s.get(k) for k in ('running','waiting','swapped')):raise SystemExit('STOP: active recorded work in '+str(path))
        if a.command=='stage':
            if 'joint18v2_hook' in content:raise SystemExit('Already v2-staged; do not overwrite a live experiment deployment')
            if 'joint18_hook' not in content:raise SystemExit('Expected the currently deployed Stage18-v1 launcher; no implicit migration')
            b=a.runtime/'joint18v2_backups'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]);b.mkdir(parents=True)
            shutil.copy2(launcher,b/'launcher.py')
            if (a.runtime/'joint18v2').exists() and any((a.runtime/'joint18v2').glob('*.py')):raise SystemExit('Existing v2 code directory: restore or inspect explicitly rather than overwrite')
            (a.runtime/'joint18v2').mkdir(exist_ok=True)
            for f in (ROOT/'joint18v2').glob('*.py'):shutil.copy2(f,a.runtime/'joint18v2'/f.name)
            shutil.copytree(ROOT/'core18v2',a.runtime/'core18v2',dirs_exist_ok=True)
            shutil.copy2(ROOT/'runtime/joint18v2_hook.py',a.runtime/'joint18v2_hook.py')
            m={'backup':str(b),'old_sha256':sha(launcher),'new_sha256':sha(ROOT/'runtime/launcher.py'),
               'hook_sha256':sha(a.runtime/'joint18v2_hook.py'),'automatic_restart':False,'live_processes_changed':False}
            atomic(b/'manifest.json',m);atomic(a.runtime/'joint18v2/control.json',{'enabled':False})
            shutil.copy2(ROOT/'runtime/launcher.py',launcher)
            print('SOURCE_STAGED='+json.dumps(m),flush=True)
        else:
            if a.backup is None:raise SystemExit('restore requires exact --backup')
            m=load(a.backup/'manifest.json')
            if sha(launcher)!=m['new_sha256'] or sha(a.backup/'launcher.py')!=m['old_sha256']:raise SystemExit('Launcher hashes changed; refusing overwrite')
            atomic(a.runtime/'joint18v2/control.json',{'enabled':False});shutil.copy2(a.backup/'launcher.py',launcher)
            print('LAUNCHER_RESTORED=True AUTOMATIC_RESTART=False')
if __name__=='__main__':main()

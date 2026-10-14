"""Read-only import of the already deployed Stage19 execution/model code."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('STAGE19_SOURCE',ROOT.parent/'spra_stage19_musique_joint_v1')).resolve()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity():
    expected=json.loads((ROOT/'dependency_identity.json').read_text())['files'];actual={};bad={}
    for name,want in expected.items():
        p=SOURCE/name;got=sha(p) if p.is_file() else None
        if got!=want:bad[name]={'expected':want,'actual':got}
        actual[name]=got
    if bad:raise RuntimeError('Stage19 dependency differs; preserve source, do not edit hashes: '+json.dumps(bad))
    return actual

def activate():
    result=identity();sys.dont_write_bytecode=True
    if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
    return result

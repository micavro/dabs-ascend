#!/usr/bin/env python3
"""Check SHA256 of all immutable source/evidence/scripts in this reproduction bundle."""
from pathlib import Path
import argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]

def sha(f):
    h=hashlib.sha256()
    with open(f,'rb') as it:
        for b in iter(lambda: it.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only-source',action='store_true')
    a=p.parse_args(); mf=ROOT/'provenance/SHA256SUMS.txt'
    if not mf.exists():raise SystemExit('no manifest: '+str(mf))
    tested=0;errors=[]
    for line in mf.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        digest,rel=line.split('  ',1)
        if a.only_source and not rel.startswith('source/'):continue
        fp=ROOT/rel
        if not fp.is_file():errors.append('MISSING '+rel)
        elif sha(fp)!=digest:errors.append('HASH '+rel)
        tested+=1
    if errors:raise SystemExit('MANIFEST_FAILED: '+str(errors[:12])+f' (total {len(errors)})')
    print(f'SHA256_MANIFEST_PASS files={tested} scope={"source" if a.only_source else "all immutable files"}')
if __name__=='__main__':main()

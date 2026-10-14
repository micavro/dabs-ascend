#!/usr/bin/env python3
"""Package compiled public inputs and separated evaluation labels, not source archives."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'dataset/compiled/manifest.json').read_text())
for f,h in m['files'].items():
    if Path(f).name!=f:raise ValueError('nonlocal dataset manifest path')
    if hashlib.sha256((ROOT/'dataset/compiled'/f).read_bytes()).hexdigest()!=h:raise ValueError('data hash mismatch')
out=ROOT.parent/'spra_stage19_musique_data.zip'
if out.exists():raise FileExistsError('preserve the existing data zip; remove it explicitly before repacking')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for name in sorted(set(m['files'])|{'manifest.json'}):
        f=ROOT/'dataset/compiled'/name;z.write(f,str(Path(ROOT.name)/f.relative_to(ROOT)))
h=hashlib.sha256(out.read_bytes()).hexdigest()
sidecar=out.with_name(out.name+'.sha256');sidecar.write_text(h+'  '+out.name+'\n')
print('DATA_BUNDLE='+str(out));print('DATA_SHA256_FILE='+str(sidecar));print('SHA256='+h)

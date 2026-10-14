#!/usr/bin/env python3
"""CPU-only official MuSiQue acquisition; no pip, GPU, endpoint or mirror fallback."""
from __future__ import annotations
import argparse,hashlib,html,json,os,shutil,sys,urllib.request,urllib.parse,zipfile
from html.parser import HTMLParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from semantic19.dataset import convert,sha
FILE_ID='1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h'
OFFICIAL='https://github.com/StonyBrookNLP/musique'
URL='https://drive.usercontent.google.com/download?id='+FILE_ID+'&export=download&confirm=t'
class Form(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.fields={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':self.action=a.get('action')
        if tag=='input' and a.get('name'):self.fields[a['name']]=a.get('value','')

def download(path):
    import http.cookiejar
    jar=http.cookiejar.CookieJar();opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar));url=URL
    temp=path.with_suffix('.part')
    for attempt in range(3):
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 MuSiQue-research-fetch'})
        with opener.open(req,timeout=90) as response:
            prefix=response.read(65536)
            if prefix.startswith(b'PK\x03\x04'):
                with temp.open('wb') as f:
                    f.write(prefix);total=len(prefix)
                    while True:
                        b=response.read(1024*1024)
                        if not b:break
                        total+=len(b)
                        if total>2_000_000_000:raise ValueError('archive exceeds2GB acquisition bound')
                        f.write(b)
                if not zipfile.is_zipfile(temp):raise ValueError('not a complete zip')
                os.replace(temp,path);return
            body=prefix.decode('utf-8','replace');form=Form();form.feed(body)
            if form.action:
                parsed=urllib.parse.urlparse(form.action)
                if parsed.scheme!='https' or parsed.hostname not in ('drive.google.com','drive.usercontent.google.com'):
                    raise ValueError('unexpected Google download confirmation target')
                url=form.action+'?'+urllib.parse.urlencode(form.fields)
            else:raise RuntimeError('official Google Drive returned HTML, not archive; download the official archive in a browser and use --archive. Response prefix: '+body[:180])
    raise RuntimeError('official download confirmation did not complete; use --archive')

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);p.add_argument('--train',type=Path);p.add_argument('--dev',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'dataset/compiled');p.add_argument('--download-dir',type=Path,default=ROOT/'dataset/source');p.add_argument('--expected-archive-sha256')
    a=p.parse_args();a.download_dir.mkdir(parents=True,exist_ok=True)
    if a.train or a.dev:
        if not(a.train and a.dev):p.error('supply both --train and --dev')
        tr,de=a.train,a.dev;pro={'source':'user supplied official-format files; origin not independently authenticated','official_repository':OFFICIAL}
    else:
        archive=a.archive or a.download_dir/'musique_v1.0.zip'
        if not archive.exists():
            if a.archive:raise FileNotFoundError(archive)
            print('DOWNLOADING_OFFICIAL_ARCHIVE='+URL,flush=True);download(archive)
        hs=sha(archive)
        if a.expected_archive_sha256 and hs!=a.expected_archive_sha256:raise ValueError('official archive sha mismatch')
        targets={'musique_ans_v1.0_train.jsonl':None,'musique_ans_v1.0_dev.jsonl':None}
        with zipfile.ZipFile(archive) as z:
            for b in targets:
                names=[n for n in z.namelist() if Path(n).name==b and '__MACOSX' not in n]
                if len(names)!=1:raise ValueError('official archive lacks unique '+b)
                info=z.getinfo(names[0])
                if info.file_size>800_000_000:raise ValueError('dataset member unexpectedly large')
                target=a.download_dir/b
                with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
                targets[b]=target
        tr=targets['musique_ans_v1.0_train.jsonl'];de=targets['musique_ans_v1.0_dev.jsonl']
        pro={'official_repository':OFFICIAL,'official_file_id':FILE_ID,'archive_sha256':hs,'archive_source':'manual --archive' if a.archive else URL,
             'publisher_checksum_available':False,'transport':'official release URL; local content hash pinned at acquisition'}
    manifest=convert(tr,de,a.out,pro)
    print('DATASET_READY='+json.dumps({'out':str(a.out),'counts':manifest['counts'],'source_sha256':manifest['source_sha256'],'exclusions':manifest['schema_exclusions']}),flush=True)
    print('GPU_REQUESTS=0 MODEL_REQUESTS=0 NO_PACKAGE_INSTALLS=True')
if __name__=='__main__':
    try:main()
    except Exception as ex:print('DATASET_STOP='+repr(ex),file=sys.stderr);sys.exit(2)

"""Post-run labels only; answer normalization follows the official MuSiQue metric."""
from __future__ import annotations
import json,re,string,statistics
from collections import Counter
from pathlib import Path
from .workflow import extract_answer
from .dataset import sha

def normalize(text):
    text=''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())

def answer_score(pred,aliases):
    p=normalize(pred).split();em=0;best=0.0
    for gold in aliases:
        g=normalize(gold).split();em=max(em,int(p==g))
        overlap=sum((Counter(p)&Counter(g)).values())
        f=float(p==g) if not p or not g else 2*overlap/(len(p)+len(g))
        best=max(best,f)
    return {'em':em,'f1':best}

def evaluate(nodes,dataset_dir,duration_s):
    d=Path(dataset_dir);manifest=json.loads((d/'manifest.json').read_text());gpath=d/'evaluation_gold.jsonl'
    if sha(gpath)!=manifest['files'][gpath.name]:raise ValueError('evaluation gold changed')
    labels={g['id']:g for g in map(json.loads,gpath.read_text().splitlines())};work={}
    for n in nodes.values():work.setdefault(n['wid'],[]).append(n)
    rows=[]
    for wid,ns in work.items():
        last=max(ns,key=lambda n:n['step_position']);gold=labels[last['dataset_id']]
        pred,formatted=extract_answer(last['stream']['text']);score=answer_score(pred,[gold['answer']]+gold['answer_aliases'])
        sub=[]
        for n in sorted(ns,key=lambda n:n['step_position']):
            pp,fmt=extract_answer(n['stream']['text'])
            sub.append({'step':n['step_position']+1,'prediction':pp,'format_ok':fmt,
              **answer_score(pp,[gold['subanswers'][n['step_position']]]),'finish_reason':n['stream'].get('finish_reason'),
              'completion_tokens':n['stream']['usage']['completion_tokens']})
        rows.append({'wid':wid,'dataset_id':last['dataset_id'],'prediction':pred,'answer_format_ok':formatted,
                     **score,'steps':sub,'hop_count':len(ns),'shape':last['graph_shape']})
    return {'semantic_quality':{'final_em':statistics.mean(r['em'] for r in rows),'final_f1':statistics.mean(r['f1'] for r in rows),
      'questions':len(rows),'final_format_rate':statistics.mean(r['answer_format_ok'] for r in rows),
      'max_tokens_finish_count':sum(n['stream'].get('finish_reason')=='length' for n in nodes.values()),
      'label_access_only_after_execution':True,'official_leaderboard_comparable':False,
      'quality_scope':'known annotation DAG + deterministic context clipping; not unknown-plan open-domain QA'},
      'exact_match_completed_workflows_s':sum(r['em'] for r in rows)/duration_s,'quality_rows':rows}

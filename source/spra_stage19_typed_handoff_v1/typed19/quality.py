"""Post-condition quality. Invalid fields cannot masquerade as resolved entities."""
from __future__ import annotations
import json,statistics
from collections import Counter
from pathlib import Path
from .answers import parse
from semantic19.dataset import sha
from semantic19.quality import answer_score

def evaluate(nodes,dataset_dir,duration_s):
    if not nodes or any(n.get('state')!='done' for n in nodes.values()):raise RuntimeError('gold refused before the whole condition completes')
    d=Path(dataset_dir);m=json.loads((d/'manifest.json').read_text());p=d/'evaluation_gold.jsonl'
    if sha(p)!=m['files'][p.name]:raise ValueError('evaluation labels changed')
    labels={x['id']:x for x in map(json.loads,p.read_text().splitlines())};groups={};responses=[];rows=[]
    for n in nodes.values():groups.setdefault(n['wid'],[]).append(n)
    for wid,ns in groups.items():
        ordered=sorted(ns,key=lambda n:n['step_position']);g=labels[ordered[-1]['dataset_id']];steps=[]
        for n in ordered:
            stream=n['stream'];a=parse(stream['text'],stream.get('finish_reason'))
            pred=a['value'] if a['status']=='answer' else ''
            score=answer_score(pred,[g['subanswers'][n['step_position']]]) if a['status']=='answer' else {'em':0,'f1':0.0}
            entry={'step':n['step_position']+1,'prediction':pred,'format_ok':a['schema_ok'],
                   'answer_status':a['status'],'parse_reason':a['reason'],**score,
                   'finish_reason':stream.get('finish_reason'),'completion_tokens':stream['usage']['completion_tokens']}
            steps.append(entry)
            responses.append({'node':n['id'],'dataset_id':n['dataset_id'],'root':not n['parents'],
                'resolved_question':n.get('resolved_question'),'raw_response':stream['text'],
                'parent_response_bindings':n.get('parent_response_bindings',{}),**entry})
        last=ordered[-1];a=parse(last['stream']['text'],last['stream'].get('finish_reason'))
        pred=a['value'] if a['status']=='answer' else ''
        sc=answer_score(pred,[g['answer']]+g.get('answer_aliases',[])) if a['status']=='answer' else {'em':0,'f1':0.0}
        rows.append({'wid':wid,'dataset_id':last['dataset_id'],'prediction':pred,'answer_status':a['status'],
            'answer_format_ok':a['schema_ok'],**sc,'steps':steps,'hop_count':len(ns),'shape':last['graph_shape']})
    counts=Counter(r['answer_status'] for r in responses);roots=[r for r in responses if r['root']]
    result={'semantic_quality':{'final_em':statistics.mean(r['em'] for r in rows),
       'final_f1':statistics.mean(r['f1'] for r in rows),'questions':len(rows),
       'final_format_rate':statistics.mean(r['answer_format_ok'] for r in rows),
       'max_tokens_finish_count':sum(n['stream'].get('finish_reason')=='length' for n in nodes.values()),
       'label_access_only_after_execution':True,'official_leaderboard_comparable':False,
       'quality_scope':'known annotation DAG; canonical chat tokens, typed JSON answer-only, gold-blind 2700 evidence budget',
       'root_em':statistics.mean(r['em'] for r in roots),'root_f1':statistics.mean(r['f1'] for r in roots),
       'answer_status_counts':dict(counts),'answer_contract_errors':counts['format_error'],
       'unanswered_requests_not_removed':True,'format_success_does_not_imply_correctness':True},
       'exact_match_completed_workflows_s':sum(r['em'] for r in rows)/duration_s,
       'quality_rows':rows,'typed_response_rows':responses}
    for r in responses:print('TYPED_RESPONSE='+json.dumps(r,ensure_ascii=False),flush=True)
    return result

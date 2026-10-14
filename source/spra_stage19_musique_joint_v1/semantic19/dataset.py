"""Official-format MuSiQue -> known DAG. Gold is separated before execution.

#N is the ONE-BASED POSITION in question_decomposition, NOT its global id.
No support index/answer is used to select context or construct runtime nodes.
"""
from __future__ import annotations
import hashlib, json, re, random
from collections import Counter, defaultdict
from pathlib import Path
REF = re.compile(r'(?<!\w)#([1-9][0-9]*)(?!\d)')
FORBIDDEN = {'answer', 'answer_aliases', 'paragraph_support_idx', 'is_supporting', 'gold'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):h.update(b)
    return h.hexdigest()


def validate_public(row):
    if set(row) != {'id','question','paragraphs','steps','graph'}:raise ValueError('unexpected public keys')
    if not row['paragraphs'] or not row['steps']:raise ValueError('empty question')
    if len({p['idx'] for p in row['paragraphs']})!=len(row['paragraphs']):raise ValueError('duplicate paragraph idx')
    for p in row['paragraphs']:
        if set(p)!={'idx','title','paragraph_text'}:raise ValueError('label in public paragraph')
    for s in row['steps']:
        if set(s)!={'position','source_subquestion_id','question','parents'}:raise ValueError('label in public step')
        refs=sorted({int(m.group(1))-1 for m in REF.finditer(s['question'])})
        if refs!=s['parents'] or any(x>=s['position'] or x<0 for x in refs):raise ValueError('forward/self reference')
    return True


def compile_record(raw):
    if raw.get('answerable') is not True:raise ValueError('not answerable split')
    dec=raw.get('question_decomposition')
    if not isinstance(dec,list) or not 2<=len(dec)<=4:raise ValueError('expected official 2-4 hop decomposition')
    steps=[]
    for i,d in enumerate(dec):
        q=d['question']
        if not isinstance(q,str) or not q.strip():raise ValueError('empty subquestion')
        ps=sorted({int(m.group(1))-1 for m in REF.finditer(q)})
        if any(p<0 or p>=i for p in ps):raise ValueError('self/forward # reference or literal ambiguity')
        steps.append({'position':i,'source_subquestion_id':d['id'],'question':q,'parents':ps})
    n=len(steps);succ=[[] for _ in steps]
    for s in steps:
        for p in s['parents']:succ[p].append(s['position'])
    sinks=[i for i,x in enumerate(succ) if not x]
    if sinks!=[n-1]:raise ValueError('not a single annotated final sink')
    ancestors=[]
    for s in steps:
        a=set(s['parents'])
        for p in s['parents']:a.update(ancestors[p])
        ancestors.append(sorted(a))
    if set(ancestors[-1])!=set(range(n-1)):raise ValueError('disconnected decomposition')
    descendants=[[] for _ in steps]
    for i in reversed(range(n)):
        a=set(succ[i])
        for ch in succ[i]:a.update(descendants[ch])
        descendants[i]=sorted(a)
    depth=[]
    for s in steps:depth.append(1+max((depth[p] for p in s['parents']),default=0))
    layers=[[i for i,d in enumerate(depth) if d==level] for level in range(1,max(depth)+1)]
    # All legal ready antichains are compiled once (<=4 nodes, <=16 masks).
    antichains=[]
    for mask in range(1,1<<n):
        ids=[i for i in range(n) if mask>>i&1]
        if all(j not in ancestors[i] and i not in ancestors[j] for i in ids for j in ids if i!=j):antichains.append(ids)
    shape=';'.join(','.join(map(str,s['parents'])) or '-' for s in steps)
    public={'id':str(raw['id']),'question':raw['question'],
        'paragraphs':[{'idx':int(p['idx']),'title':p['title'],'paragraph_text':p['paragraph_text']} for p in raw['paragraphs']],
        'steps':steps,'graph':{'successors':succ,'ancestors':ancestors,'descendants':descendants,
        'topological_order':list(range(n)),'layers':layers,'antichains':antichains,
        'joins':[i for i,s in enumerate(steps) if len(s['parents'])>1],
        'max_parallel_width':max(map(len,antichains)),'hop_count':n,'shape':shape,
        'final_step':n-1,'graph_known_before_workflow_arrival':True}}
    validate_public(public)
    gold={'id':str(raw['id']),'answer':raw['answer'],'answer_aliases':raw.get('answer_aliases',[]),
          'subanswers':[d['answer'] for d in dec],
          'supporting_paragraph_indices':[int(p['idx']) for p in raw['paragraphs'] if p.get('is_supporting')],
          'used_by_runtime_or_selector':False}
    return public,gold


def balanced_select(rows,n,seed):
    """Topology-stratified selection, never based on LLM quality or serving outcome."""
    groups=defaultdict(list)
    for row in rows:groups[row['graph']['shape']].append(row)
    for key in groups:groups[key].sort(key=lambda r:hashlib.sha256((str(seed)+r['id']).encode()).hexdigest())
    keys=sorted(groups);random.Random(seed).shuffle(keys);out=[];positions=Counter()
    while len(out)<n:
        progress=False
        for key in keys:
            p=positions[key]
            if p<len(groups[key]) and len(out)<n:
                out.append(groups[key][p]);positions[key]+=1;progress=True
        if not progress:raise ValueError(f'only {len(out)} valid questions, need {n}')
    return out


def convert(train_file,dev_file,out,provenance=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('compiled manifest exists; preserve it, use a new --out')
    bysplit={};gold={};reject={};allcounts={}
    for split,path in [('train',train_file),('dev',dev_file)]:
        rows=[];bad=Counter();seen=set()
        with Path(path).open() as f:
            for line_number,line in enumerate(f,1):
                if not line.strip():continue
                raw=json.loads(line)
                try:p,g=compile_record(raw)
                except (ValueError,KeyError,TypeError) as e:
                    bad[type(e).__name__+': '+str(e)]+=1;continue
                if p['id'] in seen:raise ValueError('duplicate source id '+p['id'])
                seen.add(p['id']);rows.append(p);gold[p['id']]=g
        bysplit[split]=rows;reject[split]=dict(bad)
        allcounts[split]={'valid':len(rows),'shape_counts':dict(Counter(r['graph']['shape'] for r in rows))}
    train=balanced_select(bysplit['train'],34,190101);dev=balanced_select(bysplit['dev'],64,190102)
    if {r['id'] for r in train}&{r['id'] for r in dev}:raise ValueError('train/dev id overlap')
    sets={'smoke':train[:2],'profile':train[2:],'pilot':dev}
    files={}
    for split,rs in sets.items():
        path=out/(split+'.public.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs));files[path.name]=sha(path)
    ids={r['id'] for rs in sets.values() for r in rs}
    gp=out/'evaluation_gold.jsonl';gp.write_text(''.join(json.dumps(gold[k],ensure_ascii=False)+'\n' for k in sorted(ids)));files[gp.name]=sha(gp)
    notice=out/'ATTRIBUTION.txt'
    notice.write_text('MuSiQue: Multi-hop Questions via Single-hop Question Composition.\n'
        'Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal. TACL 2022.\n'
        'Source: https://github.com/StonyBrookNLP/musique ; official MuSiQue-Ans v1.0.\n'
        'License: Creative Commons Attribution 4.0 International\n'
        'https://creativecommons.org/licenses/by/4.0/\n'
        'Adaptation: topology-stratified subset, answer/support labels separated; annotated DAG retained.\n'
        'Inference additionally applies equal per-document prefix clipping under a 2700-token evidence budget.\n'
        'Known-plan reading-comprehension experiment, not original leaderboard protocol or production traces.\n')
    files[notice.name]=sha(notice)
    manifest={'schema':'stage19-musique-dataset-1','upstream':'StonyBrookNLP/musique','release':'MuSiQue-Ans v1.0',
        'license':'CC-BY-4.0','source_sha256':{'train':sha(train_file),'dev':sha(dev_file)},
        'provenance':provenance,'selection':'topology-stratified SHA256 order, seeds190101/190102; no quality or performance filtering',
        'counts':{k:len(v) for k,v in sets.items()},'source_graph_coverage':allcounts,'schema_exclusions':reject,
        'selected_graphs':{k:dict(Counter(r['graph']['shape'] for r in rs)) for k,rs in sets.items()},
        'files':files,'native_edges_preserved':True,'gold_separated':True,'known_plan_setting':True,
        'not_production_traffic':True,'no_real_dataset_examples_bundled_in_code_zip':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    return manifest


def load_public(directory,split):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text());p=directory/(split+'.public.jsonl')
    if sha(p)!=m['files'][p.name]:raise ValueError('public dataset hash mismatch')
    rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in rows:validate_public(r)
    return rows,m

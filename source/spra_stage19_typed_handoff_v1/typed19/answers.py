"""One machine-readable answer; no gold, no narrative-to-entity heuristic."""
from __future__ import annotations
import hashlib,json,re
SCHEMA={'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}
REVISION='stage19-typed-answer-1'
def _pairs(items):
    d={}
    for k,v in items:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d

def legacy_refusal_signal(text):
    """Forensic flag only; never rewrites old EM/F1 or supplies an answer."""
    s=text.strip().replace('\\n','\n')
    leading=bool(re.match(r'^UNKNOWN(?=$|\s|[.,;:])',s,re.I))
    tagged=bool(re.search(r'<(?:answer|solution)>\s*UNKNOWN\s*</(?:answer|solution)>',s,re.I))
    return leading or tagged

def parse(text,finish_reason=None):
    if not isinstance(text,str):raise TypeError('raw response must be text')
    result={'revision':REVISION,'status':'format_error','value':None,'schema_ok':False,
            'raw_response_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'legacy_refusal_signal':legacy_refusal_signal(text),'reason':None}
    try:obj=json.loads(text,object_pairs_hook=_pairs)
    except (ValueError,TypeError):result['reason']='not_one_complete_json_object';return result
    if not isinstance(obj,dict) or set(obj)!={'answer'} or type(obj['answer']) is not str:
        result['reason']='expected_only_string_answer';return result
    val=obj['answer'].strip()
    if not val or '\n' in val or '\r' in val or re.search(r'<[^>]+>',val):
        result['reason']='empty_multiline_or_markup_value';return result
    if finish_reason=='length':
        result['reason']='generation_reached_max_tokens';return result
    # A refusal plus explanation is not treated as a successfully resolved entity.
    if re.match(r'^UNKNOWN\b',val,re.I) and val.upper()!='UNKNOWN':
        result['reason']='unknown_with_extra_text';return result
    result.update(schema_ok=True,status='abstain' if val.upper()=='UNKNOWN' else 'answer',
                  value=None if val.upper()=='UNKNOWN' else val,reason=None)
    return result

def bind(template,parent_streams):
    from semantic19.dataset import REF
    used={}
    def sub(m):
        i=int(m.group(1))-1
        if i not in parent_streams:raise ValueError('unresolved parent reference #'+m.group(1))
        stream=parent_streams[i];raw=stream['text'];a=parse(raw,stream.get('finish_reason'))
        bound=a['value'] if a['status']=='answer' else None
        used[str(i+1)]={'response':raw,'response_sha256':a['raw_response_sha256'],
                       'bound_value':bound,'parse_status':a['status'],'parse_reason':a['reason'],
                       'source_is_actual_parent_response':True}
        return '('+bound+')' if bound is not None else '[UNAVAILABLE_STEP_'+str(i+1)+']'
    return REF.sub(sub,template),used

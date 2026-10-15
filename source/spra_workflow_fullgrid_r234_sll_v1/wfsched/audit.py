"""Validation is independent of whether a result is fast or accurate."""
from __future__ import annotations
import json
import statistics
from pathlib import Path
from .common import finite, load, digest
from .legacy import reproduction_module

def external_work(packets, proof):
    # Deliberately exclude fresh condition namespace; do not call this a token-identical contract.
    return digest({'packets':packets,'arrivals':proof['arrivals']})

def validate_condition(folder, summary, nodes, cell):
    folder=Path(folder);R=cell['replicas']
    reproduction_module().validate_fixed_summary(summary,len(nodes))
    if summary['workflows'] != cell['workflow_count']: raise ValueError('workflow count mismatch')
    if summary.get('control_time_match') is not True: raise ValueError('native action time mismatch')
    if summary.get('actual_commit_route_mismatches') != 0: raise ValueError('actual routing mismatch')
    controls=load(folder/'controls.json')
    if not controls or any(c.get('caps')!=[4096]*R or c.get('totals')!=[4096]*R for c in controls):
        raise ValueError('published caps or totals are not constant4096')
    fixed=cell.get('measurement_mode')=='fixed_token_replay'
    waits=[];prepare_waits=[];root_hashes={};roots_outputs={};bywf={}
    from collections import Counter
    repeated_roots=Counter((n['dataset_id'],n['step_position']) for n in nodes.values() if not n['parents'])
    for key,n in nodes.items():
        if n.get('state')!='done' or not all(finite(n.get(k)) for k in ('arrival','ready_at','start','finish')):
            raise ValueError('missing request completion/clock: '+key)
        if not (n['arrival']<=n['ready_at']+1e-6<=n['start']+2e-6<=n['finish']+3e-6):
            raise ValueError('request causal clock error: '+key)
        if type(n['replica']) is not int or not 0<=n['replica']<R: raise ValueError('inactive replica used')
        for p in n['parents']:
            if nodes[p]['finish']>n['ready_at']+1e-5: raise ValueError('dependency released too early')
            if not fixed:
                entry=n['parent_response_bindings'][str(nodes[p]['step_position']+1)]
                if entry.get('response')!=nodes[p]['stream']['text']: raise ValueError('parent binding differs')
        if not n.get('parents'):
            key2=n['dataset_id']+'/'+str(n['step_position'])
            if repeated_roots[n['dataset_id'],n['step_position']]>1:
                key2=key2+'@'+n['wid']
            if key2 in root_hashes:raise ValueError('repeated root audit identity')
            root_hashes[key2]=n['hash'];roots_outputs[key2]=n['stream']['text']
        waits.append(1000*(n['start']-n['ready_at']))
        if finite(n.get('input_prepare_end')):prepare_waits.append(1000*(n['start']-n['input_prepare_end']))
        bywf.setdefault(n['wid'],[]).append(n)
    times=[1000*(max(n['finish'] for n in ns)-min(n['arrival'] for n in ns)) for ns in bywf.values()]
    if abs(statistics.mean(times)-summary['mean_jct_ms'])>1e-3:
        raise ValueError('mean workflow JCT disagrees with original request clocks')
    output=sum(n['stream']['usage']['completion_tokens'] for n in nodes.values())
    if output!=summary['output_tokens']: raise ValueError('output token mismatch')
    q=summary.get('semantic_quality') or {}
    if not fixed and not all(finite(q.get(k)) and 0<=q[k]<=1 for k in ('final_em','final_f1')):
        raise ValueError('quality missing; do not discard slower or incorrect answers')
    plans=[json.loads(l) for l in (folder/'plans.jsonl').read_text().splitlines() if l.strip()]
    if cell['method'] in ('RR','P2C','SLL','Affinity','Preble'):
        if summary.get('online_search_calls')!=0: raise ValueError('baseline invoked optimizer')
        if any(p['action'].get('order') for p in plans): raise ValueError('baseline reordered native queue')
    if cell['method']=='Ayo':
        if summary.get('online_search_calls')!=0:
            raise ValueError('Ayo proxy baseline unexpectedly used workflow optimizer')
        if not plans or not any(p.get('action',{}).get('family')=='Ayo_Alg2_TopologyProxy' for p in plans):
            raise ValueError('Ayo Algorithm-2 proxy method not used by native dispatcher')
        for p in plans:
            a=p['action']
            if a.get('family')!='Ayo_Alg2_TopologyProxy' or not a.get('ayo_algorithm_2_proxy'):
                raise ValueError('Ayo proxy action identity missing')
            if a['caps']!=[4096]*R:
                raise ValueError('Ayo proxy changed native per-replica budget')
    if cell['method'] in ('DAG-W','DAG-H','Ours'):
        identity={'DAG-W':'OursV4_reference_first_isolated_actions', 'DAG-H':'OursV5_heterogeneous_reference_first', 'Ours':'OursV5_heterogeneous_reference_first'}[cell['method']]
        if summary.get('online_search_calls') and not any(p.get('wf_worker_path')==identity for p in plans):
            raise ValueError('workflow-action worker path missing')
        accepted_cells=('00','10','20','21') if cell['method'] in ('DAG-H','Ours') else ('00','10','11','20','21')
        if any(e.get('cell') not in accepted_cells for p in plans for e in p.get('evaluations',[])):
            raise ValueError('workflow selector evaluated unregistered candidate')
        if any(p.get('v4_selected_deviation') and p.get('selected_cell') not in ('10','11','20','21') for p in plans):
            raise ValueError('V4 selected action receipt mismatched')
        if any(p.get('v5_selected_deviation') and p.get('selected_cell') not in ('10','20','21') for p in plans):
            raise ValueError('V5 selected action receipt mismatched')
    if cell['method']=='DAG-A':
        if summary.get('online_search_calls'):
            if not any(p.get('wf_worker_path')=='OursV3_V1_first_bounded_candidates' for p in plans):
                raise ValueError('adaptive worker path missing')
        if any(e.get('cell') not in ('00','10','20','30') for p in plans for e in p.get('evaluations',[])):
            raise ValueError('adaptive method evaluated unexpected cell')
        if any(p.get('v3_selected_extra_candidate') and p.get('selected_cell') not in ('20','30') for p in plans):
            raise ValueError('adaptive selection receipt mismatched')
    if cell['method']=='DAG-V2':
        if summary.get('online_search_calls'):
            if not any(p.get('wf_worker_path')=='OursV2_two_destinations' for p in plans):
                raise ValueError('Ours-v2 did not use identifiable worker')
        if any(e.get('cell') not in ('00','10','20') for p in plans
               for e in p.get('evaluations',[])):
            raise ValueError('Ours-v2 evaluated a forbidden cell')
        if any(p.get('v2_selected_extra_destination') and p.get('selected_cell')!='20'
               for p in plans):
            raise ValueError('Ours-v2 destination receipt invalid')
    if cell['method']=='DAG-Lite':
        receipts=[p for p in plans if p.get('wf_worker_path')=='no_budget_candidates']
        if summary.get('online_search_calls') and not receipts:
            raise ValueError('Lite worker did not return identifiable result')
        if any(e.get('cell') not in ('00','10') for p in plans for e in p.get('evaluations',[])):
            raise ValueError('Lite evaluated a budgeting candidate')
    if cell.get('heterogeneous_workload'):
        from .common import digest
        manifest=load(folder/'predeclared_popularity_manifest.json')
        core={k:v for k,v in manifest.items() if k!='manifest_digest'}
        if digest(core)!=manifest.get('manifest_digest'):
            raise ValueError('heterogeneous workload manifest digest invalid')
        if manifest.get('profile')!=cell.get('popularity_profile'):
            raise ValueError('heterogeneous workload popularity profile changed')
        if summary.get('source_questions')!=manifest.get('source_packet_sequence'):
            raise ValueError('heterogeneous workload source sequence changed')
        if summary.get('output_tokens')!=manifest.get('total_declared_output_tokens'):
            raise ValueError('heterogeneous workload declared output total changed')
    fixed_receipt=None
    if fixed:
        from .fixed_work import work_fingerprint, INPUT_TOKENS, OUTPUT_TOKENS
        for key,n in nodes.items():
            if cell.get('heterogeneous_workload'):
                from .heterogeneous import output_budget
                expected_output=output_budget(n['graph_shape'])
            else:expected_output=OUTPUT_TOKENS
            if not n.get('fixed_workload_replay') or n['prompt']!=INPUT_TOKENS or n['output']!=expected_output:
                raise ValueError('fixed preflight token counts drifted: '+key)
            u=n['stream']['usage']
            if u.get('prompt_tokens')!=INPUT_TOKENS or u.get('completion_tokens')!=expected_output:
                raise ValueError('fixed REAL token count mismatch: '+key)
        fixed_receipt=work_fingerprint(nodes)
    return {'fixed_work_sha256':fixed_receipt,'measurement_mode':cell.get('measurement_mode','natural_json'),
       'native_fixed4096':True,'requests':len(nodes),'workflows':len(bywf),
       'output_tokens':output,'mean_jct_recomputed_ms':statistics.mean(times),
       'mean_ready_to_submit_ms':statistics.mean(waits),
       'mean_prepared_to_submit_ms':statistics.mean(prepare_waits) if prepare_waits else None,
       'root_prompt_hashes':root_hashes,'root_outputs':roots_outputs,
       'unique_native_prompt_hashes':(len({n['hash'] for n in nodes.values()})==len(nodes)),
       'fixed_native_hash_disambiguated_nodes':sum(bool(n.get('fixed_native_hash_disambiguated')) for n in nodes.values()),
       'prompt_token_identity_claimed':False,'cold_namespace_control_retained':True,
       'loop':'original_periodic_200ms','method':cell['method'],
       'budget_proposal_skip_receipts':sum(bool(p.get('budget_candidate_generation_skipped')) for p in plans),
       'raw_time_subtraction':False}

def inactive_offsets(runtime, R):
    return {r:(p.stat().st_size if p.exists() else 0) for r in range(R,4)
            for p in [Path(runtime)/'joint18v2'/f'spra-vllm-{r}.events.jsonl']}

def inactive_proof(runtime, offsets):
    for r,off in offsets.items():
        p=Path(runtime)/'joint18v2'/f'spra-vllm-{r}.events.jsonl'
        if not p.exists():
            if off: raise ValueError('inactive log disappeared')
            continue
        if p.stat().st_size<off:raise ValueError('inactive log truncated')
        with p.open('rb') as f:
            f.seek(off)
            for line in f:
                if not line.strip(): continue
                try: e=json.loads(line)
                except (ValueError,UnicodeDecodeError):raise ValueError('inactive log incomplete; no clean-idle claim')
                if e.get('event')=='accept' or (e.get('event')=='iteration' and e.get('scheduled')):
                    raise ValueError('inactive replica executed work; shared-resource interference')
    return {'inactive_replicas':list(offsets),'pass':True,'generation_events':0,'pods_stopped':False}

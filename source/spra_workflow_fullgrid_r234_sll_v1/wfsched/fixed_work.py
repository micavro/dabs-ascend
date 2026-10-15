"""Nonsemantic fixed-token DAG replay with unique native request hashes.

Purpose: resolve repeated-workflow request/native routing correlation. The native
executor MUST map every submitted request's prompt_hash to exactly one node.
For repeated public packets, same static prompts can otherwise collide.

The only new work is a deterministic instance discriminator in the *existing
padding tail* of duplicate prompts. It does not change prompt/output token
counts, DAG dependencies, or certified shared-prefix blocks. Single-copy
runs stay byte-for-byte the same as original V4 fixed work.

Not a semantic Agent result. Do not relabel published Preble or modify GPU hooks.
"""
from __future__ import annotations
from collections import defaultdict
from .common import digest
from joint18v2.common import token_hash

INPUT_TOKENS=3584
OUTPUT_TOKENS=16


def convert_nodes(nodes, pad_id, *, heterogeneous=False):
    if type(pad_id) is not int or pad_id<0:
        raise ValueError('invalid server tokenizer pad token')
    if not nodes:
        raise ValueError('empty fixed DAG nodes')
    by_hash=defaultdict(list)
    for key,n in nodes.items():
        prefix=list(n['tokens'])
        if len(prefix)>INPUT_TOKENS:
            raise ValueError(f'node {key}: static prompt {len(prefix)} exceeds fixed {INPUT_TOKENS}; do not truncate')
        if n['share']<=0:
            raise ValueError('missing native full-block cache sharing')
        # Native cache certificate must remain unchanged by tail-only padding.
        if token_hash(prefix[:n['share']]) != n['prefix_hash']:
            raise ValueError(f'node {key}: initial prefix certificate mismatch')
        padded=prefix+[pad_id]*(INPUT_TOKENS-len(prefix))
        if heterogeneous:
            from .heterogeneous import output_budget
            out_tokens=output_budget(n['graph_shape'])
        else:out_tokens=OUTPUT_TOKENS
        if INPUT_TOKENS+out_tokens>4096:
            raise ValueError('native model 4096-token context exceeded')
        n.update(tokens=padded,prompt=INPUT_TOKENS,hash=token_hash(padded),
                 output=out_tokens,output_is_estimate=False,
                 max_output_tokens=out_tokens,semantic_mode=False,
                 typed_answer_mode=False,prompt_materialized=True,
                 fixed_workload_replay=True,parent_response_bindings={},
                 fixed_pre_pad_prompt_tokens=len(prefix),
                 fixed_native_hash_disambiguated=False,
                 fixed_output_budget_predeclared=out_tokens)
        by_hash[n['hash']].append(key)
    total_disambiguated=0
    for duplicate_keys in by_hash.values():
        if len(duplicate_keys)==1:
            continue
        duplicate_keys=sorted(duplicate_keys)
        bits=(len(duplicate_keys)-1).bit_length()
        for index,key in enumerate(duplicate_keys):
            # Existing first copy retains its exact V4 token sequence.
            if index==0:
                continue
            n=nodes[key]
            original_len=n['fixed_pre_pad_prompt_tokens']
            if original_len+bits>INPUT_TOKENS:
                raise ValueError(f'node {key}: no free *padding* bits for native hash uniqueness')
            # The marker is an existing, validated model token ID. We only set
            # a few padding tokens at the very END of the prompt.
            marker=next((int(t) for t in n['tokens'][:original_len]
                         if type(t) is int and t>=0 and t!=pad_id),None)
            if marker is None:
                raise ValueError(f'node {key}: cannot find a distinct valid prompt token marker')
            for pos in range(bits):
                n['tokens'][INPUT_TOKENS-bits+pos]=(marker if (index>>pos)&1 else pad_id)
            if token_hash(n['tokens'][:n['share']])!=n['prefix_hash']:
                raise ValueError(f'node {key}: protected shared prefix was modified')
            n['hash']=token_hash(n['tokens'])
            n['fixed_native_hash_disambiguated']=True
            n['fixed_duplicate_group_size']=len(duplicate_keys)
            total_disambiguated+=1
    if heterogeneous:
        from .heterogeneous import output_budget
    hashes=[n['hash'] for n in nodes.values()]
    if len(set(hashes))!=len(hashes):
        raise ValueError('non-unique native request prompt_hash AFTER fixed replay conversion')
    if any(len(n['tokens'])!=INPUT_TOKENS or n['output']!=(output_budget(n['graph_shape']) if heterogeneous else OUTPUT_TOKENS) for n in nodes.values()):
        raise ValueError('nonuniform fixed node token counts')
    if any(token_hash(n['tokens'])!=n['hash'] or
           token_hash(n['tokens'][:n['share']])!=n['prefix_hash'] for n in nodes.values()):
        raise ValueError('native request hash or cached prefix checksum disagrees')
    return work_fingerprint(nodes)


def work_fingerprint(nodes):
    # Same structural-token-count pairing contract as original V4. Extra
    # per-instance prompt IDs are a *nonsemantic* uniqueness requirement.
    values=sorted((n['dataset_id'],n['step_position'],n['prompt'],n['output'],
                   tuple(n['parents'])) for n in nodes.values())
    return digest(values)

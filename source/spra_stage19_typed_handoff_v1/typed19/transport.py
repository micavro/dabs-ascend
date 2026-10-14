"""Keep native streaming/measurement boundaries; add one common JSON schema.

No dependency installation, guided backend override, response retry or silent
fallback to unstructured generation is performed. Grammar compilation and any
CPU decoding cost are part of measured request latency.
"""
from __future__ import annotations
import copy
from .answers import SCHEMA,parse
from joint18v2.transport import stream as native_stream
class SchemaClient:
    def __init__(self,client):self.client=client
    def stream(self,method,url,**kwargs):
        payload=copy.deepcopy(kwargs.get('json'))
        if not isinstance(payload,dict) or not url.endswith('/v1/completions'):raise ValueError('unexpected typed request endpoint')
        if 'min_tokens' in payload or 'ignore_eos' in payload:raise ValueError('typed answers must retain natural EOS')
        payload['guided_json']=copy.deepcopy(SCHEMA)
        kwargs['json']=payload
        return self.client.stream(method,url,**kwargs)
async def stream(client,endpoint,node,model):
    if not node.get('typed_answer_mode'):
        return await native_stream(client,endpoint,node,model)
    # The inherited stream starts its clock BEFORE SchemaClient.stream is called.
    result=await native_stream(SchemaClient(client),endpoint,node,model)
    node['answer_contract']=parse(result['text'],result.get('finish_reason'))
    return result

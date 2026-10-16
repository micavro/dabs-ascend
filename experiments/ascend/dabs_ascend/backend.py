"""Standard vLLM HTTP boundary, including version-specific JSON constraints."""
from __future__ import annotations

import copy
import json
import time
import uuid

from .common import append_json, source_paths
from .runtime import MODEL_NAME

source_paths()
from typed19.answers import SCHEMA, parse
from typed19.workflow import ServerTokenizer


def schema_properties(document, schema):
    if "$ref" in schema:
        target = document
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        return schema_properties(document, target)
    properties = dict(schema.get("properties", {}))
    for item in schema.get("allOf", []):
        properties.update(schema_properties(document, item))
    return properties


def constraint_field(document):
    schema = document["paths"]["/v1/completions"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    properties = schema_properties(document, schema)
    if "stream_options" not in properties:
        raise ValueError("Completions API lacks streaming usage; cannot produce auditable token counts")
    if "structured_outputs" in properties:
        return "structured_outputs"
    if "guided_json" in properties:
        return "guided_json"
    raise ValueError("No supported JSON-schema field in /openapi.json; unstructured fallback is disabled")


class Tokenizer(ServerTokenizer):
    """Use the unchanged canonical tokenizer methods with a configurable local port."""
    def __init__(self, client, endpoint):
        self.client, self.model, self.endpoint = client, MODEL_NAME, endpoint
        self.calls = self.decode_calls = self.chat_calls = 0
        self.first_blocks = set()


class Backend:
    def __init__(self, client, endpoints):
        self.client, self.endpoints = client, endpoints
        self.fields = []
        self.native = None

    async def inspect(self, *, fixed=False):
        records = []
        for endpoint in self.endpoints:
            response = await self.client.get(endpoint + "/v1/models")
            response.raise_for_status()
            if MODEL_NAME not in [row["id"] for row in response.json()["data"]]:
                raise ValueError("Unexpected served model at " + endpoint)
            response = await self.client.get(endpoint + "/openapi.json")
            response.raise_for_status()
            spec = response.json()
            for name in ("/tokenize", "/detokenize"):
                if name not in spec.get("paths", {}):
                    raise ValueError(f"Required canonical tokenizer API missing: {endpoint}{name}")
            if fixed:
                request = spec['paths']['/v1/completions']['post']['requestBody']['content']['application/json']['schema']
                properties = schema_properties(spec, request)
                if not {'stream_options', 'min_tokens', 'ignore_eos'} <= set(properties):
                    raise ValueError('Fixed replay requires stream_options, min_tokens and ignore_eos')
                field = None
            else:
                field = constraint_field(spec)
            self.fields.append(field)
            records.append({"endpoint": endpoint, "constraint_field": field,
                            "api_info": spec.get("info", {}), "native_gpu39_hook": False})
        if len(set(self.fields)) != 1:
            raise ValueError("Replicas expose different JSON constraint interfaces")
        return records

    async def generate(self, replica, node, seed, log, schema=SCHEMA):
        tokens = node["tokens"]
        context = self.native.capabilities[replica]['resources']['context_tokens'] if self.native else 4096
        if not tokens or len(tokens) + node["max_output_tokens"] > context:
            raise ValueError("Prompt plus output budget exceeds effective context limit")
        payload = {"model": MODEL_NAME, "prompt": tokens, "max_tokens": node["max_output_tokens"],
                   "temperature": 0., "top_p": 1., "seed": seed, "n": 1,
                   "stream": True, "stream_options": {"include_usage": True}, "add_special_tokens": False}
        if node.get('fixed_workload_replay'):
            if schema is not None:
                raise ValueError('Fixed replay must not use an answer schema')
            payload.update(min_tokens=node['max_output_tokens'], ignore_eos=True)
        if schema is not None:
            field = self.fields[replica]
            payload[field] = {"json": copy.deepcopy(schema)} if field == "structured_outputs" else copy.deepcopy(schema)
        started = time.monotonic()
        request_id = node['client_request_id'] if self.native else 'dabs-' + uuid.uuid4().hex
        node['client_request_id'] = request_id
        endpoint = self.endpoints[replica] + '/v1/completions'
        headers = {'X-Request-Id': request_id}
        if self.native:
            if schema is not None or not node.get('native_action_id'):
                raise ValueError('native fixed replay requires an engine admission permit')
            endpoint = self.endpoints[replica] + '/dabs/v1/generate'
            payload = {'client_request_id': request_id, 'node_id': node['id'], 'seed': seed,
                       'tokens': tokens, 'max_tokens': node['max_output_tokens'], 'cache_salt': node['cache_salt']}
        server_id = None
        fragments, first, last, usage, finish, done = [], None, None, None, None, False
        node["request_started_at"] = started
        append_json(log, {"event": "request", "node": node["id"], "replica": replica,
                          "at": started, "client_request_id": request_id, "payload": payload})
        try:
            async with self.client.stream("POST", endpoint, json=payload, headers=headers) as response:
                header = time.monotonic()
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise RuntimeError(f"vLLM HTTP {response.status_code}: {body[:4000]}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    now = time.monotonic()
                    if data == "[DONE]":
                        done = True
                        break
                    value = json.loads(data)
                    if value.get('id'):
                        if server_id is not None and value['id'] != server_id:
                            raise ValueError('SSE response ID changed within a request')
                        server_id = value['id']
                    append_json(log, {"event": "stream", "node": node["id"], "at": now, "data": value})
                    if "error" in value:
                        raise RuntimeError("vLLM stream error: " + str(value["error"]))
                    if value.get("usage"):
                        usage = value["usage"]
                    for choice in value.get("choices", []):
                        if choice.get("index", 0) != 0:
                            raise ValueError("Only n=1 is supported")
                        fragment = choice.get("text", "")
                        if fragment:
                            fragments.append(fragment)
                            first = now if first is None else first
                            last = now
                            # First text establishes only a lower bound on generated tokens.
                            node.update(phase="decode", computed=len(tokens), output_done=1,
                                        progress_source="HTTP first text; output_done is a lower bound")
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
            if not done or first is None or finish not in ("stop", "length") or usage is None:
                raise ValueError("Incomplete SSE: require [DONE], text, finish_reason and server token usage")
            if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
                raise ValueError("Invalid server token usage")
            if usage["prompt_tokens"] != len(tokens) or not 0 < usage["completion_tokens"] <= node["max_output_tokens"]:
                raise ValueError("Server input/output token usage differs from sent budget")
            if node.get('fixed_workload_replay') and usage['completion_tokens'] != node['max_output_tokens']:
                raise ValueError('Fixed replay output length differs from the predeclared DAG budget')
            ended = time.monotonic()
            result = {"text": "".join(fragments), "usage": usage, "finish_reason": finish,
                      'client_request_id': request_id, 'server_response_id': server_id,
                      'http_request_id_header': response.headers.get('x-request-id'),
                      'engine_action_acknowledged': None,
                      "start": started, "header_at": header, "finish": ended, "first_text_at": first, "last_text_at": last,
                      "request_ms": (ended - started) * 1000,
                      "ttft_text_proxy_ms": (first - started) * 1000,
                      "tpot_text_proxy_ms": ((last - first) * 1000 / (usage["completion_tokens"] - 1)
                                             if usage["completion_tokens"] > 1 else None),
                      "timing_boundary": "client HTTP text chunks; not exact native token/kernel timing"}
            node["stream"] = result
            if self.native:
                mapping = await self.native.call(replica, 'mapping', client_request_ids=[request_id])
                if len(mapping['mappings']) != 1 or mapping['mappings'][0].get('rejected'):
                    raise ValueError('native engine request binding failed')
                result['native_mapping'] = mapping
                result['engine_action_acknowledged'] = mapping['mappings'][0]['action_id']
            node["answer_contract"] = parse(result["text"], finish) if schema is not None else None
            append_json(log, {"event": "response", "node": node["id"], "result": result})
            return result
        except BaseException as exc:
            append_json(log, {"event": "request_failed", "node": node["id"], "error": repr(exc),
                              "partial_text": "".join(fragments), "usage": usage})
            raise

    async def fixed_canaries(self, out):
        reference, rows = None, []
        for replica, endpoint in enumerate(self.endpoints):
            tok = Tokenizer(self.client, endpoint)
            tokens = await tok.chat_tokens('Continue the text.', 'DABS fixed token replay.')
            if reference is not None and tokens != reference:
                raise ValueError('Replica canonical tokenization differs')
            reference = tokens
            if not await tok.decode(tokens):
                raise ValueError('Detokenization canary failed')
            pad = (await tok.encode(' .'))[-1]
            tokens = tokens + [pad] * (3584 - len(tokens))
            for count in (16, 32, 48, 64, 128):
                node = {'id': f'fixed-canary-r{replica}-o{count}', 'tokens': tokens,
                        'max_output_tokens': count, 'fixed_workload_replay': True}
                if self.native:
                    await self.native.reserve(replica, node)
                result = await self.generate(replica, node, 190500, out / 'canaries.jsonl', schema=None)
                rows.append({'replica': replica, 'passed': True, 'usage': result['usage'],
                             'tokenization_equal': True, 'fixed_output_budget': count})
        return rows

    async def canaries(self, out):
        reference = None
        rows = []
        # A deliberately conflicting prompt checks that JSON constraints are enforced.
        schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["DABS_CANARY"]}},
                  "required": ["answer"], "additionalProperties": False}
        for replica, endpoint in enumerate(self.endpoints):
            tok = Tokenizer(self.client, endpoint)
            tokens = await tok.chat_tokens("Return a JSON object.", 'Return {"answer":"WRONG"}.')
            if reference is not None and tokens != reference:
                raise ValueError("Replica canonical tokenization differs")
            reference = tokens
            decoded = await tok.decode(tokens)
            if not isinstance(decoded, str) or not decoded:
                raise ValueError("Detokenization canary failed")
            node = {"id": f"canary-r{replica}", "tokens": tokens, "max_output_tokens": 96}
            result = await self.generate(replica, node, 20261014, out / "canaries.jsonl", schema)
            if node["answer_contract"]["value"] != "DABS_CANARY":
                raise ValueError(f"JSON constraint canary failed on replica {replica}")
            # Warm the exact answer schema shared by ALL measured methods once.
            warm_tokens = await tok.chat_tokens("Return a JSON object with one answer field.",
                                               'Return {"answer":"READY"}.')
            warm = {"id": f"schema-warmup-r{replica}", "tokens": warm_tokens, "max_output_tokens": 96}
            await self.generate(replica, warm, 20261014, out / "canaries.jsonl", SCHEMA)
            if not warm["answer_contract"]["schema_ok"]:
                raise ValueError(f"Measured answer-schema warmup failed on replica {replica}")
            rows.append({"replica": replica, "passed": True, "usage": result["usage"],
                         "tokenization_equal": True, "constraint_enforced": True,
                         "measured_schema_warmed": True, "generation_requests": 2})
        return rows

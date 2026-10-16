"""Native routes forward control to EngineCore's actual utility RPC."""
import json


def install_api(app):
    from fastapi import Request, HTTPException
    from starlette.responses import StreamingResponse
    # Request must be available in the module namespace for FastAPI resolution.
    globals()['Request'] = Request
    async def rpc(request: Request):
        body = await request.body()
        if len(body) > 8_000_000:
            raise HTTPException(413, 'native request too large')
        try:
            return await request.app.state.engine_client.engine_core.call_utility_async('dabs_native', json.loads(body))
        except Exception as exc:
            raise HTTPException(409, str(exc)) from exc
    async def generate(request: Request):
        raw = await request.body()
        if len(raw) > 8_000_000:
            raise HTTPException(413, 'native request too large')
        body = json.loads(raw)
        if (not isinstance(body, dict) or any(not isinstance(body.get(k), str) or not 1 <= len(body[k]) <= 160
                for k in ('client_request_id', 'node_id', 'cache_salt')) or
                not isinstance(body.get('tokens'), list) or not body['tokens'] or
                any(type(t) is not int or t < 0 for t in body['tokens']) or
                type(body.get('max_tokens')) is not int or body['max_tokens'] <= 0 or type(body.get('seed')) is not int):
            raise HTTPException(422, 'invalid native request binding/tokens/budget')
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        cid = body['client_request_id']
        api_id = 'dabs-' + cid
        params = SamplingParams(temperature=0, seed=body['seed'], max_tokens=body['max_tokens'],
            min_tokens=body['max_tokens'], ignore_eos=True, output_kind=RequestOutputKind.DELTA,
            extra_args={'dabs_binding': {'node_id': body['node_id'], 'client_request_id': cid, 'server_request_id': api_id}})
        prompt = {'prompt_token_ids': body['tokens'], 'cache_salt': body['cache_salt']}
        async def stream():
            count = 0
            try:
                async for output in request.app.state.engine_client.generate(prompt, params, api_id):
                    choices = []
                    for choice in output.outputs:
                        count += len(choice.token_ids)
                        choices.append({'index': choice.index, 'text': choice.text, 'finish_reason': choice.finish_reason})
                    data = {'id': api_id, 'choices': choices}
                    if output.finished:
                        data['usage'] = {'prompt_tokens': len(body['tokens']), 'completion_tokens': count}
                    yield 'data: ' + json.dumps(data) + '\n\n'
                yield 'data: [DONE]\n\n'
            except Exception as exc:
                yield 'data: ' + json.dumps({'error': type(exc).__name__}) + '\n\n'
        return StreamingResponse(stream(), media_type='text/event-stream')
    app.add_api_route('/dabs/v1/control', rpc, methods=['POST'])
    app.add_api_route('/dabs/v1/generate', generate, methods=['POST'])

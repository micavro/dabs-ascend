from __future__ import annotations
import asyncio,json,time,statistics
from .common import quantile
from .latency_metrics import request_metrics
async def stream(client,endpoint,node,model):
    """No retries; exact fixed output usage is mandatory. Client receive events are not token ITL."""
    t0=time.monotonic();rec={"request_start":t0,"header":None,"first":None,"last":None,"return":None,
                            "usage":None,"server_id":None,"text_events":0,"error":None,"done":None,"content_event_times":[],"text":"","finish_reason":None}
    payload={"model":model,"prompt":node["tokens"],"max_tokens":node["output"],"min_tokens":node["output"],
      "ignore_eos":True,"temperature":0.0,"seed":0,"stream":True,
      "stream_options":{"include_usage":True},"add_special_tokens":False}
    if node.get('semantic_mode'):
        payload.pop('min_tokens',None);payload.pop('ignore_eos',None)
        payload['max_tokens']=node['max_output_tokens']
    try:
        async with client.stream("POST",endpoint+"/v1/completions",json=payload) as response:
            response.raise_for_status();rec["header"]=time.monotonic()
            got_done=False
            async for line in response.aiter_lines():
                if not line.startswith("data:"):continue
                part=line[5:].strip()
                if part=="[DONE]":rec["done"]=time.monotonic();got_done=True;break
                obj=json.loads(part);rec["server_id"]=obj.get("id",rec["server_id"])
                if obj.get("usage") is not None:rec["usage"]=obj["usage"]
                if obj.get("error"):raise RuntimeError(str(obj["error"]))
                for c in obj.get("choices",[]):
                    text=c.get("text")
                    if c.get("finish_reason") is not None:rec["finish_reason"]=c["finish_reason"]
                    if text:
                        t=time.monotonic()
                        if rec["first"] is None:rec["first"]=t
                        rec["last"]=t;rec["text"]+=text;rec["text_events"]+=1;rec["content_event_times"].append(t)
            if not got_done:raise RuntimeError("stream ended without DONE")
        rec["return"]=time.monotonic()
        if rec["usage"] is None:raise RuntimeError("missing token usage, not substituted by0")
        count=rec["usage"].get("completion_tokens")
        if node.get('semantic_mode'):
            if type(count)is not int or not 0<=count<=node['max_output_tokens']:raise RuntimeError('invalid natural output count')
        elif count!=node["output"]:raise RuntimeError("output work mismatch")
        if rec["usage"].get("prompt_tokens")!=node["prompt"]:raise RuntimeError("prompt token work mismatch")
    except BaseException as ex:
        rec["return"]=time.monotonic();rec["error"]=type(ex).__name__+": "+str(ex)
        node["stream"]=rec;raise
    node["stream"]=rec;return rec

def metrics(nodes,start,finish,actions):
    groups={}
    for n in nodes.values():groups.setdefault(n["wid"],[]).append(n)
    jcts=[];lags=[];phases={};workflow_rows=[]
    for wid,ns in groups.items():
        arr=ns[0]["arrival"];last=max(ns,key=lambda n:n["finish"] or -1)
        jct=last["finish"]-arr;jcts.append(jct*1000)
        lags.append((arr-(start+ns[0]["arrival_offset"]))*1000)
        chain=[];q=last
        while True:
            chain.append(q)
            if not q["parents"]:break
            q=max((nodes[p] for p in q["parents"]),key=lambda n:n["finish"])
        chain.reverse();previous=arr;parts={}
        for q in chain:
            x=q["stream"]
            first=x["first"] if x["first"] is not None else x["return"]
            lasttime=x["last"] if x["last"] is not None else first
            times={"dependency_release":q["ready_at"]-previous,
             "central_queue_and_admission":x["request_start"]-q.get("input_prepare_end",q["ready_at"]),
             "input_prepare":q.get("input_prepare_end",q["ready_at"])-q["ready_at"],
             "header_wait":x["header"]-x["request_start"],
             "postprocess":q["finish"]-x["return"]}
            if x["first"] is None:
                # No receipt timestamp is invented for a missing visible content event.
                times["unobserved_content_interval"]=x["return"]-x["header"]
            else:
                times.update(first_content_wait=first-x["header"],
                  content_receive=lasttime-first,response_tail=x["return"]-lasttime)
            if any(v<-.00001 for v in times.values()):raise RuntimeError("negative client timeline")
            for k,v in times.items():parts[k]=parts.get(k,0)+v*1000
            previous=q["finish"]
        if abs(sum(parts.values())-jct*1000)>.001:raise RuntimeError("critical-path decomposition mismatch")
        for k,v in parts.items():phases.setdefault(k,[]).append(v)
        workflow_rows.append({"wid":wid,"arrival":arr,"finish":last["finish"],"jct_ms":jct*1000,
          "blocking_chain":[n["name"] for n in chain],"critical_path_components_ms":parts})
    events=sorted([(n[0]["arrival"],1) for n in groups.values()]+[(max(x["finish"] for x in n),-1) for n in groups.values()])
    backlog=0;area=0;prev=events[0][0];curve=[]
    for t,d in events:area+=backlog*(t-prev);backlog+=d;curve.append([t-start,backlog]);prev=t
    if abs(area*1000-sum(jcts))>.01:raise RuntimeError("unfinished-workflow area identity failed")
    output=sum(n["stream"]["usage"]["completion_tokens"] for n in nodes.values())
    earliest=min(n["arrival"] for n in nodes.values());actual_finish=max(n["finish"] for n in nodes.values());duration=actual_finish-earliest
    return {"condition_valid":True,"workflows":len(groups),"requests":len(nodes),"output_tokens":output,
      "mean_jct_ms":statistics.mean(jcts),"p95_jct_ms":quantile(jcts,.95),
      "planned_arrival_mean_jct_ms":statistics.mean(jcts)+statistics.mean(lags),
      "max_arrival_lag_ms":max(lags),"cohort_duration_s":duration,
      "cohort_workflows_s":len(groups)/duration,
      "runner_observation_lag_s":max(0.,finish-actual_finish),
      "legacy_runner_duration_s":finish-earliest,
      "cohort_end_boundary":"last actual request completion, not loop exit","cohort_tokens_s":output/duration,
      "maximum_capacity_established":False,"semantic_quality":None,
      "mean_critical_path_components_ms":{k:sum(w["critical_path_components_ms"].get(k,0) for w in workflow_rows)/len(workflow_rows) for k in phases},
      "requests_without_visible_content":sum(n["stream"]["first"] is None for n in nodes.values()),
      "unfinished_workflow_curve":curve,"sum_jct_ms":sum(jcts),
      "workflow_rows":workflow_rows,"controller_decisions":len(actions),
      "gpu_kernel_time_ms":None,**request_metrics(nodes)}


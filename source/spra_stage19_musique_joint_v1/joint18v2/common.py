"""Pure contracts shared by the client, the engine hook and CPU tests."""
from __future__ import annotations
import hashlib, json, math, os, time, uuid
from pathlib import Path

VERSION = "spra-joint18-v2-integration-1"
CAPS = (256, 512, 1024, 2048, 4096)
PREFIX_SIZES = (256, 512, 1024, 1536, 2048, 3072, 4096)

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def token_hash(tokens): return digest(list(tokens))
def load(p): return json.loads(Path(p).read_text())
def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+"."+uuid.uuid4().hex+".tmp")
    with tmp.open("w") as f:json.dump(obj,f,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    os.replace(tmp,p)
def append(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a") as f:f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
def boot_id():return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
def clock_namespace():
    try:return os.stat('/proc/self/ns/time').st_ino
    except FileNotFoundError:return None
def integer(x,low,high,label):
    if type(x) is not int or not low<=x<=high:raise ValueError(label)
    return x

def validate_plan(p,identity,envelope,now=None):
    """Invalid or stale leases fail to native startup TOTAL budget, never a guessed policy."""
    now=time.monotonic_ns() if now is None else now
    if not isinstance(p,dict) or p.get("enabled") is not True:raise ValueError("disabled")
    if p.get("boot_id")!=identity["boot_id"] or p.get("pid")!=identity["pid"]:
        raise ValueError("engine_identity")
    if p.get("pod")!=identity["pod"]:raise ValueError("pod_identity")
    issued=integer(p.get("issued_ns"),0,now+1_000_000,"issued_ns")
    expiry=integer(p.get("expires_ns"),0,now+2_000_000_000,"expiry")
    if expiry<=now or expiry<=issued or expiry-issued>2_000_000_000:raise ValueError("lease")
    total=integer(p.get("total"),envelope["max_num_seqs"],envelope["hard_total"],"total")
    cap=p.get("prefill_cap")
    if total % envelope["block_size"]:raise ValueError("total_alignment")
    if cap is not None:
        integer(cap,envelope["block_size"],envelope["hard_total"],"prefill_cap")
        if cap % envelope["block_size"]:raise ValueError("prefill_alignment")
    order=p.get("order",[])
    if not isinstance(order,list) or any(not isinstance(x,str) for x in order) or len(order)!=len(set(order)):
        raise ValueError("order")
    if not isinstance(p.get("case"),str) or not isinstance(p.get("plan_id"),str):raise ValueError("case_plan")
    return p

def stable_prefill_reorder(items,priority,now_s,age_limit_s=5.0):
    """Native decode positions and relative order are preserved. Unknown prefill stays stable."""
    rank={h:i for i,h in enumerate(priority)}
    positions=[i for i,x in enumerate(items) if x["prefill"]]
    pref=[items[i] for i in positions]
    if any(now_s-x["accepted_s"]>=age_limit_s for x in pref):return list(items),False,"age_fifo"
    sorted_pref=sorted(pref,key=lambda x:rank.get(x["prompt_hash"],len(rank)))
    out=list(items)
    for i,x in zip(positions,sorted_pref):out[i]=x
    changed=[x["request_id"] for x in out]!=[x["request_id"] for x in items]
    return out,changed,"rank" if changed else "unchanged"

def make_budget_class(Native,context):
    """Add independent prefill accounting without charging native decode against the prefill cap."""
    class Budget(Native):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.j18_cap=context().get("prefill_cap")
            self.j18_prefill_used=0
            self.j18_phases={}
            self.j18_counted={}
            self.j18_current_prefill=False
        def remaining_token_budget(self):
            left=super().remaining_token_budget()
            if self.j18_current_prefill and self.j18_cap is not None:
                left=min(left,self.j18_cap-self.j18_prefill_used)
            return max(0,left)
        def can_schedule(self,*,num_new_tokens,num_new_seqs):
            ok=super().can_schedule(num_new_tokens=num_new_tokens,num_new_seqs=num_new_seqs)
            if self.j18_current_prefill and self.j18_cap is not None:
                ok=ok and self.j18_prefill_used+num_new_tokens<=self.j18_cap
            return ok
        def add_num_batched_tokens(self,req_id,num_batched_tokens):
            duplicate=req_id in self._request_ids_num_batched_tokens
            super().add_num_batched_tokens(req_id,num_batched_tokens)
            if not duplicate and self.j18_phases.get(req_id,False):
                self.j18_counted[req_id]=num_batched_tokens
                self.j18_prefill_used+=num_batched_tokens
        def subtract_num_batched_tokens(self,req_id,num_batched_tokens):
            super().subtract_num_batched_tokens(req_id,num_batched_tokens)
            if req_id in self.j18_counted:self.j18_prefill_used-=self.j18_counted.pop(req_id)
    return Budget

def quantile(xs,p):
    if not xs:return None
    v=sorted(xs);a=(len(v)-1)*p;i=int(a);j=min(i+1,len(v)-1)
    return v[i]*(j-a)+v[j]*(a-i) if i!=j else v[i]

def action_key(a):
    return digest({k:a[k] for k in ("placement","admit","order","caps")})

def validate_action(a,snapshot,config):
    nodes=snapshot["nodes"];R=snapshot["replicas"];ready=set(snapshot["ready"])
    if set(a["placement"])!=ready:raise ValueError("placement must cover exactly unsubmitted ready set")
    if not set(a["admit"])<=ready or len(a["admit"])!=len(set(a["admit"])):raise ValueError("admit")
    for k,r in a["placement"].items():
        integer(r,0,R-1,"replica")
        if nodes[k]["state"]!="ready":raise ValueError("rewrite_submitted")
        if any(nodes[p]["state"]!="done" for p in nodes[k]["parents"]):raise ValueError("dependency")
    if len(a["caps"])!=R or any(type(x) is not int or x not in config["caps"] for x in a["caps"]):raise ValueError("caps")
    if len(a["order"])!=len(set(a["order"])):raise ValueError("duplicate_priority")
    for k in a["order"]:
        if k not in nodes:raise ValueError("unknown_priority")
    aged={k for k in ready if snapshot["now"]-nodes[k]["ready_at"]>=config["max_admission_wait_s"]}
    if not aged<=set(a["admit"])|set(a.get("resource_held",[])):raise ValueError("age_admission")
    return True

# The transport lease and algorithm action horizon are deliberately separate.
def resolve_control(p,identity,envelope,now=None):
    now=time.monotonic_ns() if now is None else int(now)
    p=validate_plan(p,identity,envelope,now)
    window=p.get('action_expires_ns');snapshot=p.get('snapshot_ns');not_before=p.get('not_before_ns')
    integer(window,0,now+1_000_000_000,'action_expires_ns')
    integer(snapshot,0,now+1_000_000,'snapshot_ns')
    integer(not_before,0,now+1_000_000_000,'not_before_ns')
    if window<=snapshot or window-snapshot>200_000_000:raise ValueError('action_window_must_not_exceed_200ms')
    if not_before<snapshot or not_before>=window:raise ValueError('action_activation_interval')
    continuation=p.get('continuation')
    if not isinstance(continuation,dict):raise ValueError('missing_common_continuation')
    for name in ('prefill_cap','total'):
        val=continuation.get(name)
        integer(val,envelope['block_size'] if name=='prefill_cap' else envelope['max_num_seqs'],envelope['hard_total'],'continuation_'+name)
        if val%envelope['block_size']:raise ValueError('continuation_alignment')
    out=dict(p)
    if now<not_before or now>=window:
        out.update(total=continuation['total'],prefill_cap=continuation['prefill_cap'],order=[])
        out['execution_phase']='common_continuation'
    else:out['execution_phase']='current_action'
    return out

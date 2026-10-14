import sys,copy,unittest,contextvars,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from joint18v2.common import *
from fake_native import NativeBudget
def config():
    c=load(ROOT/"configs/development.json");c["workflow_count"]=1;return c
class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.ctx={"prefill_cap":256};self.B=make_budget_class(NativeBudget,lambda:self.ctx);self.b=self.B(4096,64)
    def phase(self,rid,pre):self.b.j18_current_prefill=pre;self.b.j18_phases[rid]=pre
    def test_prefill_limit(self):
        self.phase("p",True);self.assertEqual(self.b.remaining_token_budget(),256)
        self.b.add_num_batched_tokens("p",256);self.assertEqual(self.b.remaining_token_budget(),0)
    def test_decode_not_prefill(self):
        self.phase("d",False);self.b.add_num_batched_tokens("d",1)
        self.phase("p",True);self.assertEqual(self.b.remaining_token_budget(),256)
    def test_decode_after_cap_exhausted(self):
        self.phase("p",True);self.b.add_num_batched_tokens("p",256)
        self.phase("d",False);self.assertTrue(self.b.can_schedule(num_new_tokens=1,num_new_seqs=1))
    def test_duplicate(self):
        self.phase("p",True);self.b.add_num_batched_tokens("p",128);self.b.add_num_batched_tokens("p",128)
        self.assertEqual(self.b.j18_prefill_used,128)
    def test_subtract(self):
        self.phase("p",True);self.b.add_num_batched_tokens("p",128);self.b.subtract_num_batched_tokens("p",128)
        self.assertEqual(self.b.remaining_token_budget(),256)
    def test_total_ceiling(self):
        b=self.B(64,64);b.j18_current_prefill=False;b.add_num_batched_tokens("d",64)
        self.assertFalse(b.can_schedule(num_new_tokens=1,num_new_seqs=1))
    def test_sequence_ceiling(self):
        self.b.add_num_seqs("x",64);self.assertFalse(self.b.can_schedule(num_new_tokens=1,num_new_seqs=1))
    def test_uncapped_native(self):
        self.ctx["prefill_cap"]=None;b=self.B(2048,64);b.j18_current_prefill=True
        self.assertEqual(b.remaining_token_budget(),2048)

class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.i={"pod":"p0","pid":11,"boot_id":"b"}
        self.e={"hard_total":4096,"max_num_seqs":64,"block_size":16}
        self.p={"enabled":True,**self.i,"issued_ns":100,"expires_ns":1000,"total":4096,
                "prefill_cap":256,"order":[],"case":"x","plan_id":"a"}
    def test_valid(self):self.assertEqual(validate_plan(self.p,self.i,self.e,200),self.p)
    def test_bad_pid(self):
        p=copy.deepcopy(self.p);p["pid"]=10
        with self.assertRaises(ValueError):validate_plan(p,self.i,self.e,200)
    def test_expired(self):
        with self.assertRaises(ValueError):validate_plan(self.p,self.i,self.e,2000)
    def test_boolean_budget(self):
        self.p["prefill_cap"]=True
        with self.assertRaises(ValueError):validate_plan(self.p,self.i,self.e,200)
    def test_alignment(self):
        self.p["prefill_cap"]=257
        with self.assertRaises(ValueError):validate_plan(self.p,self.i,self.e,200)
    def test_duplicate_order(self):
        self.p["order"]=["a","a"]
        with self.assertRaises(ValueError):validate_plan(self.p,self.i,self.e,200)
    def test_disabled(self):
        self.p["enabled"]=False
        with self.assertRaises(ValueError):validate_plan(self.p,self.i,self.e,200)
    def test_hard_limit(self):
        self.p["total"]=8192
        with self.assertRaises(ValueError):validate_plan(self.p,self.i,self.e,200)

class OrderingTests(unittest.TestCase):
    def items(self):
        return [{"request_id":x,"prompt_hash":x,"prefill":x!="d","accepted_s":10} for x in ("p","d","q","u")]
    def test_decode_position(self):
        out,changed,why=stable_prefill_reorder(self.items(),["q","p"],11)
        self.assertEqual([x["request_id"] for x in out],["q","d","p","u"])
    def test_age_fifo(self):
        out,changed,why=stable_prefill_reorder(self.items(),["q"],20)
        self.assertFalse(changed);self.assertEqual(why,"age_fifo")
    def test_unknown_stable(self):
        out,_,_=stable_prefill_reorder(self.items(),[],11);self.assertEqual(out,self.items())

class WorkloadTests(unittest.TestCase):
    def test_counts_and_tokens(self):
        from joint18v2.workload import build
        ns=build(config(),list(range(100,150)),12,[103]*8)
        self.assertEqual(len(ns),6);self.assertEqual(sum(n["prompt"] for n in ns.values()),12928)
        self.assertEqual(sum(n["output"] for n in ns.values()),576)
    def test_context_guard(self):
        from joint18v2.workload import build
        c=config();c["max_model_len"]=2048
        with self.assertRaises(ValueError):build(c,list(range(100,150)),12,[103]*8)
    def test_edges_not_names_control(self):
        from joint18v2.workload import edges
        self.assertEqual(edges("AsymmetricAgent")[4],[1])
        self.assertEqual(edges("Sequential")[-1],[4])
    def test_arrivals_fixed_work(self):
        from joint18v2.workload import arrivals
        self.assertEqual(len(arrivals(64,4,"wave",2)),64)
        self.assertTrue(all(a<b for a,b in zip(arrivals(64,4,"wave",2),arrivals(64,4,"wave",2)[1:])))

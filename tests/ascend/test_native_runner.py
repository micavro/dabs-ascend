"""216-node native controller branch integration on a CPU double, NOT NPU evidence."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import time
import unittest

from dabs_ascend import r234
from dabs_ascend.common import DATASET, append_json, configuration
from dabs_ascend.runner import run_condition
from dabs_ascend.scheduler import snapshot
from test_native import resources, model
from test_port import FakeTokenizer
from test_r234 import SimulatedFixedBackend
from semantic19.dataset import load_public
from typed19.workflow import compile_packets


class NativeDouble:
    log = None
    async def drain(self, replicas):
        append_json(self.log, {'cpu_double': True, 'drained': True})
    async def harvest(self, replicas, **kwargs):
        return []
    async def observe(self, nodes, cfg, index, model):
        s = snapshot(nodes, {}, cfg['replicas'], cfg, index)
        s.update(native_snapshots=[{}]*cfg['replicas'], native_mutable=[],
            native_queue_groups={str(r): {'running_prefill': [], 'waiting': []} for r in range(cfg['replicas'])},
            physical_active_blocks=[0]*cfg['replicas'], capacity_blocks=[1000]*cfg['replicas'],
            native_observation_errors=['CPU double has no native evidence'])
        return s
    async def commit(self, action, state, nodes, fallback=False):
        for k in action['admit']:
            nodes[k]['native_action_id'] = 'cpu-only-' + k
        return r234.submission_order(action, nodes), []


class BackendDouble(SimulatedFixedBackend):
    def __init__(self):
        self.native = NativeDouble()
    async def generate(self, replica, node, seed, log, schema=None):
        result = await super().generate(replica, node, seed, log, schema)
        result['native_mapping'] = {'mappings': [{'action_id': node['native_action_id']}], 'cpu_double': True}
        return result


class NativeRunnerIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_all_methods_complete_216_nodes_without_changing_work_or_dependencies(self):
        cfg = configuration(4, resources())
        cfg.update(document_tokens_budget=2000, workflow_count=64, arrival_rate=1000,
                   epoch_s=.03, planning_soft_limit_s=.01, commit_reserve_s=.003, condition_wall_s=60.)
        tok = FakeTokenizer()
        rows, _ = load_public(DATASET, 'pilot')
        packets, _ = await compile_packets(rows, tok, cfg)
        packets, _ = r234.frozen('heterogeneous').select_packets(packets, 'balanced')
        fingerprints = set()
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(1) as pool:
            for spec in r234.plan()[:4]:
                summary = await run_condition(BackendDouble(), tok, packets, model(), cfg, spec,
                                              Path(directory) / spec['label'], pool)
                self.assertTrue(summary['functional_pass'])
                self.assertEqual((summary['completed_workflows'], summary['completed_requests'], summary['output_tokens']), (64, 216, 13184))
                fingerprints.add(summary['external_work_fingerprint'])
        self.assertEqual(len(fingerprints), 1)

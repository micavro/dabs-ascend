"""Runs the actual native suite transaction using its existing CPU-only mock.
Synthetic metrics stay in TemporaryDirectory; no GPU measurements are produced.
"""
import unittest
from unittest.mock import patch
import test_suite_orchestration as native_tests
from wfsched import acceptance,runner
from wfsched.common import load

class CaseNativeOrchestration(unittest.IsolatedAsyncioTestCase):
    async def test_full_case_pause_resume_and_report(self):
        fixture=native_tests.OrchestrationTests('test_pause_resume_all_methods');fixture.setUp()
        try:
            evidence=fixture.base/'selection.txt';evidence.write_text('synthetic evidence, not performance')
            registration=fixture.base/'selected.json'
            acceptance.register(['3:2:4'],evidence,'synthetic local test',registration)
            fixture.args.suite='case-confirm';fixture.args.max_conditions=4
            with patch.object(acceptance,'REGISTRATION',registration):
                await runner.run(fixture.args)
                self.assertEqual(load(fixture.args.out/'suite_status.json')['status'],'paused')
                self.assertEqual(len(load(fixture.args.out/'index.json')),4)
                fixture.args.resume=True;fixture.args.max_conditions=None
                await runner.run(fixture.args)
                self.assertEqual(load(fixture.args.out/'suite_status.json')['status'],'complete')
                p,rows,status=runner.read_run(fixture.args.out)
                self.assertEqual(len(rows),32)
                self.assertEqual({r['block'] for r in rows},{4})
                self.assertEqual({r['method'] for r in rows},{'DAG','Affinity','P2C','RR'})
                from wfsched.accept_report import write_report
                result=write_report(fixture.args.out)
                self.assertTrue(result.is_file())
                detail=load(result.parent/'evidence.json')
                self.assertEqual(len(detail['block_results']),3)
                self.assertTrue(all(b['pairs']==8 and b['complete_case_comparison'] for b in detail['block_results']))
        finally:fixture.tearDown()

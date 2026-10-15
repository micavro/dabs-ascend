"""Regression: never precreate RAW directory before native execute."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from wfsched.common import atomic, load


class ManifestDirectoryContract(TestCase):
    def test_atomic_original_behavior_and_two_stage_manifest_placement(self):
        # Reproduces why the initial V5 fails: atomic(...) creates RAW directory,
        # then the real native executor fails on mkdir(exist_ok=False).
        manifest={'revision':'test','profile':'skew'}
        with TemporaryDirectory() as td:
            attempt=Path(td)/'attempts'/'a'
            attempt.mkdir(parents=True,exist_ok=False)
            folder=attempt/'RAW_xyz'
            atomic(attempt/'predeclared_popularity_manifest.json',manifest)
            self.assertFalse(folder.exists())
            folder.mkdir(parents=True,exist_ok=False) # native execute begins
            atomic(folder/'predeclared_popularity_manifest.json',manifest)
            self.assertEqual(load(folder/'predeclared_popularity_manifest.json'),manifest)
            self.assertEqual(load(attempt/'predeclared_popularity_manifest.json'),manifest)

    def test_runner_source_does_not_atomic_precreate_raw(self):
        from wfsched import runner
        import inspect
        text=inspect.getsource(runner.run)
        pre=text.split('with bindings(cell[\'method\']):')[0]
        self.assertNotIn("atomic(folder/'predeclared_popularity_manifest.json'",pre)
        self.assertIn("atomic(attempt/'predeclared_popularity_manifest.json'",pre)
        post=text.split('with bindings(cell[\'method\']):')[1]
        self.assertIn("atomic(folder/'predeclared_popularity_manifest.json'",post)

"""Container startup contract; network and process replacement are boundary mocks."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ENTRYPOINT = Path(__file__).resolve().parents[1] / 'docker' / 'entrypoint.py'

class StartupTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ENTRYPOINT.is_file(), 'Container model-download entrypoint is missing')
        spec = importlib.util.spec_from_file_location('entrypoint', ENTRYPOINT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = Path(self.tmp.name)
        for name in ('config.json', 'decision_config.json', 'readout.safetensors'):
            (self.snapshot / name).write_text('fixture')

    def test_selected_model_snapshot_is_passed_to_server(self):
        env={'JEFF_MODEL_REPO':'example/custom-jeff','JEFF_MODEL_REVISION':'abc123','JEFF_MODEL_CACHE':'/models'}
        with patch.dict(os.environ, env, clear=True), patch.object(self.module, 'snapshot_download', return_value=str(self.snapshot)) as download, patch.object(self.module.os, 'execvpe') as execute:
            self.module.main(['/app/.venv/bin/jeff-serve'])
        self.assertEqual(download.call_args.kwargs, {'repo_id':'example/custom-jeff','revision':'abc123','cache_dir':'/models','token':False})
        self.assertEqual(execute.call_args.args[2]['JEFF_CHECKPOINT'], str(self.snapshot))

    def test_default_revision_does_not_pin_a_different_model(self):
        with patch.dict(os.environ, {'JEFF_MODEL_REPO':'example/other'}, clear=True), patch.object(self.module, 'snapshot_download', return_value=str(self.snapshot)) as download, patch.object(self.module.os,'execvpe'):
            self.module.main(['jeff-serve'])
        self.assertEqual(download.call_args.kwargs['revision'], 'main')

    def test_download_failure_does_not_start_server(self):
        with patch.object(self.module, 'snapshot_download', side_effect=RuntimeError('download interrupted')), patch.object(self.module.os, 'execvpe') as execute:
            with self.assertRaisesRegex(RuntimeError, 'download interrupted'):
                self.module.main(['jeff-serve'])
        execute.assert_not_called()

    def test_incomplete_checkpoint_does_not_start_server(self):
        (self.snapshot/'readout.safetensors').unlink()
        with patch.object(self.module,'snapshot_download', return_value=str(self.snapshot)), patch.object(self.module.os,'execvpe') as execute:
            with self.assertRaisesRegex(RuntimeError,'readout.safetensors'):
                self.module.main(['jeff-serve'])
        execute.assert_not_called()

    def test_custom_command_does_not_download_model(self):
        with patch.object(self.module, 'snapshot_download') as download, patch.object(self.module.os,'execvpe') as execute:
            self.module.main(['python','--version'])
        download.assert_not_called()
        self.assertEqual(execute.call_args.args[:2], ('python',['python','--version']))

if __name__ == '__main__':
    unittest.main()

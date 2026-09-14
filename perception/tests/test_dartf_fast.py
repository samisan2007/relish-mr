"""Host-only checks for the RTX 3080 handoff: no weights, GPU or Docker required."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

FAST_DIR = Path(__file__).resolve().parents[1] / 'dartf'
sys.path.insert(0, str(FAST_DIR))
import fast


class FastSetupTests(unittest.TestCase):
    def test_published_scales_survive_schema_conversion(self):
        scales = {f'block{b}.{fam}': (b + 1) / 1000
                  for b in range(32) for fam in ('qkv', 'proj', 'fc1', 'fc2')}
        calibrated = fast.calibration_from_scales(scales)
        for key, scale in scales.items():
            self.assertAlmostEqual(calibrated[f'calib.{key}']['p99999'] / 127, scale)
        for invalid in ({}, {**scales, 'block0.qkv': 0}, {**scales, 'block0.qkv': float('nan')}):
            with self.assertRaises(ValueError):
                fast.calibration_from_scales(invalid)

    def test_5070_is_rejected_before_build_and_3080_is_accepted(self):
        prop = SimpleNamespace(name='RTX 5070', major=12, minor=0, total_memory=12 << 30)
        cuda = SimpleNamespace(is_available=lambda: True, get_device_properties=lambda _: prop)
        with patch.dict(sys.modules, torch=SimpleNamespace(cuda=cuda), tensorrt=SimpleNamespace(__version__='10.11')):
            with self.assertRaisesRegex(RuntimeError, 'SM86.*RTX 5070'):
                fast.checked_gpu()
            prop.name, prop.major, prop.minor, prop.total_memory = 'RTX 3080', 8, 6, 10 << 30
            self.assertEqual(fast.checked_gpu()['memory_bytes'], 10 << 30)

    def test_cached_engines_cannot_move_to_a_different_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'gpu.json').write_text(json.dumps({'name': 'RTX 5070'}))
            with patch.object(fast, 'checked_gpu', return_value={'name': 'RTX 3080'}):
                with self.assertRaisesRegex(RuntimeError, 'another GPU'):
                    fast.check_target(out)

    def test_changed_recipe_cannot_silently_reuse_old_engines(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'recipe.sha256').write_text('old recipe')
            with patch.object(sys, 'argv', ['fast.py', '--out', directory]), \
                 patch.object(fast, 'check_target', return_value={}), \
                 patch.object(fast.subprocess, 'check_output', return_value=fast.DART_COMMIT), \
                 patch.dict(fast.os.environ), patch.object(fast, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'recipe changed'):
                    fast.main()
                run.assert_not_called()

    @unittest.skipUnless(sys.platform == 'win32', 'PowerShell launcher is for Windows')
    def test_launcher_preserves_spaces_and_literal_prompt_arguments(self):
        with tempfile.TemporaryDirectory(prefix='DARTF test ') as directory:
            video = Path(directory) / 'two pens.mp4'
            video.touch()
            prompt = 'pen $(do-not-execute)'
            result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                '-File', str(FAST_DIR / 'rtx3080.ps1'), '-Action', 'Test', '-Video', str(video),
                '-Prompt', prompt, '-WorkDir', str(Path(directory) / 'local files'), '-Render', '-DryRun'],
                capture_output=True, text=True, check=True)
            args = json.loads(result.stdout)
            self.assertEqual(args[args.index('--prompt') + 1], prompt)
            self.assertEqual(args[args.index('--video') + 1], '/input/two pens.mp4')
            self.assertIn('--render', args)
            self.assertIn('relish-dartf:sm86', args)
            self.assertFalse((Path(directory) / 'local files').exists())


if __name__ == '__main__':
    unittest.main()

"""Candidate comparisons must reject changed inputs and incomplete checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image

from candidates.benchmark import hf_frames, load_case, sha256, strict_image_checkpoint


class CandidateTests(unittest.TestCase):
    def test_session_reset_does_not_clear_shared_seed_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / '00000.jpg'
            Image.new('RGB', (8, 6)).save(frame)
            processor = Mock(return_value=SimpleNamespace(original_sizes=[(6, 8)]))
            session = processor.init_video_session.return_value
            def remember_ids(*args, obj_ids, **kwargs):
                session.reset_inference_session.side_effect = obj_ids.clear
            processor.add_inputs_to_inference_session.side_effect = remember_ids
            ids = [7, 11]
            iterator = hf_frames(None, processor, [frame], np.ones((2, 6, 8), dtype=bool), ids)
            next(iterator)
            iterator.close()
            self.assertEqual(ids, [7, 11])

    def test_shared_case_integrity_and_object_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'frames').mkdir()
            frames = {}
            for i in range(10):
                path = root / 'frames' / f'{i:05d}.jpg'
                Image.new('RGB', (8, 6)).save(path)
                frames[path.name] = sha256(path)
            masks = np.zeros((2, 6, 8), dtype=bool)
            masks[0, 1:3, 1:3] = True
            masks[1, 3:5, 5:7] = True
            np.savez_compressed(root / 'seeds.npz', ids=[7, 11], masks=masks)
            (root / 'case.json').write_text(json.dumps(dict(
                frames=frames, seeds_sha256=sha256(root / 'seeds.npz'))))
            _, files, selected, ids = load_case(root, 1)
            self.assertEqual((len(files), ids), (10, [7]))
            np.testing.assert_array_equal(selected, masks[:1])
            with self.assertRaisesRegex(ValueError, 'enough seed objects'):
                load_case(root, 3)
            with (root / 'seeds.npz').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Seed masks changed'):
                load_case(root, 1)

    def test_partial_checkpoint_cannot_produce_benchmark_results(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'weights.pt'
            model = torch.nn.Linear(2, 1)
            expected = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
            torch.save({'model': {f'detector.{k}': v for k, v in expected.items()}}, checkpoint)
            strict_image_checkpoint(model, checkpoint)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key])
            torch.save({'model': {'detector.weight': expected['weight']}}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                strict_image_checkpoint(model, checkpoint)


if __name__ == '__main__':
    unittest.main()

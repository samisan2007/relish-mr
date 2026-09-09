"""Tracking regressions without model downloads or camera access.

Run from perception/: .venv/Scripts/python.exe -m unittest discover -s tests -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from ultralytics.engine.predictor import BasePredictor

from sam3_runner import Instance
from yoloe_runner import HybridVideoTracker, YoloeVideoTracker


def detection():
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:12, 4:12] = True
    return Instance(mask, (4, 4, 12, 12), 0.9, obj_id=0)


class ColorTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = 255
        self.model = Mock()
        self.model.track.return_value = [SimpleNamespace(boxes=None)]
        self.sam = Mock()
        self.sam.track_frame.return_value = ([detection()], 1.0)

    def assert_model_sees_red(self, source):
        # Use the installed preprocessing contract, including its BGR -> RGB step.
        predictor = BasePredictor.__new__(BasePredictor)
        predictor.device = torch.device('cpu')
        predictor.model = SimpleNamespace(fp16=False)
        predictor.pre_transform = lambda frames: frames
        tensor = predictor.preprocess([source])
        self.assertEqual(tensor[0, :, 0, 0].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(self.rgb[0, 0].tolist(), [255, 0, 0])

    def test_text_tracker_preserves_model_colors(self):
        with patch('ultralytics.YOLOE', return_value=self.model):
            tracker = YoloeVideoTracker()
        session = tracker.start_stream('red object')
        tracker.track_frame(session, self.rgb)
        self.assert_model_sees_red(self.model.track.call_args.args[0])

    def test_hybrid_preserves_rgb_for_sam_and_model_colors_for_yolo(self):
        with patch('ultralytics.YOLOE', return_value=self.model):
            tracker = HybridVideoTracker(self.sam)
        session = tracker.start_stream('red object')
        tracker.track_frame(session, self.rgb)
        self.assertIs(self.sam.track_frame.call_args.args[1], self.rgb)
        tracker.track_frame(session, self.rgb)
        self.assert_model_sees_red(self.model.track.call_args.args[0])


class HybridHandoffTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        self.rgb[4:12, 4:12] = [255, 20, 10]
        self.model = Mock()
        self.model.track.return_value = [SimpleNamespace(boxes=None)]
        self.sam = Mock()
        self.sam.track_frame.return_value = ([detection()], 1.0)
        with patch('ultralytics.YOLOE', return_value=self.model):
            self.tracker = HybridVideoTracker(self.sam)
        self.session = self.tracker.start_stream('object')

    def test_moving_object_uses_original_seed_pixels_and_all_boxes(self):
        second = Instance(detection().mask, (16, 16, 24, 24), 0.8, obj_id=1)
        self.sam.track_frame.return_value = ([detection(), second], 1.0)
        original = self.rgb.copy()
        self.tracker.track_frame(self.session, self.rgb)
        self.model.track.assert_not_called()
        # Simulate both camera motion and reuse of the caller's frame buffer.
        self.rgb[:] = 0
        self.rgb[20:28, 20:28] = [255, 20, 10]
        self.tracker.track_frame(self.session, self.rgb)
        call = self.model.track.call_args
        np.testing.assert_array_equal(call.args[0], self.rgb[:, :, ::-1])
        np.testing.assert_array_equal(call.kwargs['refer_image'], original[:, :, ::-1])
        np.testing.assert_array_equal(
            call.kwargs['visual_prompts']['bboxes'], [[4, 4, 12, 12], [16, 16, 24, 24]]
        )
        np.testing.assert_array_equal(call.kwargs['visual_prompts']['cls'], [0, 0])

    def test_later_frames_reuse_embeddings_without_old_coordinates(self):
        for _ in range(4):
            self.tracker.track_frame(self.session, self.rgb)
        self.sam.track_frame.assert_called_once()
        self.assertEqual(self.model.track.call_count, 3)
        self.assertIn('refer_image', self.model.track.call_args_list[0].kwargs)
        for call in self.model.track.call_args_list[1:]:
            self.assertNotIn('refer_image', call.kwargs)
            self.assertNotIn('visual_prompts', call.kwargs)
            self.assertNotIn('predictor', call.kwargs)
            self.assertTrue(call.kwargs['persist'])

    def test_empty_seed_retries_sam_without_starting_yolo(self):
        self.sam.track_frame.side_effect = [([], 1.0), ([detection()], 1.0)]
        self.tracker.track_frame(self.session, self.rgb)
        self.tracker.track_frame(self.session, self.rgb)
        self.assertEqual(self.sam.track_frame.call_count, 2)
        self.model.track.assert_not_called()
        self.tracker.track_frame(self.session, self.rgb)
        self.assertIn('refer_image', self.model.track.call_args.kwargs)

    def test_failed_handoff_keeps_reference_for_retry(self):
        self.tracker.track_frame(self.session, self.rgb)
        self.model.track.side_effect = [RuntimeError('inference failed'), [SimpleNamespace(boxes=None)]]
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            self.tracker.track_frame(self.session, self.rgb)
        self.tracker.track_frame(self.session, self.rgb)
        for call in self.model.track.call_args_list:
            np.testing.assert_array_equal(call.kwargs['refer_image'], self.rgb[:, :, ::-1])

    def test_new_stream_grounds_and_embeds_a_new_reference(self):
        self.tracker.track_frame(self.session, self.rgb)
        self.tracker.track_frame(self.session, self.rgb)
        next_frame = np.full_like(self.rgb, 64)
        next_session = self.tracker.start_stream('different object')
        self.tracker.track_frame(next_session, next_frame)
        self.tracker.track_frame(next_session, next_frame)
        self.assertEqual(self.sam.track_frame.call_count, 2)
        np.testing.assert_array_equal(self.model.track.call_args.kwargs['refer_image'], next_frame)


if __name__ == '__main__':
    unittest.main()

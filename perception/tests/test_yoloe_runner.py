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


class FakeBoxes:
    def __init__(self, count):
        self.xyxy = torch.tensor([[4.0, 4.0, 12.0, 12.0]] * count)
        self.conf = torch.tensor([0.9] * count)
        self.id = torch.tensor([float(i) for i in range(count)])

    def __len__(self):
        return len(self.conf)


def yolo_result(count=1):
    """A stand-in for an ultralytics Result. Count 0 means YOLOE lost the object,
    which the hybrid treats as a reason to re-ground."""
    return SimpleNamespace(boxes=FakeBoxes(count) if count else None, masks=None)


class ColorTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = 255
        self.model = Mock()
        self.model.track.return_value = [yolo_result()]
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
        self.model.track.return_value = [yolo_result()]
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
        self.model.track.side_effect = [RuntimeError('inference failed'), [yolo_result()]]
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


class RegroundTests(unittest.TestCase):
    """Frozen exemplars lose the object on pose change, so the seeder is re-run on a
    schedule and whenever YOLOE comes back empty."""

    def setUp(self):
        self.rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        self.rgb[4:12, 4:12] = [255, 20, 10]
        self.model = Mock()
        self.model.track.return_value = [yolo_result()]
        self.sam = Mock()
        self.sam.track_frame.return_value = ([detection()], 1.0)
        with patch('ultralytics.YOLOE', return_value=self.model):
            self.tracker = HybridVideoTracker(self.sam, reground_every=3)
        self.session = self.tracker.start_stream('object')

    def prompted(self):
        return [c for c in self.model.track.call_args_list if 'refer_image' in c.kwargs]

    def test_interval_regrounds_from_the_current_frame(self):
        self.tracker.track_frame(self.session, self.rgb)  # seed
        moved = np.zeros_like(self.rgb)
        moved[20:28, 20:28] = [255, 20, 10]
        for _ in range(3):
            self.tracker.track_frame(self.session, moved)
        self.assertEqual(self.sam.track_frame.call_count, 2)
        # The refreshed exemplars must describe the frame they were found in, not the
        # original seed frame — that is the whole point of re-grounding.
        np.testing.assert_array_equal(self.prompted()[-1].kwargs['refer_image'], moved[:, :, ::-1])

    def test_a_missed_reground_does_not_retry_every_frame(self):
        self.tracker.track_frame(self.session, self.rgb)  # seed
        self.sam.track_frame.return_value = ([], 1.0)
        for _ in range(6):
            self.tracker.track_frame(self.session, self.rgb)
        self.assertEqual(self.sam.track_frame.call_count, 3)  # seed + two scheduled attempts
        self.assertEqual(len(self.prompted()), 1)  # nothing found, so the old exemplars stand

    def test_empty_yolo_frame_regrounds_immediately(self):
        self.tracker.track_frame(self.session, self.rgb)  # seed
        self.model.track.return_value = [yolo_result(0)]
        self.tracker.track_frame(self.session, self.rgb)
        self.assertEqual(self.sam.track_frame.call_count, 1)  # nothing lost yet
        self.tracker.track_frame(self.session, self.rgb)
        self.assertEqual(self.sam.track_frame.call_count, 2)  # lost last frame, don't wait
        self.assertEqual(len(self.prompted()), 2)

    def test_interval_is_off_by_default_but_a_lost_track_still_regrounds(self):
        # Re-grounding restarts YOLOE's track IDs, so the schedule is opt-in; losing
        # every detection costs no IDs, so that trigger is not.
        with patch('ultralytics.YOLOE', return_value=self.model):
            tracker = HybridVideoTracker(self.sam)
        session = tracker.start_stream('object')
        tracker.track_frame(session, self.rgb)  # seed
        for _ in range(5):
            tracker.track_frame(session, self.rgb)
        self.sam.track_frame.assert_called_once()
        self.model.track.return_value = [yolo_result(0)]
        tracker.track_frame(session, self.rgb)  # loses everything
        tracker.track_frame(session, self.rgb)
        self.assertEqual(self.sam.track_frame.call_count, 2)

    def test_seed_time_is_included_in_the_reported_frame_time(self):
        self.tracker.track_frame(self.session, self.rgb)  # seed
        self.sam.track_frame.return_value = ([detection()], 400.0)
        for _ in range(2):
            _, quiet_ms = self.tracker.track_frame(self.session, self.rgb)
        _, ground_ms = self.tracker.track_frame(self.session, self.rgb)
        self.assertGreater(ground_ms, quiet_ms + 399)


class WorkerSeedTests(unittest.TestCase):
    """A Docker-worker seeder (SAM 3.1) stays open for the stream, so it has a startup
    hook to cancel and a container to close."""

    def setUp(self):
        self.rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        self.model = Mock()
        self.model.track.return_value = [yolo_result()]
        self.seed_session = Mock()
        self.seeder = Mock()
        self.seeder.start_stream.return_value = self.seed_session
        self.seeder.track_frame.return_value = ([detection()], 1.0)
        with patch('ultralytics.YOLOE', return_value=self.model):
            self.tracker = HybridVideoTracker(self.seeder)

    def test_on_started_is_forwarded_only_when_given(self):
        # Sam3VideoTracker.start_stream has no such parameter, so passing None would break it.
        self.tracker.start_stream('object')
        self.assertNotIn('on_started', self.seeder.start_stream.call_args.kwargs)
        hook = Mock()
        self.tracker.start_stream('object', on_started=hook)
        self.assertIs(self.seeder.start_stream.call_args.kwargs['on_started'], hook)

    def test_reset_closes_the_seed_session(self):
        session = self.tracker.start_stream('object')
        self.tracker.track_frame(session, self.rgb)
        session.reset_inference_session()
        self.seed_session.reset_inference_session.assert_called_once_with()
        session.reset_inference_session()  # cleanup runs from two places; must not double-close
        self.seed_session.reset_inference_session.assert_called_once_with()


class DriftGuardTests(unittest.TestCase):
    """YOLOE matches the exemplar embedding, not the prompt, so it can lock onto a whole
    torso with confidence. Nothing downstream can tell that box is wrong."""

    def setUp(self):
        self.rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model = Mock()
        self.sam = Mock()
        # One small exemplar: a 8x8 box, area 64.
        self.sam.track_frame.return_value = ([detection()], 1.0)
        with patch('ultralytics.YOLOE', return_value=self.model):
            self.tracker = HybridVideoTracker(self.sam)
        self.session = self.tracker.start_stream('pen')

    def yolo_box(self, x2, y2):
        boxes = FakeBoxes(1)
        boxes.xyxy = torch.tensor([[0.0, 0.0, float(x2), float(y2)]])
        return [SimpleNamespace(boxes=boxes, masks=None)]

    def test_a_box_far_larger_than_the_exemplar_is_dropped_and_regrounds(self):
        self.model.track.return_value = self.yolo_box(8, 8)
        self.tracker.track_frame(self.session, self.rgb)  # seed, exemplar area 64
        self.model.track.return_value = self.yolo_box(60, 60)  # 3600, 56x the exemplar
        instances, _ = self.tracker.track_frame(self.session, self.rgb)
        self.assertEqual(instances, [])
        self.assertTrue(self.session['lost'])
        self.tracker.track_frame(self.session, self.rgb)
        self.assertEqual(self.sam.track_frame.call_count, 2)  # corrected itself

    def test_moderate_growth_survives(self):
        self.model.track.return_value = self.yolo_box(8, 8)
        self.tracker.track_frame(self.session, self.rgb)  # seed
        self.model.track.return_value = self.yolo_box(20, 20)  # 400, 6.2x — object approaching
        instances, _ = self.tracker.track_frame(self.session, self.rgb)
        self.assertEqual(len(instances), 1)
        self.assertFalse(self.session['lost'])

    def test_guard_can_be_disabled(self):
        with patch('ultralytics.YOLOE', return_value=self.model):
            tracker = HybridVideoTracker(self.sam, drift_scale=0)
        session = tracker.start_stream('pen')
        self.model.track.return_value = self.yolo_box(8, 8)
        tracker.track_frame(session, self.rgb)
        self.model.track.return_value = self.yolo_box(64, 64)
        instances, _ = tracker.track_frame(session, self.rgb)
        self.assertEqual(len(instances), 1)


class WebcamSliderTests(unittest.TestCase):
    """The re-ground interval is a UI slider, so a wrong position in the Gradio inputs
    list would silently leave the tracker on its default."""

    def test_slider_reaches_the_tracker_and_the_run_label(self):
        import video_ui
        model = Mock()
        model.track.return_value = [yolo_result()]
        with patch('ultralytics.YOLOE', return_value=model):
            hybrid = HybridVideoTracker(Mock())
        camera = Mock(isOpened=Mock(return_value=False))
        stream = video_ui.webcam_loop('knife', True, True, 0, video_ui.MODEL_HYBRID_31,
                                      video_ui.DEFAULT_CONFIG.name, {}, 45)
        with patch.object(video_ui, 'get_active_tracker',
                          return_value=(hybrid, video_ui.MODEL_HYBRID_31)),                 patch.object(video_ui.cv2, 'VideoCapture', return_value=camera):
            next(stream)  # stops at the camera, after the tracker is configured
            stream.close()
        self.assertEqual(hybrid.reground_every, 45)

    def test_slider_order_matches_the_handler(self):
        import inspect
        import video_ui
        params = list(inspect.signature(video_ui.webcam_loop).parameters)
        self.assertEqual(params.index('reground'), params.index('run_state') + 1)
        # smoke_sam31_ui.py and the SAM 3.1 tests call this with seven positional args.
        self.assertEqual(params[:7], ['prompt', 'show_masks', 'show_boxes', 'camera_index',
                                      'model_choice', 'config_name', 'run_state'])


if __name__ == '__main__':
    unittest.main()

"""Host-only checks of SAM 3.1 output conversion; no checkpoint or GPU needed."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam31.benchmark import to_instances
from sam31.worker import prune_memory
from sam31_runner import MODEL_SAM31, MODEL_SAM31_COMPILED, Sam31Runner, Sam31Session
from sam3_runner import Instance, SegmentResult


class Sam31IntegrationTests(unittest.TestCase):
    def test_worker_arguments_pin_offline_gpu_environment_and_compile_mode(self):
        with patch('sam31_runner.Path.is_file', return_value=True), \
                patch('sam31_runner.DartfSession.__init__', return_value=None) as start:
            session = Sam31Session('red tomato', True, True, 0.3)
        args = start.call_args.kwargs['worker_args']
        self.assertIn('relish-sam31:local', args)
        self.assertIn('HF_HUB_OFFLINE=1', args)
        self.assertIn('--compile', args)
        self.assertIn('--image-only', args)
        self.assertEqual(args[args.index('--prompt') + 1], 'red tomato')
        self.assertEqual(args[args.index('--threshold') + 1], '0.3')
        self.assertGreater(session.frame_timeout, 600)
        self.assertFalse(any('token' in arg.lower() for arg in args))

    def test_image_threshold_and_worker_cleanup_on_success_and_failure(self):
        from PIL import Image
        runner = Sam31Runner()
        image = Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8))
        instances = [Instance(np.ones((4, 6), dtype=bool), (0, 0, 6, 4), score, idx)
                     for idx, score in enumerate((0.4, 0.9))]
        with patch('sam31_runner.Sam31Session') as session, \
                patch.object(runner, 'track_frame', return_value=(instances, 12)):
            result = runner.segment(image, 'tomato', 0.5)
            self.assertEqual([i.obj_id for i in result.instances], [1])
            session.return_value.reset_inference_session.assert_called_once()
        with patch('sam31_runner.Sam31Session') as session, \
                patch.object(runner, 'track_frame', side_effect=RuntimeError('GPU failed')):
            with self.assertRaisesRegex(RuntimeError, 'GPU failed'):
                runner.segment(image, 'tomato')
            session.return_value.reset_inference_session.assert_called_once()

    def test_pruning_keeps_seed_and_recent_memory_in_every_view(self):
        def outputs():
            return {'cond_frame_outputs': {0: 'seed', 16: 'old', 32: 'recent'},
                    'non_cond_frame_outputs': {8: 'old', 39: 'recent', 40: 'current'}}
        tracker = {'first_ann_frame_idx': 0, 'output_dict': outputs(),
                   'output_dict_per_obj': {0: outputs()}, 'temp_output_dict_per_obj': {0: outputs()},
                   'consolidated_frame_inds': {'cond_frame_outputs': {0, 16, 32}, 'non_cond_frame_outputs': {8, 39, 40}},
                   'frames_already_tracked': {8: {}, 39: {}, 40: {}},
                   'mask_inputs_per_obj': {0: {0: 'seed', 16: 'old', 32: 'recent'}}}
        state = {'cached_frame_outputs': {40: 'mask'}, 'sam2_inference_states': [tracker],
                 'tracker_metadata': {'obj_id_to_sam2_score_frame_wise': {8: {}, 40: {}},
                                      'rank0_metadata': {'suppressed_obj_ids': {8: set(), 40: set()}}}}
        prune_memory(state, 40, history=16)
        for output in [tracker['output_dict'], *tracker['output_dict_per_obj'].values(),
                       *tracker['temp_output_dict_per_obj'].values()]:
            self.assertEqual(set(output['cond_frame_outputs']), {0, 32})
            self.assertEqual(set(output['non_cond_frame_outputs']), {39, 40})
        self.assertEqual(tracker['consolidated_frame_inds']['cond_frame_outputs'], {0, 32})
        self.assertEqual(set(tracker['frames_already_tracked']), {39, 40})
        self.assertEqual(set(tracker['mask_inputs_per_obj'][0]), {0, 32})
        self.assertEqual(set(state['tracker_metadata']['obj_id_to_sam2_score_frame_wise']), {40})
        self.assertEqual(state['cached_frame_outputs'], {})

    def test_ui_selection_and_file_failure_close_tracking_generator(self):
        import ui
        import video_ui
        from PIL import Image
        image = Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8))
        with patch.object(ui, 'Sam31Runner') as runner:
            runner.return_value.segment.return_value = SegmentResult([], 12)
            _, info = ui.run(image, 'tomato', 0.5, True, True, MODEL_SAM31_COMPILED)
            runner.assert_called_once_with(compile_model=True)
            self.assertIn('No "tomato"', info)
        closed = []
        def failing_track(*args):
            try:
                raise RuntimeError('worker failed')
                yield
            finally:
                closed.append(True)
        active = Mock(track=failing_track)
        progress = Mock(tqdm=lambda iterable, **kwargs: iterable)
        with patch.object(video_ui, 'get_active_tracker', return_value=(active, MODEL_SAM31)), \
                patch.object(video_ui, 'load_frames', return_value=([np.asarray(image)], 10)):
            path, info = video_ui.run_file('clip.mp4', 'tomato', 10, 1, True, True, MODEL_SAM31, progress)
        self.assertIsNone(path)
        self.assertIn('worker failed', info)
        self.assertEqual(closed, [True])

    def test_webcam_start_failure_and_stop_during_startup_release_camera(self):
        import video_ui
        camera = Mock(isOpened=Mock(return_value=True))
        runner = Sam31Runner()
        stream = video_ui.webcam_loop('tomato', True, True, 0, MODEL_SAM31, video_ui.DEFAULT_CONFIG.name, {})
        with patch.object(video_ui, 'get_active_tracker', return_value=(runner, MODEL_SAM31)), \
                patch.object(video_ui.cv2, 'VideoCapture', return_value=camera), \
                patch.object(runner, 'start_stream', side_effect=RuntimeError('Docker unavailable')):
            next(stream)  # Startup status is visible before blocking initialization.
            _, info, _ = next(stream)
            self.assertIn('Docker unavailable', info)
            stream.close()
        camera.release.assert_called_once()
        self.assertEqual(video_ui._webcam_runs, {})
        camera.reset_mock()
        started_session = Mock()
        def start(prompt, on_started):
            on_started(started_session)
            return started_session
        stream = video_ui.webcam_loop('tomato', True, True, 0, MODEL_SAM31, video_ui.DEFAULT_CONFIG.name, {})
        with patch.object(video_ui, 'get_active_tracker', return_value=(runner, MODEL_SAM31)), \
                patch.object(video_ui.cv2, 'VideoCapture', return_value=camera), \
                patch.object(runner, 'start_stream', side_effect=start):
            next(stream)
            video_ui.finalize_run({}, [])
            _, info, _ = next(stream)
            self.assertIn('Stopped during startup', info)
            stream.close()
        started_session.reset_inference_session.assert_called_once()
        camera.release.assert_called_once()


class Sam31OutputTests(unittest.TestCase):
    def test_normalized_xywh_becomes_pixel_xyxy_without_resizing_masks(self):
        mask = np.zeros((1, 720, 1280), dtype=bool)
        mask[0, 100:300, 200:400] = True
        output = {"out_obj_ids": np.array([7]), "out_probs": np.array([0.9]),
                  "out_binary_masks": mask,
                  "out_boxes_xywh": np.array([[200 / 1280, 100 / 720, 200 / 1280, 200 / 720]])}
        instance, = to_instances(output, 720, 1280)
        np.testing.assert_allclose(instance.box_xyxy, [200, 100, 400, 300])
        np.testing.assert_array_equal(instance.mask, mask[0])
        self.assertEqual(instance.obj_id, 7)

    def test_rejects_wrong_mask_dimensions_and_nonfinite_results(self):
        output = {"out_obj_ids": np.array([1]), "out_probs": np.array([0.8]),
                  "out_binary_masks": np.ones((1, 4, 6), dtype=bool),
                  "out_boxes_xywh": np.array([[0., 0., 1., 1.]])}
        with self.assertRaisesRegex(ValueError, "dimensions"):
            to_instances(output, 6, 4)
        output["out_probs"][0] = float('nan')
        with self.assertRaisesRegex(ValueError, "non-finite"):
            to_instances(output, 4, 6)

    def test_empty_output(self):
        self.assertEqual(to_instances({"out_obj_ids": np.zeros(0, dtype=int),
            "out_probs": np.zeros(0), "out_binary_masks": np.zeros((0, 4, 6), dtype=bool),
            "out_boxes_xywh": np.zeros((0, 4))}, 4, 6), [])


if __name__ == '__main__':
    unittest.main()

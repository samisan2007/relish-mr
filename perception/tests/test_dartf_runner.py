"""DARTF transport and cleanup regressions without Docker, TensorRT or a GPU.

Run from perception/: .venv/Scripts/python.exe -m unittest discover -s tests -v
"""

import io
from pathlib import Path
import queue
import struct
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from dartf_runner import DartfSession, DartfVideoTracker, FastSession
from dartf_worker import MAX_PACKET_BYTES, read_packet, write_packet


class FragmentedPipe(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 7))

    def write(self, data):
        return super().write(data[:7])


class PacketTests(unittest.TestCase):
    def test_partial_reads_and_writes_preserve_packet_boundaries(self):
        pipe = FragmentedPipe()
        frame = np.full((8, 12, 3), [255, 20, 10], dtype=np.uint8)
        write_packet(pipe, frame=frame)
        write_packet(pipe, ready=True)
        pipe.seek(0)
        np.testing.assert_array_equal(read_packet(pipe)['frame'], frame)
        self.assertTrue(read_packet(pipe)['ready'])
        self.assertIsNone(read_packet(pipe))

    def test_truncated_and_invalid_lengths_fail(self):
        pipe = io.BytesIO()
        write_packet(pipe, ready=True)
        payload = pipe.getvalue()
        for cut in (1, 3, 4, len(payload) - 1):
            with self.subTest(cut=cut), self.assertRaises(EOFError):
                read_packet(io.BytesIO(payload[:cut]))
        for size in (0, MAX_PACKET_BYTES + 1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                read_packet(io.BytesIO(struct.pack('<I', size)))
        with patch('dartf_worker.MAX_PACKET_BYTES', 1), self.assertRaises(ValueError):
            write_packet(io.BytesIO(), ready=True)

    def test_malformed_payload_and_pickle_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            read_packet(io.BytesIO(struct.pack('<I', 4) + b'junk'))
        pipe = io.BytesIO()
        write_packet(pipe, unsafe=np.array([object()], dtype=object))
        pipe.seek(0)
        with self.assertRaises(ValueError):
            read_packet(pipe)

    def test_closed_writer_does_not_spin(self):
        for written in (0, None):
            with self.subTest(written=written), self.assertRaises(OSError):
                write_packet(Mock(write=Mock(return_value=written)), ready=True)


def worker_process(*packets):
    output = io.BytesIO()
    for packet in packets:
        write_packet(output, **packet)
    output.seek(0)
    return Mock(stdin=io.BytesIO(), stdout=output)


class SessionTests(unittest.TestCase):
    def setUp(self):
        remove = patch('dartf_runner.subprocess.run')
        self.remove = remove.start()
        self.addCleanup(remove.stop)

    def test_fast_transport_preserves_rgb_pixels_and_boolean_masks(self):
        session = FastSession.__new__(FastSession)
        session.process = Mock(stdin=io.BytesIO())
        frame = np.full((480, 640, 3), [255, 20, 10], dtype=np.uint8)
        session._send_frame(frame)
        session.process.stdin.seek(0)
        packet = read_packet(session.process.stdin)
        decoded = cv2.cvtColor(cv2.imdecode(packet['png'], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(decoded, frame)
        masks = np.zeros((1, 480, 640), bool)
        masks[0, 40:90, 51:173] = True
        with patch.object(DartfSession, 'request', return_value={
                'packed_masks': np.packbits(masks, axis=-1), 'ids': np.array([3])}):
            np.testing.assert_array_equal(session.request(frame)['masks'], masks)

    def test_rgb_frame_and_two_instances_roundtrip(self):
        frame = np.full((8, 12, 3), [255, 20, 10], dtype=np.uint8)
        masks = np.zeros((2, 8, 12), dtype=bool)
        masks[0, 1:4, 2:5] = True
        masks[1, 3:7, 6:10] = True
        boxes = np.array([[2, 1, 5, 4], [6, 3, 10, 7]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        process = worker_process(
            dict(ready=True),
            dict(masks=masks, boxes=boxes, scores=scores, ids=np.array([3, 17])),
        )
        with patch('dartf_runner.subprocess.Popen', return_value=process):
            session = DartfSession(Path('dart'), Path('assets'), 'red pen')
        self.addCleanup(session.reset_inference_session)
        tracker = DartfVideoTracker.__new__(DartfVideoTracker)
        instances, elapsed_ms = tracker.track_frame(session, frame)
        process.stdin.seek(0)
        np.testing.assert_array_equal(read_packet(process.stdin)['frame'], frame)
        np.testing.assert_array_equal([instance.mask for instance in instances], masks)
        np.testing.assert_array_equal([instance.box_xyxy for instance in instances], boxes)
        np.testing.assert_allclose([instance.score for instance in instances], scores)
        self.assertEqual([instance.obj_id for instance in instances], [3, 17])
        self.assertGreaterEqual(elapsed_ms, 0)
        session.reset_inference_session()
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(session.log.closed)
        self.assertIsNone(session.process)
        process.kill.assert_not_called()

    def test_failed_readiness_closes_worker(self):
        process = worker_process(dict(ready=False))
        with patch('dartf_runner.subprocess.Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'initialization'):
                DartfSession(Path('dart'), Path('assets'), 'pen')
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        process.wait.assert_called_once()

    def test_startup_failure_includes_worker_stderr(self):
        process = worker_process()
        log = tempfile.TemporaryFile()
        log.write(b'TensorRT engine is incompatible with this GPU\n')
        with patch('dartf_runner.subprocess.Popen', return_value=process), \
                patch('dartf_runner.tempfile.TemporaryFile', return_value=log):
            with self.assertRaisesRegex(RuntimeError, 'incompatible with this GPU'):
                DartfSession(Path('dart'), Path('assets'), 'pen')
        self.assertTrue(log.closed)
        self.assertTrue(process.stdout.closed)

    def test_startup_timeout_stops_only_its_container(self):
        process = worker_process()
        process.wait.side_effect = [subprocess.TimeoutExpired('docker', 5), 0]
        with patch('dartf_runner.subprocess.Popen', return_value=process), \
                patch('dartf_runner.DartfSession.receive', side_effect=RuntimeError('timed out')), \
                patch('dartf_runner.subprocess.run') as remove:
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                DartfSession(Path('dart'), Path('assets'), 'pen')
        command = remove.call_args.args[0]
        self.assertEqual(command[:3], ['docker', 'rm', '--force'])
        self.assertEqual(len(command), 4)
        self.assertTrue(command[3].startswith('relish-dartf-'))
        process.kill.assert_called_once()
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)

    def test_blocked_frame_write_can_time_out_and_stop_container(self):
        session = DartfSession.__new__(DartfSession)
        session.process = worker_process()
        session.log = tempfile.TemporaryFile()
        session.name = 'relish-dartf-test'
        session.flags = 0
        session.failed = False
        session.results = queue.Queue()
        self.addCleanup(session.reset_inference_session)
        sending, release, finished = (threading.Event() for _ in range(3))

        def blocked_write(*args, **kwargs):
            sending.set()
            release.wait(2)
            finished.set()

        def timeout(**kwargs):
            self.assertTrue(sending.wait(1))
            self.assertFalse(finished.is_set())
            raise queue.Empty

        try:
            with patch('dartf_runner.write_packet', side_effect=blocked_write), \
                    patch.object(session.results, 'get', side_effect=timeout):
                with self.assertRaisesRegex(RuntimeError, 'did not respond'):
                    session.request(np.zeros((8, 12, 3), dtype=np.uint8))
            self.assertTrue(session.failed)
            session.reset_inference_session()
            self.assertEqual(self.remove.call_args.args[0],
                             ['docker', 'rm', '--force', 'relish-dartf-test'])
        finally:
            release.set()
            self.assertTrue(finished.wait(1))

    def test_invalid_instance_dimensions_are_rejected(self):
        session = Mock(process=Mock(stdin=io.BytesIO()))
        session.request.return_value = dict(
            masks=np.zeros((1, 8, 12), dtype=bool), boxes=np.empty((0, 4)),
            scores=np.ones(1), ids=np.array([1]),
        )
        tracker = DartfVideoTracker.__new__(DartfVideoTracker)
        with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
            tracker.track_frame(session, np.zeros((8, 12, 3), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()

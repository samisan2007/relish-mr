"""Keyframe matching without models: a re-detection keeps its track's ID.

Run from perception/: .venv/Scripts/python.exe -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from keyframe_hybrid import dedupe, match_masks


def _rect(x1, y1, x2, y2):
    m = np.zeros((100, 100), bool)
    m[y1:y2, x1:x2] = True
    return m


class MatchMasksTests(unittest.TestCase):
    def test_shifted_detections_keep_ids_and_new_one_is_unmatched(self):
        tracks = {7: _rect(10, 10, 30, 30), 9: _rect(60, 60, 80, 80)}
        dets = [_rect(62, 61, 82, 81), _rect(12, 11, 32, 31), _rect(40, 0, 50, 10)]
        matched, unmatched = match_masks(dets, tracks, min_iou=0.2)
        self.assertEqual(matched, {0: 9, 1: 7})
        self.assertEqual(unmatched, [2])

    def test_one_track_is_not_claimed_twice(self):
        tracks = {1: _rect(10, 10, 40, 40)}
        dets = [_rect(10, 10, 40, 40), _rect(12, 12, 40, 40)]
        matched, unmatched = match_masks(dets, tracks, min_iou=0.2)
        self.assertEqual(list(matched.values()), [1])
        self.assertEqual(len(unmatched), 1)

    def test_below_threshold_is_new(self):
        matched, unmatched = match_masks([_rect(0, 0, 10, 10)], {1: _rect(8, 8, 30, 30)}, 0.2)
        self.assertEqual((matched, unmatched), ({}, [0]))

    def test_partial_mask_inside_whole_object_matches(self):
        # EdgeTAM kept only the top of a pen that SAM3 now returns whole.
        matched, _ = match_masks([_rect(40, 10, 45, 90)], {3: _rect(40, 10, 45, 30)}, 0.3)
        self.assertEqual(matched, {0: 3})

    def test_dedupe_drops_fragment_of_higher_scoring_detection(self):
        whole, fragment, other = _rect(40, 10, 45, 90), _rect(40, 50, 45, 70), _rect(70, 10, 75, 90)
        self.assertEqual(dedupe([fragment, whole, other], [0.5, 0.9, 0.8], 0.6), [1, 2])


if __name__ == "__main__":
    unittest.main()

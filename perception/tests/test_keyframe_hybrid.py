"""Keyframe matching without models: a re-detection keeps its track's ID.

Run from perception/: .venv/Scripts/python.exe -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from keyframe_hybrid import dedupe, mask_iou, match_masks


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

    def test_track_spread_over_two_pens_gets_the_pen_its_neighbour_lacks(self):
        # Track 1 has spread over both touching pens; track 2 is still on pen 2 alone.
        pen1, pen2 = _rect(10, 10, 15, 90), _rect(16, 10, 21, 90)
        tracks = {1: pen1 | pen2, 2: pen2}
        matched, unmatched = match_masks([pen1, pen2], tracks, min_iou=0.3)
        self.assertEqual((matched, unmatched), ({0: 1, 1: 2}, []))

    def test_iou_dedupe_keeps_a_pen_inside_a_two_pen_track(self):
        pen1, pen2 = _rect(10, 10, 15, 90), _rect(16, 10, 21, 90)
        self.assertEqual(sorted(dedupe([pen1 | pen2, pen2], [1, 0], 0.6, overlap=mask_iou)), [0, 1])
        self.assertEqual(dedupe([pen2, _rect(16, 12, 21, 90)], [1, 0], 0.6, overlap=mask_iou), [0])

    def test_dedupe_drops_fragment_of_higher_scoring_detection(self):
        whole, fragment, other = _rect(40, 10, 45, 90), _rect(40, 50, 45, 70), _rect(70, 10, 75, 90)
        self.assertEqual(dedupe([fragment, whole, other], [0.5, 0.9, 0.8], 0.6), [1, 2])

    def test_duplicate_tracks_keep_live_then_older_id(self):
        # Tracks 2 and 5 sit on one pen, as do retired 1 and live 8; 3 is elsewhere.
        pen, near, other = _rect(40, 10, 45, 90), _rect(40, 12, 45, 88), _rect(70, 10, 75, 90)
        ids, retired = [5, 2, 3, 1, 8], {1}
        masks = [near, pen, other, _rect(10, 10, 20, 20), _rect(10, 10, 21, 21)]
        keep = dedupe(masks, [(t not in retired, -t) for t in ids], 0.6, overlap=mask_iou)
        self.assertEqual(sorted(ids[k] for k in keep), [2, 3, 8])


if __name__ == "__main__":
    unittest.main()

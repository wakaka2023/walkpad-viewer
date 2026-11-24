from pathlib import Path
import unittest

import numpy as np

from test_side_onsets import fixture
from side_onset_rules import grounded_descent_onset
from video_to_intervals import label_poses, parser

def add_early_hip_motion(f):
    retreat = -.25*np.maximum(f['t']-1.4, 0)**2
    f['base']['k'][:, 12, 0] = retreat
    f['groups'][2][:, 0, 0] = retreat
    f['groups'][1][:, 0, 0] -= retreat


class SideTorsoOnsetTests(unittest.TestCase):
    def test_real_torso_first_descent_is_kept_before_vertical_descent(self):
        f, e, args = fixture(horizontal=True)
        add_early_hip_motion(f)
        e['start_video_frame'] = 51
        onset, details = grounded_descent_onset(f, e, args)
        self.assertAlmostEqual(f['t'][onset], 1.4, delta=.06)
        self.assertLess(f['t'][onset], 1.8)
        self.assertEqual(details['side_onset_status'], 'grounded_raw_body_and_torso_descent')

    def test_small_preparation_drift_does_not_lock_the_old_onset(self):
        f, e, args = fixture(horizontal=True)
        x = .04*f['t']+.35*np.maximum(f['t']-1.4, 0)**2
        f['base']['k'][:, 6, 0] = x
        f['groups'][1][:, 0, 0] = x
        add_early_hip_motion(f)
        e['start_video_frame'] = 51
        original = f['base']['k'].copy()
        onset, details = grounded_descent_onset(f, e, args)
        self.assertAlmostEqual(f['t'][onset], 1.4, delta=.06)
        self.assertEqual(details['side_torso_status'], 'supported_raw_torso_change_point')
        np.testing.assert_equal(f['base']['k'], original)

    def test_mirrored_torso_motion_has_the_same_onset(self):
        f, e, args = fixture(horizontal=True)
        add_early_hip_motion(f)
        e['start_video_frame'] = 51
        first, _ = grounded_descent_onset(f, e, args)
        f['base']['k'][:, :, 0] *= -1
        f['groups'][1][:, :, 0] *= -1
        mirrored, _ = grounded_descent_onset(f, e, args)
        self.assertEqual(first, mirrored)

    def test_missing_raw_torso_retains_the_review_flag(self):
        f, e, args = fixture(horizontal=True)
        add_early_hip_motion(f)
        e['start_video_frame'] = 51
        f['base']['k'][85, 6, 0] = np.nan
        onset, details = grounded_descent_onset(f, e, args)
        self.assertEqual(onset, 50)
        self.assertEqual(details['side_onset_needs_review'], 1)

    def test_irregular_pts_preserve_torso_start_in_seconds(self):
        f, e, args = fixture(horizontal=True)
        g, _, _ = fixture(irregular=True, horizontal=True)
        add_early_hip_motion(f)
        add_early_hip_motion(g)
        e['start_video_frame'] = 51
        first, _ = grounded_descent_onset(f, e, args)
        second, _ = grounded_descent_onset(g, e, args)
        self.assertLess(abs(f['t'][first]-g['t'][second]), .04)

    def test_torso_adjustment_alone_does_not_start_the_squat(self):
        f, e, args = fixture(horizontal=True)
        e['start_video_frame'] = 51
        onset, details = grounded_descent_onset(f, e, args)
        self.assertAlmostEqual(f['t'][onset], 2.0, delta=.06)
        self.assertEqual(details['side_torso_status'], 'preparation_without_supported_lower_body_motion')

    def test_uploaded_video_excludes_entry_and_torso_preparation(self):
        path = Path(__file__).resolve().parents[1]/'fixtures/side_109_pose.npz'
        with np.load(path, allow_pickle=False) as data:
            args = parser().parse_args(['--output', 'unused'])
            events, frames, _, _ = label_poses(data, args)
        self.assertEqual(len(events), 10)
        self.assertGreater(events[0]['start_video_s'], 4.1)
        self.assertLess(events[0]['start_video_s'], 4.24)
        self.assertTrue(all(frames[i]['rep_id'] == 0 for i in range(90)))
        self.assertEqual(events[0]['side_onset_needs_review'], 0)


if __name__ == '__main__':
    unittest.main()

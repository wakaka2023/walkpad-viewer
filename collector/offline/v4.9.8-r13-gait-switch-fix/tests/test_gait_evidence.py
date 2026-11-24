import unittest

import numpy as np

from action_calibration import relative_cut
from action_features import pose_features
from action_rules import classify_raw
from gait_evidence import judge
from gait_phase import window_evidence
from test_action_boundaries import CONFIG, add_identity_camera, pose_data


class GaitEvidenceTests(unittest.TestCase):
    def test_faster_walking_is_not_running(self):
        t = np.arange(94) / 30
        for stride in (.8, 1.1, 1.5):
            x = 2*np.pi*stride*t
            apart = 46*np.sin(x)
            y = 6*np.sin(x)**2
            evidence = window_evidence(t, np.c_[apart, apart*.25, y, y*1.1], 100.)
            self.assertIsNotNone(evidence)
            self.assertEqual(evidence['activity'], 'gait_walk')

    def test_running_and_walking_can_have_the_same_step_frequency(self):
        t = np.arange(94) / 30
        x = 2*np.pi*1.25*t
        apart = 47*np.sin(x)
        for sign, tag in [(1,'gait_walk'), (-1,'running')]:
            y = sign*9*np.sin(x)**2
            outcome = window_evidence(t, np.c_[apart, apart*.2, y, y*.92], 100.)
            self.assertIsNotNone(outcome)
            self.assertEqual(outcome['activity'], tag)
            self.assertAlmostEqual(outcome['inferred_step_hz'], 2.5, delta=.1)
        rejected = window_evidence(t, np.c_[apart, apart*.2, np.zeros(len(t)), 7*np.sin(x)**2], 100.)
        self.assertIsNone(rejected)

    def test_partial_walk_preserves_main_activity_when_arms_move(self):
        for motion, has_seed in [('walking', True), ('walking', False), ('camera', True), ('stationary', True)]:
            with self.subTest(motion=motion, has_seed=has_seed):
                data = pose_data()
                t, pose = data['time_s'], data['keypoints']
                if motion != 'stationary':
                    pose[:, :, 0] += 60*t[:, None]
                pose[50 if has_seed else 0:, 13:17] = np.nan
                pose[:, 9, 1] += 15*np.sin(2*np.pi*t)
                f = pose_features(data, CONFIG)
                f['foot_speed'] = np.where(t < 2.4, .5, np.nan)
                f['camera_speed'] = np.zeros(len(t))
                camera = dict(source_indices=np.arange(len(t)), valid=np.ones(len(t), bool),
                              matrices=np.tile([[1., 0, 3. if motion=='camera' else 0.], [0., 1., 0.]], (len(t),1,1)))
                frames = [dict(activity='walking' if has_seed and t[i] < 2.5 else 'unknown',
                               action_event_id=0, activity_reason='test_seed') for i in range(len(t))]
                rows, _ = judge(f, camera, pose, frames, CONFIG)
                expected = ('walking' if has_seed else 'body_motion') if motion=='walking' else 'unknown'
                self.assertEqual(rows[61]['activity'], expected)
                self.assertNotIn('arm_state', rows[61])
                self.assertEqual(rows[61]['locomotion_partial'], int(motion=='walking'))

    def test_stationary_hand_gesture_keeps_body_state(self):
        for feet_visible, expected in [(True, 'standing'), (False, 'unknown')]:
            data = pose_data()
            t, pose = data['time_s'], data['keypoints']
            pose[:, 9, 1] += 35*np.sin(2*np.pi*t)
            if not feet_visible:
                pose[:, 13:17] = np.nan
            f = add_identity_camera(pose_features(data, CONFIG))
            legacy = classify_raw(f, CONFIG)[1]
            body = classify_raw(f, CONFIG, arm_semantics=False)[1]
            self.assertIn('arm_activity', {row['activity'] for row in legacy})
            self.assertEqual({row['activity'] for row in body if 1.<row['video_time_s']<7.}, {expected})

    def test_adaptation_requires_distinct_active_and_quiet_evidence(self):
        quiet = np.arange(80) < 40
        f = np.where(quiet, .04, .51)
        use = np.ones(len(f), bool)
        self.assertLess(relative_cut(f, quiet, use, .25), .25)
        self.assertGreater(relative_cut(1.9*f, quiet, use, .25), .25)
        self.assertEqual(relative_cut(f, quiet, quiet, .25), .25)
        self.assertEqual(relative_cut(np.full(len(f), .1), quiet, use, .25), .25)

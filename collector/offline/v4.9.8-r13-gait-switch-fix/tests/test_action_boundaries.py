import copy
import json
from pathlib import Path
import unittest

import numpy as np

import action_features as af
import action_pipeline as pipeline
import action_rules as rules
from test_individual_quality import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
CONFIG = pipeline.load_config()


def pose_data():
    t = np.arange(160) / 20
    k = np.full((len(t), 17, 3), np.nan)
    for joint, point in {0: [200, 40], 5: [170, 80], 6: [230, 80], 7: [160, 130],
                         8: [240, 130], 9: [160, 180], 10: [240, 180],
                         11: [180, 180], 12: [220, 180], 13: [180, 260],
                         14: [220, 260], 15: [180, 340], 16: [220, 340]}.items():
        k[:, joint, :2], k[:, joint, 2] = point, .99
    return dict(time_s=t, keypoints=k, width=480, height=400)


def add_identity_camera(f):
    n = len(f['time_s'])
    camera = dict(source_indices=np.arange(n), matrices=np.tile([[1., 0, 0], [0, 1., 0]], (n, 1, 1)),
                  valid=np.ones(n, bool))
    return af.camera_features(f, camera, CONFIG, 480, 400)


def cycle_data():
    t = np.arange(160) / 20
    shape = np.exp(-((t - 4) / .9) ** 2)
    return dict(time_s=t, knee_flexion=10 + 100 * shape, foot_speed=np.full(len(t), .03),
                hip_vertical_speed_raw=np.gradient(.6 * shape, t), stance_width=np.full(len(t), .9),
                knee_asymmetry=np.full(len(t), 20.))


class ActionBoundaryTests(unittest.TestCase):
    def test_real_source_pts_are_required(self):
        data = pose_data()
        data['time_s'][20] = data['time_s'][19]
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            af.pose_features(data, CONFIG)

    def test_rigid_camera_translation_does_not_create_activity(self):
        data = pose_data()
        n = len(data['time_s'])
        movement = np.c_[np.arange(n) * .5, np.arange(n) * .2]
        data['keypoints'][:, :, :2] += movement[:, None, :]
        f = af.pose_features(data, CONFIG)
        matrices = np.tile([[1., 0, .5], [0, 1., .2]], (n, 1, 1))
        camera = dict(source_indices=np.arange(n), matrices=matrices, valid=np.ones(n, bool))
        f = af.camera_features(f, camera, CONFIG, 480, 400)
        self.assertLess(np.nanmedian(f['hip_speed']), .01)
        self.assertLess(np.nanmedian(f['foot_speed']), .01)
        segments, _, events, _, _ = rules.classify(f, CONFIG)
        self.assertEqual(events, [])
        self.assertGreater(sum(s['duration_s'] for s in segments if s['activity'] == 'standing'), 7.)

    def test_invisible_person_is_unknown(self):
        data = pose_data()
        data['keypoints'][:] = np.nan
        f = add_identity_camera(af.pose_features(data, CONFIG))
        segments, frames, events, _, _ = rules.classify(f, CONFIG)
        self.assertEqual({s['activity'] for s in segments}, {'unknown'})
        self.assertEqual(events, [])
        self.assertTrue(all(row['needs_review'] for row in frames))

    def test_long_pose_gap_is_not_interpolated(self):
        data = pose_data()
        data['keypoints'][40:80] = np.nan
        f = add_identity_camera(af.pose_features(data, CONFIG))
        _, frames, _, _, _ = rules.classify(f, CONFIG)
        self.assertTrue(all(row['activity'] == 'unknown' for row in frames[42:78]))

    def test_grounded_flexion_needs_observed_hip_recovery(self):
        data = cycle_data()
        events, _ = rules.cycle_events(data, CONFIG)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['activity'], 'split_squat')
        self.assertLess(events[0]['start_video_s'], events[0]['peak_video_s'])
        self.assertGreater(events[0]['end_video_s'], events[0]['peak_video_s'])
        data['hip_vertical_speed_raw'][:] = 0
        events, rejected = rules.cycle_events(data, CONFIG)
        self.assertEqual(events, [])
        self.assertIn('no_coherent_hip_descent_and_recovery', [r['reason'] for r in rejected])

    def test_moving_feet_are_not_squat_cycles(self):
        data = cycle_data()
        data['foot_speed'][:] = 1.5
        events, rejected = rules.cycle_events(data, CONFIG)
        self.assertEqual(events, [])
        self.assertIn('moving_feet_not_supported_as_grounded_flexion', [r['reason'] for r in rejected])

    def test_small_knee_wobble_is_not_a_squat(self):
        data = cycle_data()
        data['knee_flexion'] = 8 + (data['knee_flexion'] - 10) * .1
        events, _ = rules.cycle_events(data, CONFIG)
        self.assertEqual(events, [])

    def test_timeline_is_continuous_and_each_frame_has_one_segment(self):
        data = pose_data()
        f = add_identity_camera(af.pose_features(data, CONFIG))
        segments, frames, _, boundaries, _ = rules.classify(f, CONFIG)
        self.assertEqual(len(frames), len(data['time_s']))
        self.assertEqual(segments[0]['start_video_frame'], 1)
        self.assertEqual(segments[-1]['end_video_frame'], len(frames))
        for left, right in zip(segments[:-1], segments[1:]):
            self.assertEqual(left['end_video_s'], right['start_video_s'])
            self.assertEqual(left['end_video_frame'] + 1, right['start_video_frame'])
        self.assertEqual(sum(s['end_video_frame'] - s['start_video_frame'] + 1 for s in segments), len(frames))
        self.assertEqual(boundaries[-1]['to_activity'], 'VIDEO_END')
        self.assertEqual(boundaries[-1]['video_frame'], len(frames) + 1)

    def test_imported_cache_requires_video_and_weight_identity(self):
        data = pose_data()
        data.update(source_sha256='video-A', model_sha256='weights-A')
        with self.assertRaisesRegex(ValueError, 'source hash'):
            pipeline.check_pose(data, 'video-B', 'weights-A')
        with self.assertRaisesRegex(ValueError, 'model hash'):
            pipeline.check_pose(data, 'video-A', 'weights-B')

    def test_camera_cache_signature_changes_invalidate_it(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache, stamp = root / 'cache.npz', root / 'stamp.json'
            cache.write_bytes(b'fixture')
            stamp.write_text(json.dumps({'source': 'A', 'pose': 'P', 'camera_hz': 10}), encoding='utf-8')
            self.assertTrue(pipeline.cache_matches(cache, stamp, {'source': 'A', 'pose': 'P', 'camera_hz': 10}))
            self.assertFalse(pipeline.cache_matches(cache, stamp, {'source': 'B', 'pose': 'P', 'camera_hz': 10}))
            self.assertFalse(pipeline.cache_matches(cache, stamp, {'source': 'A', 'pose': 'P', 'camera_hz': 5}))

    def test_runtime_config_cannot_accept_manual_boundaries(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'override.json'
            path.write_text(json.dumps({'manual_start': 9}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                pipeline.load_config(path)
            path.write_text(json.dumps({'minimum_bout_s': -1}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'positive'):
                pipeline.load_config(path)

    def test_classification_does_not_change_pose_samples(self):
        data = pose_data()
        before = copy.deepcopy(data)
        f = add_identity_camera(af.pose_features(data, CONFIG))
        rules.classify(f, CONFIG)
        np.testing.assert_array_equal(data['time_s'], before['time_s'])
        np.testing.assert_array_equal(data['keypoints'], before['keypoints'])


if __name__ == '__main__':
    unittest.main()

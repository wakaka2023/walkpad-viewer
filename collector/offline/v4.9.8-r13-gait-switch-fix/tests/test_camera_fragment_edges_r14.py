import unittest

import numpy as np

from multi_action_analysis import camera_for_fragment
from running_turn import _world_xy


class CameraFragmentEdgesTest(unittest.TestCase):
    def setUp(self):
        self.t = np.cumsum([0., .032, .031, .032, .039, .034, .029, .037, .032, .041, .037, .036, .034])
        indices = np.arange(0, len(self.t), 3)
        times = self.t[indices]
        mats = np.tile(np.eye(2, 3), (len(indices), 1, 1))
        mats[1:, 0, 2] = 96 * np.diff(times)
        self.camera = dict(source_indices=indices, time_s=times, matrices=mats,
                           valid=np.r_[False, np.ones(len(indices) - 1, bool)])
        self.world = np.array([[30., 20.], [34., 21.]])
        self.image = np.broadcast_to(self.world, (len(self.t), 2, 2)).copy()
        self.image[:, :, 0] += 96 * self.t[:, None]

    def compensate(self, camera, lo=2, hi=8, *, bracket=True):
        subset = camera_for_fragment(camera, lo, hi, bracket=bracket)
        return _world_xy(self.t[lo:hi], self.image[lo:hi], subset, 0, hi-lo)

    def test_sparse_camera_samples_bracket_both_fragment_edges(self):
        subset = camera_for_fragment(self.camera, 2, 8, bracket=True)
        np.testing.assert_array_equal(subset['source_indices'], [-2, 1, 4, 7])
        np.testing.assert_array_equal(subset['time_s'], self.t[[0, 3, 6, 9]])
        self.assertIsNone(self.compensate(self.camera, bracket=False))
        np.testing.assert_allclose(self.compensate(self.camera),
                                   np.broadcast_to(self.world, (6, 2, 2)))

    def test_fragment_can_lie_entirely_between_sparse_camera_samples(self):
        out = self.compensate(self.camera, 1, 3)
        np.testing.assert_allclose(out, np.broadcast_to(self.world, (2, 2, 2)))

    def test_reject_invalid_transform_at_either_edge(self):
        for index in (1, 3):
            with self.subTest(index=index):
                valid = self.camera['valid'].copy()
                valid[index] = False
                self.assertIsNone(self.compensate(dict(self.camera, valid=valid)))

    def test_never_extrapolate_when_right_bracket_missing(self):
        incomplete = {k: v[:3] for k, v in self.camera.items()}
        self.assertIsNone(self.compensate(incomplete))

    def test_pose_feature_window_uses_only_in_fragment_samples(self):
        subset = camera_for_fragment(self.camera, 2, 8)
        np.testing.assert_array_equal(subset['source_indices'], [1, 4])
        self.assertNotIn('time_s', subset)


if __name__ == '__main__':
    unittest.main()

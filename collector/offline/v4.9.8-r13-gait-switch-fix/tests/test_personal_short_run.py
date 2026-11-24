import unittest

import numpy as np

from personal_short_run import detect


class PersonalShortRunTests(unittest.TestCase):
    def example(self, *, turn=False, arms_only=False):
        n = 160
        t = np.arange(n, dtype=float) / 50
        k = np.arange(n) // 20
        phase = (np.arange(n) % 20) / 20
        walked = np.isin(k, (1, 2))
        ran = np.isin(k, (3, 4))
        target = k == 5
        fast_walk = k == 6
        pose = np.zeros((n, 17, 3))
        valid = np.ones((n, 17), bool)
        scale = np.full(n, 100.)
        amplitude = np.where(ran | (target & (not arms_only)) | fast_walk, 18., 4.)
        shift = 3 * np.arange(n, dtype=float)
        if turn:
            shift[target] = 300 + 20 * np.minimum(phase[target], 1 - phase[target])
        trunk = amplitude * np.sin(np.pi * phase) ** 2
        spread = np.where(ran | (target & (not arms_only)) | fast_walk, 45., 20.)
        cos = np.where(ran, .15, np.where(target, -.4, -.9))
        for sign, s, e, w, h, a in ((-1, 5, 7, 9, 11, 15), (1, 6, 8, 10, 12, 16)):
            pose[:, s, 0] = shift + sign * 7
            pose[:, s, 1] = trunk + 20
            pose[:, h, :2] = pose[:, s, :2] + (0., 25.)
            pose[:, e, :2] = pose[:, s, :2] + (0., 10.)
            pose[:, w, 0] = pose[:, e, 0] + 10 * np.sqrt(1 - cos**2)
            pose[:, w, 1] = pose[:, e, 1] - 10 * cos
            pose[:, a, 0] = shift + sign * spread / 2
            pose[:, a, 1] = trunk + 65.
        steps = [(i, (-1)**s) for s, i in enumerate(range(10, n, 20))]
        unresolved = target | fast_walk
        return dict(t=t, pose=pose, valid=valid, scale=scale, batches=[(t - .25 / 50, steps)], walked=walked,
                    ran=ran, unresolved=unresolved), target

    def test_same_person_short_run_only_when_cues_agree(self):
        args, want = self.example()
        np.testing.assert_array_equal(detect(**args), want)

    def test_relative_decision_is_invariant_to_time_units_and_image_zoom(self):
        args, want = self.example()
        for factor in (.34, 1.0, 2.3):
            t = args['t'] * factor + 5
            np.testing.assert_array_equal(detect(**dict(args, t=t, batches=[(args['batches'][0][0] * factor + 5, args['batches'][0][1])],
                                                       scale=args['scale']*1.7,pose=args['pose']*1.7)), want)

    def test_turn_and_arm_only_are_not_running(self):
        for kwargs in (dict(turn=True),dict(arms_only=True)):
            args, _ = self.example(**kwargs)
            self.assertFalse(detect(**args).any())

    def test_missing_reference_or_unobserved_joint_abstains(self):
        for key in ('walked', 'ran'):
            args, _ = self.example()
            args[key][:] = False
            self.assertFalse(detect(**args).any())
        args, target = self.example()
        args['valid'][target, 15] = False
        self.assertFalse(detect(**args).any())

    def test_only_unresolved_frames_can_change(self):
        args, target = self.example()
        args['unresolved'][102:105] = False
        target[102:105] = False
        np.testing.assert_array_equal(detect(**args), target)

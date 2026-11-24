import unittest

import numpy as np

from action_temporal import stabilize
from multi_labels import LABELS, review
from running_turn import apply


class RunningTurnTest(unittest.TestCase):
    def case(self, *, fps=30, clock=1., turning=True, independently_walk=False,
             missing=False, invalid_camera=False, moving_camera=False, stopped=False,
             hips_only=False, stretch=1.):
        t = np.arange(4 * fps) / fps * clock
        a, b = np.searchsorted(t, np.array([1.5, 1.5 + .75 * stretch]) * clock)
        m = np.ones(len(t), bool)
        p = np.zeros((len(t), 17, 3))
        p[:, :, 2] = 1
        pivot = (t[a] + t[b]) / 2
        path = 40 * (np.abs(t - pivot) if turning else t) / clock
        for k, y in [(5, 0), (6, 0), (11, 40), (12, 40), (15, 90), (16, 90)]:
            p[:, k, 0] = path if k not in (15, 16) or not hips_only else 40 * t / clock
            p[:, k, 1] = y + np.sin(2 * np.pi * t / (.36 * clock))
        matrix = np.tile([[1., 0., 0.], [0., 1., 0.]], (len(t), 1, 1))
        cvalid = np.ones(len(t), bool)
        cvalid[0] = False  # no prior frame at first sample
        if moving_camera:
            camera_x = -55 * (t / clock - np.abs(t - pivot) / clock)
            p[:, :, 0] += camera_x[:, None]
            matrix[1:, 0, 2] = np.diff(camera_x)
        if missing:
            m[(a + b) // 2] = False
        if invalid_camera:
            cvalid[(a + b) // 2] = False
        r = [dict(activity='running', frame_end_s=x + clock / fps,
                  personal_gait_step_period_s=.36 * clock) for x in t]
        for row in r[a:b]:
            row.update(activity='standing' if stopped else 'walking',
                       gait_evidence='person_local_cycle_morphology',
                       pre_cycle_gait_evidence='walking_support_handoff' if independently_walk
                       else 'translation_with_step_motion_but_unresolved_gait')
        apply(t, p, p[:, :, 2] > 0, r, m,
              dict(source_indices=np.arange(len(t)), matrices=matrix, valid=cvalid))
        return t, a, b, r

    def test_reversal_across_frame_rate_and_relative_timing(self):
        for fps, clock in [(24, .8), (30, 1.), (60, 1.5)]:
            with self.subTest(fps=fps, clock=clock):
                t, a, b, rows = self.case(fps=fps, clock=clock)
                self.assertTrue(all(r['activity'] == 'running_turn' for r in rows[a:b]))
                self.assertTrue(all(r['pre_turn_activity'] == 'walking' for r in rows[a:b]))
                self.assertTrue(a < np.searchsorted(t, rows[a]['turn_pivot_video_s']) < b)
                original = [r['activity'] for r in rows]
                labels, inferred, _ = stabilize(t, original, np.ones(len(t), bool), [], {}, preserve_walk=True)
                self.assertListEqual(labels.tolist(), original)
                self.assertFalse(inferred.any())
                self.assertEqual(review('running_turn', rows[a], False, False, {}, a),
                                 dict(annotation=LABELS['running_turn'], activity_status='transition', action_candidates=''))

    def test_keep_independent_walk_and_standing(self):
        for kwargs in (dict(independently_walk=True), dict(stopped=True), dict(stretch=2.)):
            with self.subTest(kwargs=kwargs):
                _, a, b, r = self.case(**kwargs)
                self.assertTrue(all(q['activity'] == ('standing' if kwargs.get('stopped') else 'walking') for q in r[a:b]))

    def test_reject_unsupported_or_camera_only_reversal(self):
        for kwargs in (dict(turning=False), dict(missing=True), dict(invalid_camera=True),
                       dict(hips_only=True), dict(turning=False, moving_camera=True)):
            with self.subTest(kwargs=kwargs):
                _, a, b, r = self.case(**kwargs)
                self.assertTrue(all(q['activity'] == 'walking' for q in r[a:b]))


if __name__ == '__main__':
    unittest.main()

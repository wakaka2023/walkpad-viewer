import unittest

import numpy as np

from bilateral_gait_continuity import apply, EVIDENCE


def scene(speed=1.):
    t = np.arange(0., 22., 1 / 30.) / speed
    ends = t + 1 / (30 * speed)
    p = np.ones((len(t), 17, 3), dtype=float)
    x = t * speed
    p[:, [5, 11, 13, 15], 0] = 43.
    p[:, [6, 12, 14, 16], 0] = 57.
    p[:, [5, 6], 1] = 30.
    p[:, [11, 12], 1] = 70.
    p[:, [13, 14], 1] = 105.
    p[:, [15, 16], 1] = 140.
    for a, b in ((4., 13.), (16., 21.)):
        i = (x >= a) & (x < b)
        wave = 6 * np.sin(2 * np.pi * (x[i] - a) / 1.5)
        p[i, 15, 1] += wave
        p[i, 16, 1] -= wave
    valid = np.ones((len(t), 17), bool)
    measured = np.ones(len(t), bool)
    return t, ends, p, valid, measured, x


def row(kind, step=None, **extra):
    return dict(activity=kind, action_event_id=0, personal_gait_step_period_s=step, **extra)


class BilateralGaitTest(unittest.TestCase):
    def test_walk_without_torso_travel_preserves_pause_at_different_cadences(self):
        for speed in (.7, 1., 1.6):
            with self.subTest(speed=speed):
                t,e,p,v,m,x = scene(speed)
                labels = [row('gait_walk', .75 / speed) if 7 <= j < 10 or 11 <= j < 13 or 16 <= j < 21 else
                          row('body_motion') if 10 <= j < 11 else row('standing') for j in x]
                apply(t,e,p,v,labels,m)
                self.assertTrue(all(r['activity']=='gait_walk' for j,r in zip(x,labels) if 5 <= j < 7 or 10 <= j < 11))
                self.assertTrue(all(r['activity']=='standing' for j,r in zip(x,labels) if j < 3.5 or 13.1 <= j < 15.9))
                self.assertTrue(any(r.get('activity_reason') == EVIDENCE for r in labels))

    def test_running_gap_and_protected_evidence(self):
        t,e,p,v,m,x = scene()
        labels = [row('running', .75) if 4 <= j < 13 and not 9 <= j < 9.3 else row('unknown') for j in x]
        labels[np.searchsorted(x,3.7)].update(action_event_id=2)
        apply(t,e,p,v,labels,m)
        self.assertTrue(all(r['activity']=='running' for j,r in zip(x,labels) if 9 <= j < 9.3))
        self.assertTrue(all(r['activity']=='unknown' for j,r in zip(x,labels) if j < 4 or j >= 13))
        labels = [row('gait_walk', .75) if 4 <= j < 13 and not 9 <= j < 9.3 else row('unknown') for j in x]
        invalid = (x >= 9) & (x < 9.3)
        v[invalid, 15] = False
        apply(t,e,p,v,labels,m)
        self.assertTrue(all(r['activity']=='unknown' for i,r in zip(invalid,labels) if i))


if __name__ == '__main__':
    unittest.main()

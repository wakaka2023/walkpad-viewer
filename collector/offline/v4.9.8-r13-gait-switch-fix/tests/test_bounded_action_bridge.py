import unittest
import numpy as np
from action_pipeline import load_config
from bounded_action_bridge import bridge, WALK_CANDIDATE_EVIDENCE


class BoundedContinuity(unittest.TestCase):
    def setUp(self):
        n = 40
        self.t = np.arange(n) / 20
        self.ends = self.t + .05
        self.labels = np.full(n, 'walking', object)
        self.labels[16:22] = 'unknown'
        self.rows = [dict(activity=k, action_event_id=0) for k in self.labels]
        self.inferred = np.zeros(n, bool)
        self.protected = np.zeros(n, bool)
        self.reviews = [dict(activity_status='other' if k == 'unknown' else 'supported', action_candidates='') for k in self.labels]
        self.measured = np.ones(n, bool)
        self.visible = np.ones(n, bool)
        self.cfg = load_config()

    def evaluate(self):
        return bridge(self.t, self.ends, self.labels, self.rows, self.inferred, self.protected,
                      self.reviews, self.measured, self.visible, self.cfg)

    def candidate(self):
        for i in range(11,16):
            self.rows[i]['gait_evidence'] = WALK_CANDIDATE_EVIDENCE
            self.reviews[i].update(activity_status='candidate', action_candidates='walking')
        for i in range(16,22):
            self.rows[i].update(activity='body_motion', gait_evidence='gait_type_unresolved')

    def test_two_direct_sides_only_fill_middle(self):
        out = self.evaluate()
        self.assertEqual(list(np.flatnonzero(out != '')),list(range(16,22)))
        self.assertEqual(set(out[16:22]), {'short_other_same_direct_action'})
        self.assertEqual(set(self.labels[16:22]), {'unknown'})
        self.assertFalse(self.inferred.any())

    def test_unresolved_moving_gait_with_one_direct_side(self):
        self.candidate()
        self.assertEqual(set(self.evaluate()[16:22]), {'short_unresolved_motion_with_walking_context'})

    def test_quiet_unknown_with_only_one_direct_side_is_not_walked_over(self):
        self.candidate()
        for i in range(16,22): self.rows[i].update(activity='unknown', gait_evidence='')
        self.assertFalse(any(self.evaluate()))

    def test_actual_pause_or_action_change_is_not_erased(self):
        for k in ('standing','stairs_down','bending'):
            self.rows[19]['activity'] = k
            self.assertFalse(any(self.evaluate()))
        self.rows[19]['activity'] = 'unknown'
        self.labels[22:] = 'running'
        self.assertFalse(any(self.evaluate()))

    def test_evidence_for_second_action_or_protected_events_blocks_fill(self):
        self.rows[19]['action_hypotheses'] = ('squat',)
        self.assertFalse(any(self.evaluate()))
        self.rows[19].pop('action_hypotheses')
        self.rows[19]['action_event_id'] = 1
        self.assertFalse(any(self.evaluate()))

    def test_lost_frame_inferred_anchor_or_hidden_torso_blocks_fill(self):
        for array,i,val in ((self.measured,18,False),(self.visible,20,False),(self.inferred,15,True)):
            initial=array[i]; array[i]=val
            self.assertFalse(any(self.evaluate())); array[i]=initial

    def test_uses_pts_to_reject_long_gap(self):
        self.t[17:] += .31; self.ends[16:] += .31
        self.assertFalse(any(self.evaluate()))

    def test_squat_is_not_extended_or_counted(self):
        self.labels[self.labels=='walking'] = 'squat'
        for r in self.rows:
            if r['activity']=='walking':r['activity']='squat'
        self.assertFalse(any(self.evaluate()))

if __name__ == '__main__': unittest.main()

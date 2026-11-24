import unittest

import numpy as np

import action_pipeline as pipeline
import action_rules as rules
import action_temporal as temporal


CONFIG = pipeline.load_config()


def bend_data():
    t = np.arange(240) / 20
    signal = sum(.72 * np.exp(-((t - center) / .38) ** 2) for center in [3., 5., 7.])
    f = dict(time_s=t, local_torso_ratio=1 - signal, foot_speed=np.full(len(t), .03),
             knee_flexion=np.full(len(t), 15.), camera_speed=np.full(len(t), .002))
    hidden = (t >= 4.4) & (t <= 5.6)
    f['foot_speed'][hidden] = np.nan
    f['knee_flexion'][hidden] = np.nan
    return f


class HierarchicalActivityTests(unittest.TestCase):
    def test_motion_intensity_changes_do_not_create_new_actions(self):
        t = np.arange(300) / 30
        raw = np.array(['stepping'] * len(t), dtype=object)
        raw[60:140], raw[180:230] = 'fast_stepping', 'fast_stepping'
        labels, inferred, _ = temporal.stabilize(t, raw, np.ones(len(t), bool), [], CONFIG)
        self.assertEqual(set(labels), {'stepping'})
        self.assertFalse(inferred.any())

    def test_real_walking_to_stationary_stepping_transition_is_retained(self):
        t = np.arange(300) / 30
        raw = np.array(['walking'] * 150 + ['stepping'] * 150, dtype=object)
        labels, _, _ = temporal.stabilize(t, raw, np.ones(len(t), bool), [], CONFIG)
        np.testing.assert_array_equal(labels, raw)

    def test_short_gap_is_marked_as_inferred_not_observed(self):
        t = np.arange(200) / 20
        raw = np.array(['body_motion'] * len(t), dtype=object)
        raw[70:86] = 'unknown'
        usable = np.ones(len(t), bool)
        usable[70:86] = False
        labels, inferred, _ = temporal.stabilize(t, raw, usable, [], CONFIG)
        self.assertEqual(set(labels), {'body_motion'})
        self.assertTrue(inferred[70:86].all())
        self.assertFalse(inferred[:70].any())

    def test_long_unobserved_gap_is_never_filled(self):
        t = np.arange(200) / 20
        raw = np.array(['body_motion'] * len(t), dtype=object)
        raw[70:130] = 'unknown'
        labels, inferred, _ = temporal.stabilize(t, raw, raw != 'unknown', [], CONFIG)
        self.assertTrue((labels[70:130] == 'unknown').all())
        self.assertFalse(inferred[70:130].any())

    def test_no_action_is_invented_before_or_after_observed_activity(self):
        t = np.arange(200) / 20
        raw = np.array(['walking'] * len(t), dtype=object)
        raw[:10] = raw[-10:] = 'unknown'
        labels, _, _ = temporal.stabilize(t, raw, raw != 'unknown', [], CONFIG)
        self.assertTrue((labels[:10] == 'unknown').all())
        self.assertTrue((labels[-10:] == 'unknown').all())

    def test_cycle_group_keeps_quiet_recovery_inside_same_activity(self):
        t = np.arange(200) / 20
        raw = np.array(['standing'] * len(t), dtype=object)
        raw[40:70] = raw[80:110] = 'split_squat'
        events = [dict(event_id=1, activity='split_squat', start_video_frame=41, end_video_frame=70),
                  dict(event_id=2, activity='split_squat', start_video_frame=81, end_video_frame=110)]
        labels, _, protected = temporal.stabilize(t, raw, np.ones(len(t), bool), events, CONFIG)
        self.assertTrue((labels[40:110] == 'split_squat').all())
        self.assertTrue(protected[40:110].all())
        self.assertEqual(labels[39], 'standing')
        self.assertEqual(labels[110], 'standing')

    def test_walking_between_cycles_is_not_hidden_inside_exercise_group(self):
        t = np.arange(200) / 20
        raw = np.array(['standing'] * len(t), dtype=object)
        raw[70:90] = 'walking'
        events = [dict(event_id=1, activity='squat', start_video_frame=41, end_video_frame=70),
                  dict(event_id=2, activity='squat', start_video_frame=91, end_video_frame=120)]
        labels, _, _ = temporal.stabilize(t, raw, np.ones(len(t), bool), events, CONFIG)
        self.assertTrue((labels[70:90] == 'walking').all())

    def test_context_supported_partial_bend_is_recovered(self):
        events = rules.bend_events(bend_data(), CONFIG)
        self.assertEqual(len(events), 3)
        middle = events[1]
        self.assertAlmostEqual(middle['peak_video_s'], 5., delta=.1)
        self.assertEqual(middle['lower_body_support'], 'partial_observations_with_two_confirmed_neighbor_cycles')
        self.assertLess(middle['foot_observed_fraction'], .5)

    def test_isolated_partial_torso_shortening_is_not_called_bending(self):
        t = np.arange(240) / 20
        visible = (t < 2) | (t >= 10)
        f = dict(time_s=t, local_torso_ratio=1 - .72 * np.exp(-((t - 6) / 1.2) ** 2),
                 foot_speed=np.where(visible, .03, np.nan), knee_flexion=np.where(visible, 15., np.nan),
                 camera_speed=np.full(len(t), .002))
        events, rejected = rules.bend_events(f, CONFIG, return_rejected=True)
        self.assertEqual(events, [])
        self.assertIn('partial_bend_without_consistent_confirmed_neighbors', [row['reason'] for row in rejected])

    def test_camera_motion_cannot_supply_bend_context(self):
        f = bend_data()
        f['camera_speed'][:] = .8
        self.assertEqual(rules.bend_events(f, CONFIG), [])

    def test_large_knee_flexion_is_not_relabelled_as_forward_bending(self):
        f = bend_data()
        f['knee_flexion'][:] = 95.
        self.assertEqual(rules.bend_events(f, CONFIG), [])

    def test_sustained_standing_interrupts_continuous_stepping(self):
        t = np.arange(300) / 30
        raw = np.array(['stepping'] * len(t), dtype=object)
        raw[100:180] = 'standing'
        labels, _, _ = temporal.stabilize(t, raw, np.ones(len(t), bool), [], CONFIG)
        self.assertTrue((labels[100:180] == 'standing').all())

    def test_post_bend_scale_drift_does_not_extend_recovery_into_next_activity(self):
        t = np.arange(240) / 20
        ratio = 1 - .72 * np.exp(-((t - 4) / .5) ** 2) + .04 * np.maximum(t - 6, 0)
        f = dict(time_s=t, local_torso_ratio=ratio, foot_speed=np.full(len(t), .03),
                 knee_flexion=np.full(len(t), 15.), camera_speed=np.full(len(t), .002))
        events = rules.bend_events(f, CONFIG)
        self.assertEqual(len(events), 1)
        self.assertLess(events[0]['end_video_s'], 5.5)


if __name__ == '__main__':
    unittest.main()

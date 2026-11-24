import unittest

import numpy as np

from action_temporal import stabilize
from squat_bouts import adaptive_cycle_recovery, normalize_squat_cycles
from test_action_boundaries import CONFIG


class SquatBoutsTests(unittest.TestCase):
    def test_stationary_recovery_joins_squat_stance_variants_without_erasing_steps(self):
        t = np.arange(76) / 20
        for condition in ('stationary','foot_move','step','track_gap'):
            with self.subTest(condition=condition):
                pts=np.zeros((len(t),17,2))
                active=np.ones(len(t),bool)
                labels=np.full(len(t),'unknown',object)
                if condition=='foot_move':
                    pts[26:39,15,0]=np.linspace(0,41,13)
                elif condition=='step':
                    labels[28:36]='walking'
                elif condition=='track_gap':
                    active[30:33]=False
                f=dict(time_s=t, points=pts, torso_visible=np.ones(len(t),bool),
                       raw_valid=np.ones((len(t),17),bool), torso_px=np.full(len(t),100.),
                       foot_speed=np.full(len(t),.15))
                events=[dict(event_id=1,activity='split_squat', start_video_frame=5,end_video_frame=26,
                             start_video_s=t[4],end_video_s=t[26]),
                        dict(event_id=2,activity='squat', start_video_frame=40,end_video_frame=64,
                             start_video_s=t[39],end_video_s=t[64])]
                rows=[dict(activity=x) for x in labels]
                normalize_squat_cycles(f,active,events,rows,CONFIG)
                result,_,protected=stabilize(t,[r['activity'] for r in rows],f['torso_visible'],events,CONFIG,preserve_walk=True)
                self.assertEqual([r['activity'] for r in events], ['squat','squat'])
                self.assertEqual(events[0]['stance_rule_activity'],'split_squat')
                self.assertEqual(bool(protected[30]),condition=='stationary')
                self.assertEqual(result[31], 'squat' if condition=='stationary' else 'walking' if condition=='step' else 'unknown')
                self.assertEqual([(r['start_video_frame'],r['end_video_frame']) for r in events],[(5,26),(40,64)])

    def test_subtle_squat_can_be_found_without_legacy_seed_but_bow_cannot(self):
        t = np.arange(240) / 30
        depth = np.maximum(0., np.cos(np.pi*(t-4)/1.15))**2 * (np.abs(t-4)<.575)
        for drop in (26., 0.):
            torso = np.full(len(t), 100.)
            hip = np.c_[np.zeros(len(t)), 200 + drop * depth]
            pts = np.zeros((len(t),17,2))
            pts[:,15:17,1] = 350.
            f = dict(time_s=t, points=pts, hip=hip, torso_px=torso,
                     torso_visible=np.ones(len(t),bool), raw_valid=np.ones((len(t),17),bool),
                     knee_flexion=10+13*depth, foot_speed=np.full(len(t), .02),
                     hip_vertical_speed_raw=np.gradient(hip[:,1],t)/torso,
                     local_torso_ratio=1-.25*depth, stance_width=np.full(len(t), .4),
                     knee_asymmetry=np.zeros(len(t)))
            frames = [dict(activity='unknown',action_event_id=0) for _ in t]
            events, _ = adaptive_cycle_recovery(f, [], frames, CONFIG)
            self.assertEqual(len(events), int(drop > 0.))
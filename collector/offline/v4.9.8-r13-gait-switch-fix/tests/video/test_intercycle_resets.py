import copy
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from intercycle_reset_rules import reset_cores, resolve_overlaps, trusted_reset
from video_to_intervals import parser, label_poses


def fixture():
    t=np.arange(250)/50
    y=np.maximum(0,1-abs(t-1)/.7)+np.maximum(0,1-abs(t-3.5)/.7)
    g=np.zeros((len(t),1,2));g[:,0,1]=y
    f=dict(t=t,valid=np.ones(len(t),bool),body_top_signal=y,groups=[g.copy() for _ in range(5)],
           base=dict(height=y,ankle_speed=np.zeros(len(t))))
    events=[dict(rep_id=1,start_video_frame=25,end_video_frame=85,bottom_video_frame=51),
            dict(rep_id=2,start_video_frame=100,end_video_frame=210,bottom_video_frame=176)]
    return f,events,[None,(115,130),None],SimpleNamespace(min_hip_drop=.04,boundary_fraction=.1)


class IntercycleResetTests(unittest.TestCase):
    def test_stale_crossing_restarts_after_observed_reset(self):
        f,e,h,a=fixture();cores,trusted,barriers=reset_cores(f,e,h,a,[])
        self.assertGreater(cores[1]['start_video_frame'],h[1][1])
        self.assertEqual(cores[0],e[0]);self.assertEqual(cores[1]['bottom_video_frame'],e[1]['bottom_video_frame'])
        self.assertIn(1,barriers);self.assertIn(1,trusted)

    def test_slow_deep_pause_is_not_a_standing_separator(self):
        f,e,h,a=fixture();f['groups'][2][115:130,0,1]=1
        self.assertFalse(trusted_reset(f,e[0],e[1],h[1],a,[]))

    def test_walking_does_not_supply_reset(self):
        f,e,h,a=fixture();f['base']['ankle_speed'][115:130]=1
        self.assertFalse(trusted_reset(f,e[0],e[1],h[1],a,[]))

    def test_missing_pose_does_not_supply_reset(self):
        f,e,h,a=fixture();f['valid'][120]=False
        self.assertFalse(trusted_reset(f,e[0],e[1],h[1],a,[]))

    def test_timestamp_gap_does_not_supply_reset(self):
        f,e,h,a=fixture();f['t'][120:]+=.7
        self.assertFalse(trusted_reset(f,e[0],e[1],h[1],a,[]))

    def test_jump_does_not_supply_reset(self):
        f,e,h,a=fixture();j=[dict(start_video_frame=117,end_video_frame=125,reason='jump_like_bilateral_lift_and_fast_rise')]
        self.assertFalse(trusted_reset(f,e[0],e[1],h[1],a,j))

    def test_disjoint_proposals_are_unchanged(self):
        f,e,h,a=fixture();previous=copy.deepcopy(e)
        self.assertFalse(resolve_overlaps(e,e,f['t'],{},{}));self.assertEqual(e,previous)

    def test_ambiguous_overlap_uses_cores_and_explicit_review(self):
        f,cores,h,a=fixture();events=copy.deepcopy(cores)
        events[0]['end_video_frame']=125;events[1]['start_video_frame']=105
        resolve_overlaps(events,cores,f['t'],{},{} )
        self.assertEqual([(x['start_video_frame'],x['end_video_frame']) for x in events],[(x['start_video_frame'],x['end_video_frame']) for x in cores])
        self.assertTrue(all(x['body_boundary_needs_review']==1 for x in events))
        self.assertEqual([x['bottom_video_frame'] for x in events],[x['bottom_video_frame'] for x in cores])

    def test_supported_overlap_excludes_the_entire_reset_hold(self):
        f,cores,h,a=fixture();events=copy.deepcopy(cores)
        events[0]['end_video_frame']=125;events[1]['start_video_frame']=105
        resolve_overlaps(events,cores,f['t'],{1:h[1]},{} )
        self.assertLessEqual(events[0]['end_video_frame'],115)
        self.assertGreaterEqual(events[1]['start_video_frame'],131)

    def test_real_failure_recording_finishes_without_overlaps(self):
        path=Path(__file__).resolve().parents[1]/'fixtures/intercycle_reset_pose.npz'
        with np.load(path) as z:data=dict(z)
        args=parser().parse_args(['--output','unused'])
        events,frames,rejected,_=label_poses(data,args)
        self.assertEqual(len(events),10)
        self.assertTrue(all(a['end_video_frame']<b['start_video_frame'] for a,b in zip(events,events[1:])))
        self.assertTrue(all(not e.get('body_boundary_needs_review') for e in events))
        for e in events:
            a,b=e['start_video_frame']-1,e['end_video_frame']
            self.assertTrue(all(f['squat_label']==1 and f['rep_id']==e['rep_id'] for f in frames[a:b]))


if __name__=='__main__':unittest.main()

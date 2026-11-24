import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from body_boundary_rules import extend_event

def fixture(irregular=False):
    t=np.arange(0,5,.02)
    if irregular:
        t=t+.003*np.sin(np.arange(len(t))*1.3)
    body=np.maximum(0,1-np.abs((t-2.5)/1.8))
    hip=np.maximum(0,1-np.abs((t-2.5)/1.0))
    groups=[]
    for wave in [body,body,hip,hip,body]:
        arr=np.zeros((len(t),1,2))
        arr[:,0,1]=wave*.5
        groups.append(arr)
    f=dict(t=t,valid=np.ones(len(t),bool),groups=groups)
    event=dict(start_video_frame=76,end_video_frame=175)
    return f,event,(0,20),(230,250)

class BodyBoundaryTests(unittest.TestCase):
    def test_early_and_late_upper_body_motion(self):
        f,e,left,right=fixture()
        a,b,_,status=extend_event(f,e,left,right,.03)
        self.assertLess(f['t'][a],.9)
        self.assertGreater(f['t'][b],4.1)
        self.assertEqual(status,'body_rule_candidate')

    def test_same_hip_cycle_without_upper_motion_is_shorter(self):
        f,e,left,right=fixture()
        a,b,_,_=extend_event(f,e,left,right,.03)
        for index in [0,1,4]:
            f['groups'][index][:]=0
        a0,b0,_,_=extend_event(f,e,left,right,.03)
        self.assertLess(a,a0)
        self.assertGreater(b,b0)

    def test_brief_isolated_pose_error_does_not_expand_to_it(self):
        f,e,left,right=fixture()
        a,b,_,_=extend_event(f,e,left,right,.03)
        f['groups'][0][25:27,0,0]=1
        a1,b1,_,_=extend_event(f,e,left,right,.03)
        self.assertEqual((a,b),(a1,b1))

    def test_missing_upper_body_is_explicit_fallback(self):
        f,e,left,right=fixture()
        f['valid'][85:110]=False
        a,b,details,status=extend_event(f,e,left,right,.03)
        self.assertEqual((a,b),(75,175))
        self.assertIsNone(details)
        self.assertEqual(status,'missing_upper_body_or_time_gap')

    def test_presentation_time_gap_is_not_bridged(self):
        f,e,left,right=fixture()
        f['t'][100:]+=.7
        _,_,_,status=extend_event(f,e,left,right,.03)
        self.assertEqual(status,'presentation_time_gap')

    def test_irregular_pts_give_matching_seconds(self):
        f,e,left,right=fixture()
        g,_,_,_=fixture(True)
        a,b,_,_=extend_event(f,e,left,right,.03)
        x,y,_,_=extend_event(g,e,left,right,.03)
        self.assertLess(abs(f['t'][a]-g['t'][x]),.03)
        self.assertLess(abs(f['t'][b]-g['t'][y]),.03)

if __name__=='__main__':
    unittest.main()

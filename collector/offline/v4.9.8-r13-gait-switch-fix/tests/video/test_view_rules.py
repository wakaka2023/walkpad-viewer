import unittest
from types import SimpleNamespace
import numpy as np
from view_body_rules import adaptive_reference,local_reference_bounds,estimate_view,side_features
from video_to_intervals import parser

class ViewRuleTests(unittest.TestCase):
    def test_view_uses_projected_torso_geometry(self):
        k=np.ones((50,17,3),float);k[:,:,2]=.99
        for shoulder_width,expected in [(8,'side'),(70,'front')]:
            k[:,5,:2]=[100,20];k[:,6,:2]=[100+shoulder_width,20]
            k[:,11,:2]=[100,120];k[:,12,:2]=[100+shoulder_width,120]
            info=estimate_view({'keypoints':k},SimpleNamespace(view='auto',keypoint_conf=.5))
            self.assertEqual(info['view'],expected)

    def test_explicit_view_overrides_oblique_ambiguity(self):
        k=np.ones((20,17,3));k[:,:,2]=.99
        k[:,5,:2]=[100,20];k[:,6,:2]=[130,20]
        k[:,11,:2]=[100,120];k[:,12,:2]=[130,120]
        info=estimate_view({'keypoints':k},SimpleNamespace(view='front',keypoint_conf=.5))
        self.assertEqual(info['view'],'front');self.assertTrue(info['view_uncertain'])

    def test_turnaround_is_not_described_as_stationary(self):
        t=np.arange(100)/50
        f=dict(t=t,valid=np.ones(100,bool),body_top_signal=(t-1)**2,body_speed=np.ones(100))
        ref,quality,_=adaptive_reference(f,0,100)
        self.assertIsNotNone(ref)
        self.assertEqual(quality,'high_posture_turnaround_no_stationary_hold')

    def test_missing_pose_does_not_create_reference(self):
        f=dict(t=np.arange(100)/50,valid=np.zeros(100,bool),body_top_signal=np.zeros(100),body_speed=np.zeros(100))
        ref,quality,_=adaptive_reference(f,0,100)
        self.assertIsNone(ref);self.assertEqual(quality,'missing_visible_body_reference')

    def test_foot_support_excludes_setup_and_departure(self):
        t=np.arange(100)/50;ankles=np.zeros((100,1,2))
        ankles[:20,0,0]=1;ankles[80:,0,0]=1
        f=dict(t=t,valid=np.ones(100,bool),base=dict(ankles=ankles,scale=1))
        lo,hi=local_reference_bounds(f,dict(start_video_frame=41,end_video_frame=60),0,100,[])
        self.assertEqual((lo,hi),(20,80))

    def test_jump_is_barrier_to_body_extension(self):
        f=dict(t=np.arange(100)/50,valid=np.ones(100,bool),base=dict(ankles=np.zeros((100,1,2)),scale=1))
        jumps=[dict(start_video_frame=11,end_video_frame=30,reason='jump_like_bilateral_lift_and_fast_rise'),
               dict(start_video_frame=71,end_video_frame=90,reason='jump_like_bilateral_lift_and_fast_rise')]
        lo,hi=local_reference_bounds(f,dict(start_video_frame=41,end_video_frame=60),0,100,jumps)
        self.assertEqual((lo,hi),(30,70))

    def test_single_visible_side_does_not_invent_hidden_landmarks(self):
        t=np.arange(100)/50;k=np.full((100,17,3),np.nan)
        for i,xy in [(6,[0,0]),(12,[0,100]),(14,[5,150]),(16,[0,200])]:
            k[:,i,:2]=xy;k[:,i,2]=.99
        args=parser().parse_args(['--output','unused','--view','side']);args._geometry_side='right'
        f=side_features(dict(time_s=t,keypoints=k),args,'right')
        self.assertTrue(f['valid'].all());self.assertEqual(f['groups'][0].shape,(100,1,2))
        self.assertTrue(np.isnan(k[:,5]).all())

if __name__=='__main__':unittest.main()

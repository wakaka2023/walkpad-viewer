from pathlib import Path
import unittest
import numpy as np
from video_to_intervals import parser
from side_onset_rules import grounded_descent_onset,observed_step_boundary
from view_body_rules import side_features

def fixture(irregular=False,horizontal=False):
    t=np.arange(0,4,.02)
    if irregular:t+=.002*np.sin(np.arange(len(t)))
    body=.005*t+.35*np.maximum(t-2,0)**2
    k=np.ones((len(t),17,3));k[:,:,2]=.99;k[:,15,:]=np.nan
    k[:,16,:2]=[0,2];k[:,6,1]=.2+body;k[:,12,1]=1+body
    k[:,6,0]=0;k[:,12,0]=0
    if horizontal:k[:,6,0]=.25*np.maximum(t-1.4,0)**2
    positions=k[:,[6,12,14],:2]-k[:,16,None,:2]
    groups=[positions[:,:1],(positions[:,0]-positions[:,1])[:,None],positions[:,1:2],positions[:,2:3]]
    f=dict(t=t,valid=np.ones(len(t),bool),body_top_signal=.6*k[:,6,1]+.4*k[:,12,1]-2,
           groups=groups,base=dict(k=k,scale=1.,geometry_side='right'))
    e=dict(start_video_frame=51,hip_core_start_video_frame=121,hip_core_end_video_frame=175,bottom_video_frame=151)
    args=parser().parse_args(['--output','unused'])
    return f,e,args

class SideOnsetTests(unittest.TestCase):
    def test_standing_drift_is_removed_with_one_visible_chain(self):
        f,e,args=fixture();old=f['base']['k'].copy();a,d=grounded_descent_onset(f,e,args)
        self.assertLess(abs(f['t'][a]-2),.06)
        self.assertEqual(d['side_onset_status'],'grounded_raw_body_descent')
        np.testing.assert_equal(f['base']['k'],old)

    def test_head_nod_cannot_start_descent(self):
        f,e,args=fixture();f['base']['k'][:,0,1]=np.sin(f['t']*20)
        a,_=grounded_descent_onset(f,e,args);self.assertGreater(f['t'][a],1.9)

    def test_torso_and_hip_first_motion_is_preserved(self):
        f,e,args=fixture(horizontal=True);e['start_video_frame']=71
        retreat=-.25*np.maximum(f['t']-1.4,0)**2
        f['base']['k'][:,12,0]=retreat
        f['groups'][2][:,0,0]=retreat
        f['groups'][1][:,0,0]-=retreat
        a,d=grounded_descent_onset(f,e,args)
        self.assertEqual(a,70);self.assertEqual(d['side_onset_status'],'earlier_torso_motion_preserved')

    def test_other_visible_foot_can_block_entry_step(self):
        f,e,args=fixture();k=f['base']['k'];k[:,15]=k[:,16];k[:75,15,0]=.6
        boundary,drivers=observed_step_boundary(f,e,35,args)
        self.assertGreaterEqual(f['t'][boundary],1.4);self.assertIn(15,drivers)

    def test_brief_foot_outlier_is_not_a_step(self):
        f,e,args=fixture();k=f['base']['k'];k[:,15]=k[:,16];k[60,15,0]=.3
        boundary,_=observed_step_boundary(f,e,35,args)
        self.assertEqual(boundary,35)

    def test_missing_chain_is_flagged_without_filling(self):
        f,e,args=fixture();f['valid'][85:90]=False
        a,d=grounded_descent_onset(f,e,args)
        self.assertEqual(a,50);self.assertEqual(d['side_onset_needs_review'],1)

    def test_pts_gap_is_not_bridged(self):
        f,e,args=fixture();f['t'][85:]+=.6
        a,d=grounded_descent_onset(f,e,args)
        self.assertEqual(a,50);self.assertEqual(d['side_onset_status'],'presentation_time_gap')

    def test_irregular_source_pts_preserve_onset_seconds(self):
        f,e,args=fixture();g,_,_=fixture(True)
        a,_=grounded_descent_onset(f,e,args);b,_=grounded_descent_onset(g,e,args)
        self.assertLess(abs(f['t'][a]-g['t'][b]),.04)

    def test_no_extension_keeps_core(self):
        f,e,args=fixture();e['start_video_frame']=e['hip_core_start_video_frame']
        a,d=grounded_descent_onset(f,e,args);self.assertEqual(a,120)
        self.assertEqual(d['side_onset_status'],'no_precore_extension')

    def test_real_entry_standing_case(self):
        path=Path(__file__).resolve().parents[1]/'fixtures/side_entry_pose.npz'
        with np.load(path) as z:data=dict(z)
        args=parser().parse_args(['--output','unused']);args._geometry_side='right'
        f=side_features(data,args,'right')
        e=dict(start_video_frame=73,hip_core_start_video_frame=132,hip_core_end_video_frame=162,bottom_video_frame=150)
        a,d=grounded_descent_onset(f,e,args)
        self.assertGreaterEqual(f['t'][a],4.0);self.assertLessEqual(f['t'][a],4.2)
        self.assertGreaterEqual(d['side_onset_support_start_s'],2.65)
        self.assertEqual(d['side_onset_status'],'grounded_raw_body_descent')

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from front_onset_rules import raw_descent_onset

def fixture(onset=2.0, irregular=False, torso_first=False):
    t=np.arange(0,4,.02)
    if irregular:t=t+.002*np.sin(np.arange(len(t))*1.2)
    body=.005*t+.35*np.maximum(t-onset,0)**2
    k=np.ones((len(t),17,3));k[:,:,2]=.99
    k[:,[15,16],1]=2
    for ids,base in [([5,6],.2),([11,12],1)]:
        k[:,ids,1]=base+body[:,None]
    if torso_first:
        body_sh=.005*t+.45*np.maximum(t-1.4,0)**2
        k[:,[5,6],1]=.2+body_sh[:,None]
    signal=(.6*k[:,[5,6],1].mean(axis=1)+.4*k[:,[11,12],1].mean(axis=1)-2)
    event=dict(start_video_frame=51,hip_core_start_video_frame=121,bottom_video_frame=151)
    f=dict(t=t,valid=np.ones(len(t),bool),body_top_signal=signal,base=dict(k=k,scale=1))
    return f,event

class FrontOnsetTests(unittest.TestCase):
    def test_standing_drift_and_head_nod_do_not_start_squat(self):
        f,e=fixture();f['base']['k'][:,0,1]=np.sin(f['t']*7)
        a,d=raw_descent_onset(f,e)
        self.assertLess(abs(f['t'][a]-2),.06)
        self.assertEqual(d['onset_fit_status'],'raw_body_descent_change_point')

    def test_existing_upper_body_onset_is_not_delayed_to_hip(self):
        f,e=fixture(torso_first=True)
        e['start_video_frame']=71
        a,d=raw_descent_onset(f,e)
        self.assertAlmostEqual(f['t'][a],1.4)

    def test_centered_smoothing_cannot_supply_raw_onset_evidence(self):
        f,e=fixture();f['body_top_signal']+=.01*np.maximum(f['t']-.9,0)
        a,_=raw_descent_onset(f,e)
        self.assertGreater(f['t'][a],1.9)

    def test_time_weights_handle_irregular_pts(self):
        f,e=fixture();g,e=fixture(irregular=True)
        a,_=raw_descent_onset(f,e);b,_=raw_descent_onset(g,e)
        self.assertLess(abs(f['t'][a]-g['t'][b]),.05)

    def test_missing_pose_retains_flagged_original(self):
        f,e=fixture();f['valid'][85:90]=False
        a,d=raw_descent_onset(f,e)
        self.assertEqual(a,50);self.assertEqual(d['onset_fit_status'],'missing_pose_or_time_gap')

    def test_presentation_gap_is_not_crossed(self):
        f,e=fixture();f['t'][90:]+=.7
        a,d=raw_descent_onset(f,e)
        self.assertEqual(a,50);self.assertEqual(d['onset_fit_status'],'missing_pose_or_time_gap')

    def test_already_moving_retains_early_torso_onset(self):
        f,e=fixture(onset=.9)
        a,_=raw_descent_onset(f,e)
        self.assertEqual(a,50)

if __name__=='__main__':unittest.main()

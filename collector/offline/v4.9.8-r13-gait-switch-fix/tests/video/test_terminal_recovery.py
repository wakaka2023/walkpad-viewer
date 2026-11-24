import unittest
import numpy as np
from body_boundary_rules import terminal_standing_stop

def fixture():
    t=np.arange(0,7,.02)
    body=np.maximum(0,1-np.abs((t-3)/2.2))*.8
    leg=np.maximum(0,1-np.abs((t-3)/1.4))*.5
    groups=[]
    for wave in (body,body,leg,leg,np.maximum(0,1-np.abs((t-4)/2.8))):
        v=np.zeros((len(t),1,2));v[:,0,1]=wave;groups.append(v)
    speed=np.maximum(abs(np.gradient(body,t)),abs(np.gradient(leg,t)))
    f=dict(t=t,groups=groups,valid=np.ones(len(t),bool),body_speed=speed)
    core=dict(end_video_frame=210,bottom_video_frame=151)
    return f,core,(0,20),340

class TerminalRecoveryTests(unittest.TestCase):
    def test_head_only_motion_does_not_extend_completed_body_recovery(self):
        f,e,b,end=fixture()
        stop,info=terminal_standing_stop(f,e,b,end)
        self.assertGreater(f['t'][stop],5.15)
        self.assertLess(f['t'][stop],5.3)
        self.assertEqual(info['status'],'first_upright_body_hold')
        self.assertGreater(f['groups'][4][stop,0,1],.1)

    def test_slow_torso_recovery_is_kept_after_legs_finish(self):
        f,e,b,end=fixture()
        stop,_=terminal_standing_stop(f,e,b,end)
        self.assertGreater(f['t'][stop],5.15)
        self.assertEqual(f['groups'][2][stop,0,1],0)

    def test_quiet_bent_over_pause_is_not_standing(self):
        f,e,b,end=fixture()
        f['body_speed'][215:240]=0
        stop,_=terminal_standing_stop(f,e,b,end)
        self.assertGreater(f['t'][stop],5.15)

    def test_confirmation_dwell_is_excluded_from_squat(self):
        f,e,b,end=fixture()
        stop,info=terminal_standing_stop(f,e,b,end)
        self.assertEqual(stop,info['start'])
        self.assertGreaterEqual(f['t'][info['confirmation_end']]-f['t'][stop],.2-1e-9)

    def test_no_quiet_hold_does_not_invent_an_endpoint(self):
        f,e,b,end=fixture()
        f['body_speed'][210:]=.4
        stop,info=terminal_standing_stop(f,e,b,end)
        self.assertEqual(stop,end)
        self.assertEqual(info['status'],'no_confirmed_terminal_standing_hold')

    def test_missing_pose_and_large_pts_gap_do_not_confirm_a_hold(self):
        f,e,b,end=fixture()
        f['valid'][270:]=False
        f['t'][267:]+=.7
        stop,_=terminal_standing_stop(f,e,b,end)
        self.assertEqual(stop,end)

if __name__=='__main__':
    unittest.main()

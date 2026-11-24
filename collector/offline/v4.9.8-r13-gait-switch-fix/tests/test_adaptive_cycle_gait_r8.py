import unittest
import numpy as np
from adaptive_cycle_gait import _fit, _mode_split, apply
from bounded_action_bridge import bridge
from action_pipeline import load_config


class AdaptiveGait(unittest.TestCase):
    def test_phase_shape_not_pace(self):
        fitted=[]
        for stride_hz in (.67,1.46):
            t=np.arange(120)/30
            gait=np.sin(2*np.pi*stride_hz*t)
            recoil=4*np.cos(4*np.pi*stride_hz*t)
            pose=np.ones((len(t),17,3),float)
            pose[:,:,0]=120
            pose[:,[5,6],1]=50+recoil[:,None]
            pose[:,[11,12],1]=150+recoil[:,None]
            pose[:,[15,16],1]=310+recoil[:,None]
            pose[:,15,0]+=25*gait
            pose[:,16,0]-=25*gait
            result=_fit(t,pose,np.ones((len(t),17),bool))
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result['period'],.5/stride_hz,delta=.04)
            fitted.append(result['impulse_ratio'])
        self.assertAlmostEqual(fitted[0],fitted[1],delta=.02)

    def test_static_pose_abstains_and_protected_event_unchanged(self):
        t=np.arange(60)/30
        pose=np.ones((60,17,3),float)
        pose[:,[5,6],1]=20
        pose[:,[11,12],1]=120
        pose[:,[15,16],1]=300
        rows=[dict(activity='squat',action_event_id=1) for _ in t]
        self.assertIsNone(_fit(t,pose,np.ones((60,17),bool)))
        apply(t,pose,np.ones((60,17),bool),rows,np.ones(len(t),bool))
        self.assertEqual(rows,[dict(activity='squat',action_event_id=1) for _ in t])

    def test_modes_need_evidence(self):
        self.assertIsNone(_mode_split(np.linspace(.98,1.02,20),np.full(20,.07)))
        x=np.r_[np.linspace(.40,.48,12),np.linspace(1.20,1.30,12)]
        self.assertTrue(.48 < _mode_split(x,np.full(len(x),.08)) < 1.20)

    def test_gap_length_tracks_observed_cycle_and_does_not_cover_stop(self):
        n=60
        t=np.arange(n)/30
        labels=np.full(n,'running',object)
        labels[18:39]='unknown'
        rows=[dict(activity=k,action_event_id=0,personal_gait_step_period_s=.4) for k in labels]
        for r in rows[18:39]:r.update(activity='body_motion',gait_evidence='gait_type_unresolved')
        measured=np.ones(n,bool)
        reviews=[dict(activity_status='other' if k=='unknown' else 'supported',action_candidates='') for k in labels]
        def decide():
            return bridge(t,t+1/30,labels,rows,np.zeros(n,bool),np.zeros(n,bool),reviews,measured,np.ones(n,bool),load_config())
        self.assertEqual(np.count_nonzero(decide()!=''),21)
        for i in (17,39):rows[i]['personal_gait_step_period_s']=.3
        self.assertFalse(any(decide()))
        for i in (17,39):rows[i]['personal_gait_step_period_s']=.4
        rows[25]['activity']='standing'
        self.assertFalse(any(decide()))
        rows[25]['activity']='body_motion'; measured[25]=False
        self.assertFalse(any(decide()))


if __name__=='__main__':
    unittest.main()

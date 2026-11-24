import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import grade_repetitions as g
import individual_quality as q
import motion_fusion as m
from test_individual_quality import row, reviewed

def signals_fixture(burst=True, asymmetric=False, shift=0., active_metrics=3):
    t=np.arange(200)*.02
    event=(t >= 1.3+shift)&(t < 2.+shift)
    signals={}
    for side in g.SIDES:
        amplitude=30. if side=='left' or not asymmetric else 3.
        gyro=np.full((len(t),3),.2)
        alpha=np.ones((len(t),3))
        acc=np.full(len(t),1.)
        if burst:
            if active_metrics >= 1:gyro[event,0]=amplitude
            if active_metrics >= 2:alpha[event,0]=500.
            if active_metrics >= 3:acc[event]=1.+.1*np.sin(2*np.pi*8*t[event])
        signals[side]={'gyro':gyro,'alpha':alpha,'acc_magnitude':acc}
    e={'kind':'foot_displacement','start_s':1.4,'stop_s':1.9,'center_s':1.65,'level':1}
    r=dict(start_video_s=.2,end_video_s=3.8,video_motion_events_json=json.dumps([e]))
    return t,signals,r

class MotionFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cfg=g.load_config()

    def test_video_motion_alone_has_no_imu_fusion(self):
        t,s,r=signals_fixture(burst=False)
        result=m.imu_video_evidence(r,t,s,self.cfg)
        self.assertEqual(result['joint_video_imu_evidence_level'],0)

    def test_intense_imu_without_video_event_has_no_fusion(self):
        t,s,r=signals_fixture()
        r['video_motion_events_json']='[]'
        result=m.imu_video_evidence(r,t,s,self.cfg)
        self.assertEqual(result['joint_video_imu_evidence_level'],0)

    def test_matching_video_and_imu_can_change_grade(self):
        t,s,r=signals_fixture()
        evidence=m.imu_video_evidence(r,t,s,self.cfg)
        self.assertEqual(evidence['joint_video_imu_evidence_level'],1)
        result=q.decide(row(**evidence),self.cfg,reviewed())
        self.assertEqual(result['candidate_grade_code'],2)
        quiet=m.imu_video_evidence(r,t,signals_fixture(burst=False)[1],self.cfg)
        self.assertEqual(q.decide(row(**quiet),self.cfg,reviewed())['candidate_grade_code'],3)

    def test_noncoincident_burst_is_not_combined(self):
        t,s,r=signals_fixture(shift=1.3)
        self.assertEqual(m.imu_video_evidence(r,t,s,self.cfg)['joint_video_imu_evidence_level'],0)

    def test_one_high_scalar_cannot_trigger_multimetric_burst(self):
        t,s,r=signals_fixture(active_metrics=1)
        self.assertEqual(m.imu_video_evidence(r,t,s,self.cfg)['joint_video_imu_evidence_level'],0)

    def test_acceleration_sd_can_be_the_second_supporting_metric(self):
        t,s,r=signals_fixture()
        for side in g.SIDES:s[side]['alpha'][:]=1.
        out=m.imu_video_evidence(r,t,s,self.cfg)
        self.assertEqual(out['joint_video_imu_evidence_level'],1)
        votes=json.loads(out['imu_event_details_json'])[0]['time_shift_tests'][0]['transient_votes']
        self.assertTrue(votes['gyro_rms_dps'] and votes['acc_magnitude_sd_g'])
        self.assertFalse(votes['angular_acceleration_rms_dps2'])

    def test_alpha_can_be_the_second_supporting_metric(self):
        t,s,r=signals_fixture(active_metrics=2)
        self.assertEqual(m.imu_video_evidence(r,t,s,self.cfg)['joint_video_imu_evidence_level'],1)

    def test_fourth_metric_requires_video_support_and_reliable_rotation(self):
        t,s,r=signals_fixture(burst=False)
        for side,value in [('left',10.),('right',3.)]:
            s[side]['gyro'][:]=0.;s[side]['gyro'][:,0]=value
        out=m.imu_video_evidence(r,t,s,self.cfg)
        self.assertEqual(out['imu_asymmetry_supported_by_video'],1)
        self.assertEqual(out['joint_video_imu_evidence_level'],1)
        r['video_motion_events_json']='[]'
        self.assertEqual(m.imu_video_evidence(r,t,s,self.cfg)['joint_video_imu_evidence_level'],0)
        for side in g.SIDES:s[side]['gyro']*=.001
        r['video_motion_events_json']=signals_fixture()[2]['video_motion_events_json']
        self.assertEqual(m.imu_video_evidence(r,t,s,self.cfg)['joint_video_imu_evidence_level'],0)

    def test_major_dynamic_video_and_concurrent_imu_can_support_poor(self):
        t,s,r=signals_fixture()
        event=json.loads(r['video_motion_events_json'])[0]
        event.update(kind='trunk_leg_dissociation',level=2)
        r['video_motion_events_json']=json.dumps([event])
        evidence=m.imu_video_evidence(r,t,s,self.cfg)
        result=q.decide(row(**evidence),self.cfg,reviewed())
        self.assertEqual(result['candidate_grade_code'],1)
        self.assertTrue(np.isfinite(result['pelvis_control_fault_level']))
        self.assertEqual(result['pelvis_control_fault_level'],0)

    def test_insufficient_background_cannot_support_fusion(self):
        t,s,r=signals_fixture()
        r.update(start_video_s=1.4,end_video_s=1.9)
        self.assertEqual(m.imu_video_evidence(r,t,s,self.cfg)['joint_video_imu_evidence_level'],0)

    def test_bias_or_constant_motion_is_not_a_transient(self):
        t,s,r=signals_fixture(burst=False)
        for side in g.SIDES:s[side]['gyro'][:]=100.;s[side]['alpha'][:]=500.
        self.assertEqual(m.imu_video_evidence(r,t,s,self.cfg)['joint_video_imu_evidence_level'],0)

    def test_zero_motion_exports_strict_json(self):
        t,s,r=signals_fixture(burst=False)
        for side in g.SIDES:s[side]['gyro'][:]=0.
        result=m.imu_video_evidence(r,t,s,self.cfg)
        self.assertNotIn('NaN',result['imu_event_details_json'])

    def test_sustained_event_rejects_spike_and_pts_gap(self):
        t=np.arange(100)*.02;x=np.zeros(100);x[50]=1.
        valid=np.ones(100,bool)
        self.assertIsNone(m.sustained_event(t,x,valid,.5,.12,'test'))
        x[:]=1.;t=np.arange(100)*.02+(np.arange(100)//3)*.3
        self.assertIsNone(m.sustained_event(t,x,valid,.5,.12,'test'))

    def test_static_lean_does_not_penalize(self):
        r=q.decide(row(video_trunk_shin_difference_deg=60.),self.cfg,reviewed())
        self.assertEqual(r['candidate_grade_code'],3)

    def test_front_dissociation_alone_does_not_claim_major_or_pelvic_fault(self):
        r=q.decide(row(video_control_dissociation_level=2,video_view_hint='front_or_oblique'),self.cfg)
        self.assertEqual(r['candidate_grade_code'],2)
        self.assertTrue(np.isnan(r['pelvis_control_fault_level']))
        self.assertTrue(np.isnan(r['lumbar_control_fault_level']))

    def test_torso_leg_synchronous_motion_is_not_dissociation(self):
        t=np.arange(151)*.02;p=np.maximum(0,1-abs(t-1.5)/.9)
        hip=np.c_[np.zeros(len(t)),80+40*p]
        knee=np.c_[np.zeros(len(t)),120+20*p]
        ankle=np.tile([0.,180.],(len(t),1))
        events=m.joint_video_events(t,(t>=.6)&(t<2.4),np.ones(len(t),bool),hip,knee,ankle,
                                    1-.6*p,100.,1.5,p,False,self.cfg)
        self.assertFalse(any(e['kind']=='trunk_leg_dissociation' for e in events))

    def test_torso_change_with_quiet_legs_has_sustained_joint_evidence(self):
        t=np.arange(151)*.02;p=np.ones(len(t))
        hip=np.tile([0.,120.],(len(t),1));knee=np.tile([0.,140.],(len(t),1))
        ankle=np.tile([0.,180.],(len(t),1))
        torso=1-.5*np.clip((t-1.)/.4,0,1)
        events=m.joint_video_events(t,(t>=.6)&(t<2.4),np.ones(len(t),bool),hip,knee,ankle,
                                    torso,100.,1.5,p,False,self.cfg)
        self.assertTrue(any(e['kind']=='trunk_leg_dissociation' for e in events))

if __name__=='__main__':unittest.main()

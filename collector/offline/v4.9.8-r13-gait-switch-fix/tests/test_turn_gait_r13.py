import unittest
import numpy as np
from standing_turn import apply
from gait_gap_evidence import _alternation, bridges
from cropped_passage import _cycle


def camera(n, shift=0.):
    mats=np.tile(np.eye(2,3),(n,1,1)); mats[1:,0,2]=shift
    return dict(source_indices=np.arange(n), matrices=mats, valid=np.ones(n,bool))


def turn(mode, fps):
    t=np.arange(4*fps)/fps; n=len(t)
    p=np.zeros((n,17,3)); p[:,:,2]=1
    change=np.clip((t-1.6)/.6,0,1)
    p[:,[5,6],1]=100; p[:,[11,12],1]=200; p[:,13:15,1]=247; p[:,15:17,1]=300
    p[:,[5,6],0]=100 + change[:,None]*np.array([-32.,32.])
    p[:,[11,12],0]=100 + change[:,None]*np.array([-21.,21.])
    p[:,15:17,0]=[96.,104.]
    if mode=='sway':
        p[:,[5,6,11,12],0]=100+(p[:,[5,6,11,12],0]-100)*.1
    if mode!='upper_only':p[:,15,0]-=38*change
    else:p[:,[11,12],0]=100
    if mode=='walking':p[:,:,0]+=110*change[:,None]
    shift=1.5; p[:,:,0]+=shift*np.arange(n)[:,None]
    ok=np.ones((n,17),bool); seen=np.ones(n,bool)
    if mode=='unobserved':seen[(t>1.58)&(t<2.21)]=False
    rows=[dict(activity='standing',action_event_id=int(mode=='squat' and 1.3<x<2.5)) for x in t]
    apply(t,p,ok,rows,seen,camera(n,shift),.23)
    return t,np.array([r['activity']=='standing_turn' for r in rows])


class TurnAndGaitR13(unittest.TestCase):
    def test_turn_at_different_sample_rates_leaves_rest(self):
        for fps in (24,30,60):
            t,y=turn('turn',fps)
            self.assertGreater(y.sum(),0)
            self.assertFalse(y[(t<1.35)|(t>2.45)].any())

    def test_turn_rejects_upper_body_only_travel_squat_and_untracked(self):
        for mode in ('upper_only','walking','squat','unobserved','sway'):
            self.assertFalse(turn(mode,30)[1].any(),mode)

    def test_local_phase_is_fit_from_flanks_and_rejects_pause(self):
        for step in (.28,.52,.81):
            t=np.arange(200)*step/31; a,b=79,108
            signal=np.c_[np.cos(np.pi*t/step),.5*np.sin(np.pi*t/step)]
            self.assertTrue(_alternation(t,signal,np.arange(a),np.arange(b,len(t)),a,b,step,.015))
            gap=signal.copy(); gap[a:b]=gap[a]
            self.assertFalse(_alternation(t,gap,np.arange(a),np.arange(b,len(t)),a,b,step,.015))

    def test_bilateral_cycle_not_joint_common_motion(self):
        p=.42; t=np.arange(65)/50
        x=np.c_[.4*np.cos(np.pi*t/p),.3*np.sin(np.pi*t/p)]
        self.assertTrue(_cycle(t,np.stack([x,-x],axis=1),p,.006))
        self.assertFalse(_cycle(t,np.stack([x,x],axis=1),p,.006))

    def test_gait_bridge_does_not_cross_a_missing_track_or_a_real_stop(self):
        for bad in ('missing','event','stop','none'):
            n=145;t=np.arange(n)/40; step=.5; a,b=61,80
            pose=np.zeros((n,17,3)); pose[:,[11,12],1]=100; pose[:,15:17,1]=175
            swing=22*np.cos(np.pi*t/step)
            pose[:,15,0]=swing;pose[:,16,0]=-swing
            labels=np.full(n,'running',object); labels[a:b]='unknown'
            if bad=='stop':pose[a:b]=pose[a]
            v=np.ones((n,17),bool); seen=np.ones(n,bool); protected=np.zeros(n,bool)
            if bad=='event':protected[(a+b)//2]=True
            if bad=='missing':seen[(a+b)//2]=False
            rows=[dict(activity=k,action_event_id=0,personal_gait_step_period_s=step) for k in labels]
            rv=[dict(activity_status='supported' if k=='running' else 'other',action_candidates='') for k in labels]
            result=bridges(t,t+.025,labels,rows,np.zeros(n,bool),protected,rv,seen,pose,v,camera(n))
            self.assertEqual(np.count_nonzero(result!=''),b-a if bad=='none' else 0,bad)


if __name__=='__main__':unittest.main()

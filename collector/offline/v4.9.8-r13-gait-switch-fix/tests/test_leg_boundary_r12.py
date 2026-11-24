import unittest
import numpy as np
from leg_boundary_evidence import recover_walk_handoffs, recover_clipped_walk, recover_squat_edges


def trial(n):
    t = np.arange(n)/30.
    p = np.ones((n,17,3))
    v = np.ones((n,17), bool)
    c = dict(source_indices=np.arange(n), matrices=np.tile(np.eye(2,3),(n,1,1)), valid=np.ones(n,bool))
    return t,p,v,c


def row(kind, **extra):
    return dict(activity=kind, activity_reason='original', action_event_id=0, **extra)


class LegBoundaryTest(unittest.TestCase):
    def test_measured_handoff_repairs_gap_but_not_still_feet_or_missing_track(self):
        for mode in ('move','still','untracked'):
            t,p,v,c = trial(40)
            p[:,15:17,0] = [20.,45.]
            if mode!='still':
                p[15:24,15,0] += np.arange(9)*2
            r = [row('unknown') if 15 <= i < 24 else row('gait_walk',personal_gait_step_period_s=.45) for i in range(len(t))]
            m = np.ones(len(t),bool)
            if mode=='untracked': m[19] = False
            recover_walk_handoffs(t,t+1/30,p,v,r,m,c)
            self.assertTrue(all(x['activity']==('gait_walk' if mode=='move' else 'unknown') for x in r[15:24]))

    def test_receding_thigh_walk_requires_visible_leg_asymmetry(self):
        for motion in (True,False):
            t,p,v,c = trial(80)
            torso = 320*np.exp(-.90*t)
            swing = (.065*np.sin(2*np.pi*t/.90))*motion
            p[:,11:13,0] = [80,135]
            p[:,11:13,1] = 600-torso[:,None]
            p[:,13:15,:2] = p[:,11:13,:2]
            p[:,13:15,1] += torso[:,None]*np.stack([1+swing,1-swing],axis=1)
            v[:32,15:17] = False
            r = [row('unknown') if i<40 else row('gait_walk',personal_gait_step_period_s=.45) for i in range(len(t))]
            recover_clipped_walk(t,t+1/30,p,v,r,np.ones(len(t),bool),c,630)
            self.assertEqual(any(x.get('boundary_inferred') for x in r[:35]),motion)

    def test_squat_edges_do_not_change_event_or_count(self):
        t,p,v,c=trial(64)
        k=np.zeros(len(t))+3
        phase=np.linspace(0,np.pi,25)
        k[17:42]+=90*np.sin(phase)
        p[:,15:17,1]=250
        p[:,15:17,0]=[35,75]
        p[:,11:13,1]=130+(k[:,None]-3)*.65
        ev=dict(activity='squat',start_video_frame=22,end_video_frame=37,peak_video_frame=30)
        ranges = [row('standing') if i < 17 or i >= 42 else row('unknown') for i in range(len(t))]
        for i in range(21,37): ranges[i].update(activity='squat',action_event_id=1)
        before=dict(ev)
        recover_squat_edges(t,p,v,ranges,np.ones(len(t),bool),dict(knee_flexion=k),[ev],c)
        self.assertEqual(ev,before)
        self.assertEqual(ranges[20]['activity'],'squat')
        self.assertEqual(ranges[37]['activity'],'squat')
        self.assertEqual(ranges[20]['action_event_id'],0)
        self.assertEqual(ranges[42]['activity'],'standing')


if __name__=='__main__':
    unittest.main()

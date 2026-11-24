import unittest

import numpy as np

from standing_turn import apply, _synchronous_evidence, STAGGERED


def example(fps, *, mode='pivot', yaw_delay=True, split_label=True, size=1., clock=1.):
    u=np.arange(4*fps)/fps
    t=u*clock
    ramp=lambda start,duration: np.clip((u-start)/duration,0,1)
    shoulder=ramp(1.25,.30)
    pelvis=ramp(1.61 if yaw_delay else 1.25,.32)
    ankle=ramp(1.98 if yaw_delay else 1.31,.33)
    if mode=='sway':
        shoulder*=.08; pelvis*=.08
    elif mode=='upper_only':
        pelvis*=0
    p=np.zeros((len(t),17,3),float)
    p[:,:,2]=1
    p[:,[5,6],:2]=[100,100]; p[:,[11,12],:2]=[100,200]
    p[:,13:15,:2]=[100,247]; p[:,15:17,1]=300
    p[:,[5,6],0]+=shoulder[:,None]*[-33,33]
    p[:,[11,12],0]+=pelvis[:,None]*[-22,22]
    if mode=='flip':
        p[:,[5,6],0]+=[16,-16]
        p[:,[11,12],0]+=[11,-11]
    p[:,15:17,0]=[96,104]
    if mode!='planted': p[:,15,0]-=38*ankle
    if mode=='walk': p[:,:,0]+=u[:,None]*75
    imageshift=23*u
    p[:,:,:2]*=size
    p[:,:,0]+=imageshift[:,None]
    mats=np.tile(np.eye(2,3),(len(t),1,1)); mats[1:,0,2]=np.diff(imageshift)
    cam=dict(source_indices=np.arange(len(t)),time_s=t,matrices=mats,valid=np.ones(len(t),bool))
    ok=np.ones(p.shape[:2],bool); seen=np.ones(len(t),bool)
    if mode=='gap': seen[(u>1.54)&(u<1.72)]=False
    if mode=='camera_gap': cam['valid'][np.searchsorted(u,1.64)]=False
    if mode=='ankle_hidden': ok[(u>1.94)&(u<2.34),15]=False
    rows=[dict(activity='walking' if mode=='walk' or split_label and 1.51<x<1.94 else 'standing',
               action_event_id=int(mode=='squat' and 1.50<x<1.9),
               stair_sequence_supported=int(mode=='stairs' and 1.50<x<1.9)) for x in u]
    return u,(t,p,ok,rows,seen,cam,.23)


def tags(rows):
    return np.array([r.get('activity_reason')==STAGGERED for r in rows])


class StaggeredTurnTest(unittest.TestCase):
    def test_turn_crosses_wrong_walk_seed_with_no_overlapping_shoulder_hip_movement(self):
        for fps,scale,clock in ((24,.75,1.4),(30,1.,1.),(60,1.9,.8)):
            with self.subTest(fps=fps):
                u,args=example(fps,size=scale,clock=clock)
                rows=args[3]
                _synchronous_evidence(*args)
                self.assertFalse(any(r['activity']=='standing_turn' for r in rows))
                apply(*args)
                added=tags(rows)
                self.assertTrue(added[np.searchsorted(u,1.4)])
                self.assertTrue(added[np.searchsorted(u,2.18)])
                self.assertFalse(added[(u<1.15)|(u>2.4)].any())
                self.assertTrue(all(r['activity']=='standing_turn' for r,x in zip(rows,u) if 1.52<x<1.93))

    def test_turn_extent_independent_of_walking_seed(self):
        _,a=example(30,split_label=False); _,b=example(30,split_label=True)
        apply(*a); apply(*b)
        np.testing.assert_array_equal(tags(a[3]),tags(b[3]))
        self.assertTrue(tags(a[3]).any())

    def test_opposite_facing_can_recover_even_if_every_seed_is_walking(self):
        u,args=example(30,mode='flip')
        for row in args[3]:
            row['activity']='walking'
        apply(*args)
        within=[row for x,row in zip(u,args[3]) if 1.52<x<1.93]
        self.assertTrue(all(row.get('turn_evidence_level')=='observed_opposite_facings' for row in within))
        self.assertTrue(all(row['activity']=='walking' for x,row in zip(u,args[3]) if x<1.15 or x>2.4))

    def test_partial_facing_yields_candidate_not_supported(self):
        _,args=example(30)
        apply(*args)
        from multi_labels import review
        turns=[row for row in args[3] if row.get('activity_reason')==STAGGERED]
        self.assertTrue(turns)
        self.assertTrue(all(review('standing_turn',row,False,False,{},0)['activity_status']=='candidate' for row in turns))

    def test_no_turn_for_negative_controls(self):
        for mode in ('walk','planted','sway','upper_only','gap','camera_gap','ankle_hidden','squat','stairs'):
            with self.subTest(mode=mode):
                _,a=example(30,mode=mode)
                apply(*a)
                self.assertFalse(any(r['activity']=='standing_turn' for r in a[3]))


if __name__=='__main__':
    unittest.main()

import unittest
import numpy as np

from activity_rules import runs
from glass_surface import foreground_slats, persistent_foreground_slats
from multi_labels import LABELS, CANDIDATE, review
from walk_continuity import bridges, REASON


class SurfaceEvidence(unittest.TestCase):
    def fixture(self, box, pitch, tilt, kind):
        rng = np.random.default_rng(10)
        y,x = np.indices((440,570))
        stripe = np.cos(2*np.pi*(y - tilt*x)/pitch)
        x1,y1,x2,y2 = box
        person = ((x>(x1+.10*(x2-x1))) & (x<(x2-.10*(x2-x1))) & (y>=y1)&(y<y2))
        add = np.where(person, -27, 0) + 1.5*rng.standard_normal(x.shape)
        if kind == 'wall_only': stripe *= ~person
        if kind == 'clothing_only': stripe *= person
        if kind == 'misaligned': stripe = np.where(person,np.cos(2*np.pi*(y-tilt*x)/pitch + np.pi),stripe)
        return (133 + 15*stripe + add).clip(0,255).astype(np.uint8)

    def test_shared_signal_recognized_at_different_scales_and_positions(self):
        for box, pitch, tilt in (([130,87,246,305],6,.02),([340,123,424,277],8,-.035)):
            with self.subTest(box=box):
                image=self.fixture(box,pitch,tilt,'overlaid')
                self.assertIsNotNone(persistent_foreground_slats([image]*9, box))
                self.assertIsNotNone(foreground_slats(image.T, np.array(box)[[1,0,3,2]]))

    def test_real_person_in_front_and_stripes_on_clothing_are_not_removed(self):
        box=[190,87,306,305]
        for kind in ('wall_only','clothing_only','misaligned'):
            with self.subTest(kind=kind):
                self.assertIsNone(foreground_slats(self.fixture(box,6,.04,kind),box))
        covered=self.fixture(box,6,.04,'overlaid')
        opaque=self.fixture(box,6,.04,'wall_only')
        self.assertIsNone(persistent_foreground_slats([covered,opaque,covered],box))


class WalkContinuity(unittest.TestCase):
    def fixture(self, mode=None):
        t=np.arange(180)/30
        n=len(t)
        a,b,c,d=(68,79,84,113)
        labels=np.full(n,'walking',object)
        labels[a:b]=labels[c:d]='unknown'
        rows=[]; reviews=[]
        for k in labels:
            rows.append(dict(activity='gait_walk' if k=='walking' else 'body_motion', action_event_id=0,
                             gait_evidence='' if k=='walking' else 'gait_type_unresolved', personal_gait_step_period_s=.5))
            reviews.append(dict(activity_status='supported' if k=='walking' else 'other',action_candidates=''))
        measured=np.ones(n,bool); valid=np.ones((n,17),bool)
        if mode=='gap': measured[c]=False
        if mode=='stop': labels[c]='standing'; rows[c]['activity']='standing'
        angles=2*np.pi*t
        distance=t if mode not in ('camera','turn') else np.zeros_like(t) if mode=='camera' else -np.abs(t-t[c])
        if mode=='paused': distance=np.where(t<t[c],t,np.where(t<t[d],t[c],t-t[d]+t[c]))
        scale=np.exp(.13*distance)
        poses=np.zeros((n,17,3)); poses[:,:,2]=1.
        poses[:,5:7,0]=250.; poses[:,11:13,0]=250.
        poses[:,5:7,1]=175. - 40*scale[:,None]; poses[:,11:13,1]=175.
        poses[:,15,0]=248 + 14*np.sin(angles)*scale; poses[:,16,0]=252 - 14*np.sin(angles)*scale
        poses[:,15,1]=175 + (68+7*np.cos(angles))*scale; poses[:,16,1]=175 + (68-7*np.cos(angles))*scale
        matrices=np.repeat([[[1.,0.,0.],[0.,1.,0.]]],n,axis=0)
        if mode=='camera':
            m=np.exp(.13*t); poses[:,:,:2]*=m[:,None,None]
            matrices[1:,0,0]=matrices[1:,1,1]=m[1:]/m[:-1]
        camera=dict(source_indices=np.arange(n),valid=np.ones(n,bool),matrices=matrices)
        z=np.zeros(n,bool)
        res=bridges(t,t+1/30,labels,rows,z,z,reviews,measured,poses,valid,camera)
        return [(l,r) for l,r in runs(res==REASON)],[(a,b),(c,d)]

    def test_two_gaps_with_short_walking_island(self):
        self.assertEqual(*self.fixture())

    def test_no_fill_for_camera_zoom_reversal_pause_or_missing_observation(self):
        for mode in ('camera','turn','paused','gap','stop'):
            with self.subTest(mode=mode):
                self.assertEqual(self.fixture(mode)[0], [])

    def test_contextual_and_single_weak_walking_use_same_public_status(self):
        target=dict(annotation=LABELS['walking']+CANDIDATE, activity_status='candidate',action_candidates='walking')
        self.assertEqual(review('walking',dict(gait_inferred=1),True,False,{},0),target)
        self.assertEqual(review('walking',dict(action_hypotheses=('walking',),ambiguous_gait=1),False,False,{},0),target)
        self.assertEqual(review('walking',{},False,False,{},0)['activity_status'],'supported')


if __name__=='__main__': unittest.main()

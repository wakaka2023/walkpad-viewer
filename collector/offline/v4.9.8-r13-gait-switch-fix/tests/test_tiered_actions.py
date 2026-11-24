import unittest

import numpy as np

from multi_labels import LABELS, CANDIDATE, annotation
from stair_evidence import detect
from gait_phase import window_evidence


def track(direction=-1, transfers=3):
    t=np.arange(0,4,1/30)
    def edge(start):
        a=np.clip((t-start)/.36,0,1)
        return a*a*(3-2*a)
    a,b,c=(edge(x) if i<transfers else np.zeros(len(t)) for i,x in enumerate((.65,1.3,1.95)))
    pose=np.zeros((len(t),17,3))
    for j,q in enumerate((a+c,b)):
        pose[:,15+j,0]=100+22*j+35*q
        pose[:,15+j,1]=490+direction*33*q
    for pts,y in (([11,12],310),([5,6],210)):
        pose[:,pts,1]=(y+direction*33*.85*(a+b+c))[:,None]
    return t,pose,100+18*(a+b+c)


class TieredActionTests(unittest.TestCase):
    def test_stairs_require_directional_landings_and_torso_agreement(self):
        for sign,action in ((-1,'stairs_up'),(1,'stairs_down')):
            for transfers in (2,3):
                t,p,_=track(sign,transfers)
                ok=np.ones(len(t),bool)
                labels,_,supported=detect(t,p,np.tile(ok[:,None],(1,17)),np.full(len(t),100.),ok,~ok,720)
                self.assertEqual(set(labels),{'',action})
                self.assertEqual(supported.any(),transfers==3)
                self.assertFalse((supported & (labels=='')).any())
                self.assertFalse((labels[t<.4]!='').any())

    def test_stairs_reject_single_lift_level_walk_squat_perspective_and_camera_pan(self):
        for mode in ('single','level','squat','perspective','pan','occluded'):
            with self.subTest(mode=mode):
                t,p,depth=track(0 if mode in ('level','squat') else -1,1 if mode=='single' else 3)
                ok=np.ones(len(t),bool); v=np.tile(ok[:,None],(1,17)); unit=np.full(len(t),100.)
                if mode=='pan': ok[:]=False
                if mode=='squat': p[:,[5,6,11,12],1]+=25*np.sin(np.pi*t[:,None]/4)**2
                if mode=='perspective': unit=depth
                if mode=='occluded': v[:,16]=False
                labels,_,supported=detect(t,p,v,unit,ok,np.zeros(len(t),bool),720)
                self.assertEqual(set(labels),{''})
                self.assertFalse(supported.any())

    def test_conflicting_gait_evidence_retains_two_candidates(self):
        t=np.arange(94)/30
        x=2*np.pi*1.25*t
        apart=47*np.sin(x)
        y=9*np.sin(x)**2
        result=window_evidence(t,np.c_[apart,apart*.2,y,-y],100.)
        self.assertEqual((result['activity'],result['walking_support'],result['running_support']),('',1.,1.))
        row=annotation('walking',('walking','running'),False)
        self.assertEqual(row['action_candidates'],'walking;running')
        self.assertEqual(row['annotation'],' / '.join(LABELS[x]+CANDIDATE for x in ('walking','running')))
        self.assertEqual(annotation('unknown',(),True)['activity_status'],'other')

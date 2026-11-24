import unittest

import numpy as np

from gait_ground import advancing_walk, bracketed_walk_gaps, continuity, motion_above_own_baseline, support_handoffs
from stair_evidence import extend_anchored_traversals


class AdaptiveGaitAndStairsTests(unittest.TestCase):
    def test_grounded_pattern_at_different_individual_step_rates(self):
        t = np.arange(0, 4.5, 1/30.)
        for period in (.48, .82, 1.11):
            wave = np.sin(np.pi*t/period)
            speed = np.maximum(0., np.c_[wave, -wave]-.43)**2
            x = np.cumsum(speed, axis=0)*8
            ankles = np.stack([x, x*.15], axis=2)
            with self.subTest(period=period):
                self.assertGreaterEqual(len(support_handoffs(t, ankles, .5*np.sin(t), 100.)), 3)
                self.assertEqual(support_handoffs(t, ankles, 70*wave**2, 100.), [])

    def test_short_advance_with_transverse_turn_but_no_unchanged_footing(self):
        t = np.arange(0,2.1,1/30.)
        wave = np.sin(np.pi*(t+.2)/.52)
        swing = np.cumsum(np.maximum(0.,np.c_[wave,-wave]-.43)**2,axis=0)
        pose = np.zeros((len(t),17,3))
        pose[:,15:17,0] = swing*np.array([-4.,4.])
        pose[:,15:17,1] = -2*swing-t[:,None]*4.
        valid = np.ones((len(t),17),bool)
        allowed = np.ones(len(t),bool)
        scale = np.full(len(t),100.)
        for zoom in (.25,3.):
            result = advancing_walk(t,pose*zoom,valid,scale*zoom,allowed,allowed)
            self.assertTrue(result[int(len(t)/2)])
        self.assertFalse(advancing_walk(t,pose,valid,scale,~allowed,allowed).any())
        pose[:,15:17,1] = 0.
        self.assertFalse(advancing_walk(t,pose,valid,scale,allowed,allowed).any())

    def test_motion_is_relative_to_own_reference(self):
        motion = np.r_[np.linspace(.05,.12,40), np.linspace(.5,.7,20)]
        ref = np.arange(len(motion)) < 40
        a = motion_above_own_baseline(motion, ref)
        b = motion_above_own_baseline(motion*5.0+.3, ref)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(a[ref].any())
        self.assertTrue(a[~ref].all())
        self.assertFalse(motion_above_own_baseline(motion, np.zeros_like(ref)).any())

    def test_continuity_cannot_bridge_conflicting_evidence_or_prolonged_occlusion(self):
        t = np.arange(180)/30.
        direct = ((t >= 1.) & (t < 2.)) | ((t >= 3.5) & (t < 4.5))
        compatible = (t >= 1.) & (t < 4.5)
        feet = np.ones(len(t), bool)
        self.assertTrue(continuity(t, direct, compatible, feet)[90])
        compatible[90] = False
        self.assertFalse(continuity(t, direct, compatible, feet)[90])
        compatible[90] = True
        feet[60:105] = False
        self.assertFalse(continuity(t, direct, compatible, feet)[90])

    def test_stair_promotion_needs_two_step_islands_and_torso_agreement(self):
        t = np.arange(0, 15., 1/30.)
        y = np.interp(t, [0.,1.,3.,4.,6.,7.,10.,11.,14.,14.9], [160,160,60,60,160,160,60,60,160,160])
        for islands, agree in ((2,True),(1,True),(2,False)):
            pose = np.zeros((len(t),17,3))
            pose[:,:,0] = 110
            pose[:,[5,6],1] = y[:,None]-35
            pose[:,[11,12],1] = y[:,None]+35
            if not agree:
                end = t >= 11
                pose[end,5:7,1] = 25
                pose[end,11:13,1] = 2*y[end,None]-25
            labels = np.full(len(t),'',object)
            evidence = np.full(len(t),np.nan)
            supported = np.zeros(len(t),bool)
            cases = [(1.,3.,'stairs_up',True),(4.,6.,'stairs_down',True),(7.,10.,'stairs_up',True),
                     (11.35,11.8,'stairs_down',False),(12.95,13.38,'stairs_down',False)]
            for a,b,tag,solid in cases[:3+islands]:
                mask = (t>=a) & (t<b)
                labels[mask], evidence[mask], supported[mask] = tag, .2, solid
            valid = np.ones((len(t),17), bool)
            extend_anchored_traversals(t, pose, valid, np.full(len(t),100.), valid[:,0], ~valid[:,0], labels, evidence, supported)
            descent = (t>=13.2) & (t<13.7)
            self.assertTrue((labels[descent]=='stairs_down').all())
            self.assertEqual(bool(supported[descent].all()), islands==2 and agree)
            self.assertFalse(supported[t>=14.7].any())

    def test_bracketed_turn_uses_individual_steps_instead_of_projected_motion(self):
        t = np.arange(0,6,1/30.)
        for pace in (.48,.82):
            swing = np.sin(np.pi*t/pace)
            speed = np.maximum(0.,np.c_[swing,-swing]-.38)**2
            pose = np.zeros((len(t),17,3))
            pose[:,15:17,0] = np.cumsum(speed,axis=0)*5
            gap = abs(t-4*pace) < .27*pace
            direct = ~gap
            compatible = np.ones(len(t),bool)
            observed = compatible.copy()
            with self.subTest(pace=pace):
                np.testing.assert_array_equal(bracketed_walk_gaps(t,pose,direct,compatible,observed),gap)
                self.assertFalse(bracketed_walk_gaps(t,np.zeros_like(pose),direct,compatible,observed).any())
                observed[gap] = False
                self.assertFalse(bracketed_walk_gaps(t,pose,direct,compatible,observed).any())
                observed[gap] = True
                compatible[np.argmin(abs(t-4*pace))] = False
                self.assertFalse(bracketed_walk_gaps(t,pose,direct,compatible,observed).any())

    def test_reference_fallback_uses_only_current_track_measurement_scale(self):
        values = np.r_[np.linspace(.10,.15,30), np.linspace(.60,.9,30)]
        reference = np.zeros(len(values),bool)
        a = motion_above_own_baseline(values,reference,allow_lower_tail=True)
        b = motion_above_own_baseline(8*values,reference,allow_lower_tail=True)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(a[30:].all())
        self.assertFalse(motion_above_own_baseline(np.ones(len(values)),reference,allow_lower_tail=True).any())

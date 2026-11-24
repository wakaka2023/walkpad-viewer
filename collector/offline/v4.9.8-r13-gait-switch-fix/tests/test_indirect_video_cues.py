import json
from types import SimpleNamespace
import unittest

import numpy as np

import grade_repetitions as grading
from indirect_video_cues import extract


def fixture(view='front_or_oblique',kind='normal',dt=.02):
    t=np.arange(0,5+dt/2,dt)
    phase=np.clip(np.minimum((t-1)/1.5,(4-t)/1.5),0,1)
    p=np.sin(phase*np.pi/2)**2
    if kind=='bottom':
        phase=np.clip(np.minimum(t-1,4-t),0,1)
        p=np.sin(phase*np.pi/2)**2
    shoulder=np.empty((len(t),2));hip=np.column_stack((-30*p,200+80*p))
    knee=np.column_stack((40*p,300+20*p));ankle=np.column_stack((np.zeros(len(t)),np.full(len(t),400.)))
    if view=='front_or_oblique':
        hip[:,0]=0;knee[:,0]=0
        fold=.7*p
        if kind=='early':fold=np.where(t<=2.5,.7*np.clip((t-1)/.3,0,1),.7*p)
        if kind=='late':fold=np.where(t>2.5,.7*np.minimum(1,p/.04),.7*p)
        if kind=='bottom':fold=.35*p+.3*np.clip((t-2.05)/.7,0,1)*np.clip((4-t)/.5,0,1)
        if kind=='ascent':fold=.35*p+.3*np.where(t>2.5,np.sin(np.clip((t-2.5)/1.5,0,1)*np.pi)**2,0)
        shoulder[:,0]=hip[:,0];shoulder[:,1]=hip[:,1]-120*(1-fold)
    else:
        lean=10+35*p
        if kind=='early':lean=np.where(t<=2.5,10+35*np.clip((t-1)/.3,0,1),10+35*p)
        if kind=='late':lean=np.where(t>2.5,10+35*np.minimum(1,p/.04),10+35*p)
        if kind=='bottom':lean=10+20*p+25*np.clip((t-2.05)/.7,0,1)*np.clip((4-t)/.5,0,1)
        if kind=='ascent':lean=10+20*p+25*np.where(t>2.5,np.sin(np.clip((t-2.5)/1.5,0,1)*np.pi)**2,0)
        shoulder[:,0]=hip[:,0]+120*np.sin(np.radians(lean))
        shoulder[:,1]=hip[:,1]-120*np.cos(np.radians(lean))
    body=np.stack((shoulder,hip,knee,ankle),axis=1)
    rep=SimpleNamespace(start_video_s=1.,end_video_s=4.,bottom_video_s=2.5)
    return t,rep,body,np.ones(len(t),bool),200.,view


class IndirectCueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cfg=grading.load_config()

    def run_case(self,view='front_or_oblique',kind='normal',**changes):
        t,rep,body,valid,scale,view=fixture(view,kind)
        for key,value in changes.items():
            if key=='body':body=value(t,body)
            elif key=='valid':valid=value(t,valid)
        return extract(t,rep,body,valid,scale,view,True,self.cfg)

    def kinds(self,result):return {x['kind'] for x in json.loads(result['indirect_cues_json'])}

    def test_coordinated_large_front_foreshortening_is_not_a_fault(self):
        r=self.run_case()
        self.assertEqual(r['indirect_assessment_available'],1)
        self.assertEqual(r['indirect_fault_level'],0)

    def test_coordinated_side_lean_is_not_a_fault(self):
        self.assertEqual(self.run_case('side_or_oblique')['indirect_fault_level'],0)

    def test_side_leaning_reference_uses_visible_segment_length(self):
        for initial in [65.,85.]:
            t,rep,body,valid,scale,v=fixture('side_or_oblique')
            p=(body[:,1,1]-200)/80
            lean=np.radians(initial+5*p)
            body[:,0]=body[:,1]+120*np.column_stack((np.sin(lean),-np.cos(lean)))
            result=extract(t,rep,body,valid,scale,v,True,self.cfg)
            self.assertEqual(result['indirect_assessment_available'],1)
            self.assertEqual(result['indirect_fault_level'],0)

    def test_front_compressed_reference_remains_unavailable(self):
        t,rep,body,valid,scale,v=fixture()
        body[:,0,1]=body[:,1,1]-.3*(body[:,1,1]-body[:,0,1])
        result=extract(t,rep,body,valid,scale,v,True,self.cfg)
        self.assertEqual(result['indirect_assessment_available'],0)

    def test_side_horizontal_reference_does_not_require_vertical_normalization(self):
        t,rep,body,valid,scale,v=fixture('side_or_oblique','ascent')
        body[:,0,1]=body[:,1,1]
        body[:,0,0]=body[:,1,0]+120
        result=extract(t,rep,body,valid,scale,v,True,self.cfg)
        self.assertEqual(result['indirect_assessment_available'],1)
        self.assertEqual(result['indirect_fault_level'],0)

    def test_front_early_fold_has_visible_phase_evidence(self):
        self.assertIn('early_trunk_fold',self.kinds(self.run_case(kind='early')))

    def test_front_late_recovery_has_visible_phase_evidence(self):
        self.assertIn('late_trunk_recovery',self.kinds(self.run_case(kind='late')))

    def test_side_early_fold_requires_leg_phase(self):
        self.assertIn('early_trunk_fold',self.kinds(self.run_case('side_or_oblique','early')))

    def test_side_late_recovery_requires_leg_phase(self):
        self.assertIn('late_trunk_recovery',self.kinds(self.run_case('side_or_oblique','late')))

    def test_front_bottom_fold_has_a_previous_matched_leg_frame(self):
        r=self.run_case(kind='bottom')
        self.assertIn('bottom_extra_fold',self.kinds(r))
        for e in json.loads(r['indirect_cues_json']):
            if e['kind']=='bottom_extra_fold':
                self.assertLess(e['reference_s'],e['center_s'])
                self.assertLessEqual(e['leg_match_error_leg_ratio'],self.cfg['indirect_leg_match_tolerance'])

    def test_side_bottom_fold_has_corresponding_angle_change(self):
        self.assertIn('bottom_extra_fold',self.kinds(self.run_case('side_or_oblique','bottom')))

    def test_front_ascent_comparison_matches_leg_posture(self):
        self.assertIn('ascent_extra_fold',self.kinds(self.run_case(kind='ascent')))

    def test_side_ascent_comparison_matches_leg_posture(self):
        self.assertIn('ascent_extra_fold',self.kinds(self.run_case('side_or_oblique','ascent')))

    def test_indirect_cues_never_claim_severe_anatomical_fault(self):
        for view in ['front_or_oblique','side_or_oblique']:
            r=self.run_case(view,'bottom')
            self.assertEqual(r['indirect_fault_level'],1)
            for e in json.loads(r['indirect_cues_json']):
                self.assertEqual(e['level'],1)
                self.assertIn('unconfirmed',e['interpretation'])

    def test_translation_and_scale_do_not_change_flags(self):
        for view in ['front_or_oblique','side_or_oblique']:
            t,rep,body,valid,scale,v=fixture(view,'late')
            a=extract(t,rep,body,valid,scale,v,True,self.cfg)
            b=extract(t,rep,body*2.3+np.array([510.,80.]),valid,scale*2.3,v,True,self.cfg)
            self.assertEqual(self.kinds(a),self.kinds(b))

    def test_left_right_mirror_does_not_change_side_flags(self):
        t,rep,body,valid,scale,v=fixture('side_or_oblique','ascent')
        a=extract(t,rep,body,valid,scale,v,True,self.cfg)
        body[:,:,0]*=-1
        b=extract(t,rep,body,valid,scale,v,True,self.cfg)
        self.assertEqual(self.kinds(a),self.kinds(b))

    def test_normal_motion_across_frame_rates(self):
        for dt in [.01,.02,.04]:
            for view in ['front_or_oblique','side_or_oblique']:
                t,rep,body,valid,scale,v=fixture(view,dt=dt)
                self.assertEqual(extract(t,rep,body,valid,scale,v,True,self.cfg)['indirect_fault_level'],0)

    def test_normal_motion_across_action_speeds(self):
        for speed in [.6,2.]:
            t,rep,body,valid,scale,v=fixture()
            rep=SimpleNamespace(start_video_s=rep.start_video_s*speed,end_video_s=rep.end_video_s*speed,bottom_video_s=rep.bottom_video_s*speed)
            self.assertEqual(extract(t*speed,rep,body,valid,scale,v,True,self.cfg)['indirect_fault_level'],0)

    def test_isolated_front_keypoint_spike_is_not_a_cue(self):
        def spike(t,body):
            body[np.argmin(abs(t-1.35)),0,1]+=90
            return body
        self.assertEqual(self.run_case(body=spike)['indirect_fault_level'],0)

    def test_standing_after_interval_is_not_scored(self):
        def change(t,body):
            body[t>4.1,0,1]+=110
            return body
        self.assertEqual(self.run_case(body=change)['indirect_fault_level'],0)

    def test_unknown_or_turning_view_does_not_produce_cues(self):
        t,rep,body,valid,scale,v=fixture(kind='late')
        for view,stable in [('oblique_or_uncertain',True),(v,False)]:
            r=extract(t,rep,body,valid,scale,view,stable,self.cfg)
            self.assertEqual(r['indirect_assessment_available'],0)
            self.assertEqual(r['indirect_cues_json'],'[]')

    def test_missing_points_are_not_reported_as_pass(self):
        def gaps(t,valid):
            valid[(t>2)&(t<2.5)]=False
            return valid
        self.assertEqual(self.run_case(valid=gaps)['indirect_assessment_available'],0)

    def test_missing_initial_reference_is_reported(self):
        t,rep,body,valid,scale,v=fixture()
        keep=t>=1
        r=extract(t[keep]-1,SimpleNamespace(start_video_s=0.,end_video_s=3.,bottom_video_s=1.5),body[keep],valid[keep],scale,v,True,self.cfg)
        self.assertEqual(r['indirect_assessment_available'],0)

    def test_cue_times_and_frames_are_inside_repetition(self):
        r=self.run_case(kind='bottom')
        for e in json.loads(r['indirect_cues_json']):
            self.assertGreaterEqual(e['start_s'],1.)
            self.assertLessEqual(e['stop_s'],4.)
            self.assertLessEqual(e['start_s'],e['center_s'])
            self.assertLess(e['center_s'],e['stop_s'])

    def test_side_projection_length_failure_is_not_reported_as_pass(self):
        def shrink(t,body):
            use=(t>1.5)&(t<3.5)
            body[use,0]=body[use,1]+.4*(body[use,0]-body[use,1])
            return body
        self.assertEqual(self.run_case('side_or_oblique',body=shrink)['indirect_assessment_available'],0)


if __name__=='__main__':unittest.main()

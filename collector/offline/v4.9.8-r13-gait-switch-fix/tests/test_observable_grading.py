import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import grade_repetitions as g
import individual_quality as q
from test_individual_quality import row, pose_fixture, TemporaryDirectory


class ObservableGradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=g.load_config()

    def decide(self,**changes):
        return q.decide(row(**changes),self.cfg)

    def test_complete_front_can_pass_without_manual_anatomical_review(self):
        r=self.decide(imu_samples=0)
        self.assertEqual(r['candidate_grade_code'],3)
        self.assertEqual(r['grading_basis'],'VIDEO_ONLY')
        self.assertEqual(r['observable_criteria_count'],6)
        self.assertEqual(r['critical_evidence_missing_count'],0)
        self.assertNotIn('direct evidence unavailable',r['grade_reasons'])

    def test_side_has_a_different_observable_checklist(self):
        r=self.decide(video_view_hint='side_or_oblique')
        c=json.loads(r['observable_criteria_json'])
        self.assertEqual(set(c),{'completion','foot_motion','depth','hip_knee_coordination','trunk_control','indirect_control'})
        self.assertEqual(r['candidate_grade_code'],3)

    def test_front_does_not_score_side_depth_or_hip_angle(self):
        r=self.decide(video_depth_deficit_thigh_ratio=10.,video_hip_shoulder_ascent_lag_leg_ratio=10.)
        self.assertEqual(r['candidate_grade_code'],3)
        self.assertNotIn('depth',json.loads(r['observable_criteria_json']))

    def test_side_does_not_score_occluded_far_leg_tracking(self):
        r=self.decide(video_view_hint='side_or_oblique',bilateral_pose_coverage=0.,
                      video_knee_inward_change_leg_ratio=1.,video_trunk_lateral_change_deg=80.)
        self.assertEqual(r['candidate_grade_code'],3)

    def test_missing_actual_frontal_joint_keeps_observation_limit(self):
        r=self.decide(video_knee_inward_change_leg_ratio=np.nan)
        self.assertEqual(r['candidate_grade_code'],2)
        self.assertEqual(r['grade_provisional'],1)
        self.assertEqual(r['observable_criteria_count'],5)

    def test_low_actual_pose_coverage_is_not_good(self):
        r=self.decide(video_pose_coverage=.6)
        self.assertEqual(r['candidate_grade_code'],2)
        self.assertGreater(r['critical_evidence_missing_count'],0)

    def test_long_front_landmark_gap_disables_view_specific_criteria(self):
        r=self.decide(bilateral_largest_gap_s=.4)
        self.assertEqual(r['candidate_grade_code'],2)
        c=json.loads(r['observable_criteria_json'])
        self.assertEqual(c['knee_tracking_proxy']['status'],'UNAVAILABLE')

    def test_long_side_landmark_gap_disables_depth(self):
        r=self.decide(video_view_hint='side_or_oblique',side_leg_largest_gap_s=.4)
        self.assertEqual(r['candidate_grade_code'],2)
        self.assertEqual(json.loads(r['observable_criteria_json'])['depth']['status'],'UNAVAILABLE')

    def test_turning_view_does_not_get_confident_pass(self):
        self.assertEqual(self.decide(video_view_stable=0)['candidate_grade_code'],2)

    def test_unknown_view_does_not_pass_on_only_common_criteria(self):
        self.assertEqual(self.decide(video_view_hint='oblique_or_uncertain')['candidate_grade_code'],2)

    def test_front_minor_tracking_issue_is_mid(self):
        self.assertEqual(self.decide(video_knee_inward_change_leg_ratio=.15)['candidate_grade_code'],2)

    def test_front_major_tracking_issue_is_poor(self):
        self.assertEqual(self.decide(video_knee_inward_change_leg_ratio=.25)['candidate_grade_code'],1)

    def test_severe_lateral_motion_is_poor(self):
        self.assertEqual(self.decide(video_trunk_lateral_change_deg=30.)['candidate_grade_code'],1)

    def test_severe_return_failure_is_not_averaged_away(self):
        self.assertEqual(self.decide(video_return_error_fraction=.5)['candidate_grade_code'],1)

    def test_static_front_torso_compression_is_only_context(self):
        r=self.decide(video_torso_combined_projection_ratio=.2)
        self.assertEqual(r['candidate_grade_code'],3)
        self.assertIn('context only',r['video_reasons'])

    def test_side_static_lean_is_only_context(self):
        r=self.decide(video_view_hint='side_or_oblique',video_trunk_shin_difference_deg=70.)
        self.assertEqual(r['candidate_grade_code'],3)

    def test_frontal_dissociation_is_not_an_anatomical_major_fault(self):
        r=self.decide(video_control_dissociation_level=2)
        self.assertEqual(r['candidate_grade_code'],2)
        self.assertTrue(np.isnan(r['lumbar_control_fault_level']))

    def test_dynamic_side_dissociation_can_be_severe(self):
        r=self.decide(video_view_hint='side_or_oblique',video_control_dissociation_level=2)
        self.assertEqual(r['candidate_grade_code'],1)

    def test_partial_imu_coverage_uses_the_same_video_grade(self):
        r=self.decide(imu_time_coverage=.2,joint_video_imu_evidence_level=2)
        self.assertEqual(r['candidate_grade_code'],3)
        self.assertEqual(r['grading_basis'],'VIDEO_ONLY')

    def test_missing_fusion_details_do_not_crash_valid_imu_metrics(self):
        r=self.decide(joint_video_imu_evidence_level=np.nan,imu_asymmetry_supported_by_video=np.nan)
        self.assertEqual(r['candidate_grade_code'],3)
        self.assertEqual(r['fusion_evidence_level'],0)

    def test_nan_view_stability_is_not_a_valid_view(self):
        self.assertEqual(self.decide(video_view_stable=np.nan)['candidate_grade_code'],2)

    def test_nan_joint_observation_is_not_a_pass(self):
        r=self.decide(video_joint_motion_observed=np.nan)
        self.assertEqual(r['candidate_grade_code'],2)

    def test_observable_criteria_json_has_explicit_measures_and_thresholds(self):
        c=json.loads(self.decide()['observable_criteria_json'])
        self.assertTrue(all({'status','level','value','unit','warn','severe'}<=set(x) for x in c.values()))
        self.assertTrue(all(x['status']=='PASS' for x in c.values()))

    def test_video_only_run_writes_grade_tables_and_local_video_link(self):
        with TemporaryDirectory() as d:
            p=Path(d);t,k,reps=pose_fixture()
            reps['start_video_frame']=31;reps['end_video_frame']=120
            reps.to_csv(p/'intervals.csv',index=False)
            np.savez(p/'keypoints.npz',time_s=t,keypoints=k)
            (p/'metadata.json').write_text(json.dumps({'parameters':{'view':'front','side':'auto'}}))
            pd.DataFrame({'rep_id':np.where((t>=.6)&(t<2.4),1,0)}).to_csv(p/'frame_labels.csv',index=False)
            report,grades=g.run(p,None,p/'grading',mapped=False)
            self.assertEqual(report['grading_basis_counts'],{'VIDEO_ONLY':1})
            self.assertEqual(grades.candidate_grade_code.tolist(),[3])
            self.assertFalse(report['missing_imu_lowers_video_grade'])
            self.assertTrue((p/'frame_candidate_grades.csv').is_file())
            self.assertIn('src="../review.mp4"',(p/'grading/review.html').read_text(encoding='utf-8'))

    def test_metadata_view_is_shared_with_boundary_detector(self):
        with TemporaryDirectory() as d:
            p=Path(d);t,k,reps=pose_fixture()
            np.savez(p/'keypoints.npz',time_s=t,keypoints=k)
            (p/'metadata.json').write_text(json.dumps({'parameters':{'side':'auto'},
                  'view_information':{'view':'side','view_uncertain':False}}))
            f=q.pose_features(p,reps,self.cfg)
            self.assertEqual(f.video_view_hint.tolist(),['side_or_oblique'])

    def test_frontal_foreshortening_does_not_turn_into_lateral_lean(self):
        with TemporaryDirectory() as d:
            p=Path(d);t,k,reps=pose_fixture(collapse=True)
            progress=np.maximum(0,1-np.abs(t-1.5)/.9)
            k[:,[5,6],1]+=40*progress[:,None]
            np.savez(p/'keypoints.npz',time_s=t,keypoints=k)
            (p/'metadata.json').write_text(json.dumps({'parameters':{'view':'front','side':'auto'}}))
            f=q.pose_features(p,reps,self.cfg).iloc[0]
            self.assertAlmostEqual(f.video_trunk_lateral_change_deg,0.,places=8)

    def test_indirect_cue_is_a_review_flag_not_anatomical_poor(self):
        r=self.decide(indirect_fault_level=2)
        self.assertEqual(r['candidate_grade_code'],2)
        self.assertEqual(r['indirect_review_required'],1)
        self.assertFalse(r['indirect_anatomical_diagnosis'])
        self.assertTrue(np.isnan(r['pelvis_control_fault_level']))
        self.assertTrue(np.isnan(r['lumbar_control_fault_level']))

    def test_indirect_flag_cannot_hide_a_direct_major_fault(self):
        self.assertEqual(self.decide(indirect_fault_level=1,video_knee_inward_change_leg_ratio=.3)['candidate_grade_code'],1)

    def test_missing_indirect_observation_is_not_a_pass(self):
        r=self.decide(indirect_assessment_available=0,indirect_fault_level=np.nan)
        self.assertEqual(r['candidate_grade_code'],2)
        self.assertEqual(json.loads(r['observable_criteria_json'])['indirect_control']['status'],'UNAVAILABLE')

    def test_indirect_flag_requires_supported_view(self):
        r=self.decide(indirect_fault_level=1,video_view_stable=0)
        self.assertEqual(r['indirect_review_required'],0)
        self.assertNotIn('Indirect coordination risk',r['grade_reasons'])

    def test_indirect_config_rejects_impossible_phases(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'settings.json'
            path.write_text(json.dumps({'indirect_near_top_progress':.9}))
            with self.assertRaises(ValueError):g.load_config(path)

    def test_indirect_config_rejects_reversed_length_bounds(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'settings.json'
            path.write_text(json.dumps({'indirect_side_length_ratio_min':1.1}))
            with self.assertRaises(ValueError):g.load_config(path)


if __name__=='__main__':
    unittest.main()

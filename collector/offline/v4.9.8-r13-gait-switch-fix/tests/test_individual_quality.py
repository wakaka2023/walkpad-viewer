from pathlib import Path
from contextlib import contextmanager
import uuid
import json
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grade_repetitions as g
import individual_quality as q

@contextmanager
def TemporaryDirectory():

    p=Path(__file__).resolve().parent/'_scratch'/uuid.uuid4().hex
    p.mkdir(parents=True)
    yield str(p)

def row(**changes):
    r = dict(rep_id=1, video_pose_coverage=1., video_largest_gap_s=.02,
             video_view_hint='front_or_oblique', video_view_stable=1,
             bilateral_pose_coverage=1., bilateral_largest_gap_s=.02,
             side_leg_pose_coverage=1., side_leg_largest_gap_s=.02,
             video_knee_inward_change_leg_ratio=0., video_trunk_lateral_change_deg=0.,
             video_depth_deficit_thigh_ratio=-.1, video_hip_shoulder_ascent_lag_leg_ratio=.01,
             video_joint_motion_observed=1, video_control_dissociation_level=0,
             indirect_assessment_available=1, indirect_fault_level=0, indirect_cues_json='[]',
             video_return_error_fraction=.02, video_ankle_motion_leg_ratio=.01,
             imu_samples=60, imu_time_coverage=1., gyro_rms_dps=5.,
             angular_acceleration_rms_dps2=30., acc_magnitude_sd_g=.005,
             left_gyro_rms_dps=5., right_gyro_rms_dps=5.,
             bilateral_gyro_rms_abs_diff_dps=0., bilateral_gyro_rms_relative_diff_pct=0.)
    r.update(changes)
    return r

def reviewed(**changes):
    items = {k: 0 for k in q.REVIEW_ITEMS}
    items.update(changes)
    return dict(items=items, reviewer='TEST_ONLY', note='Synthetic observed checklist, not a real human grade')

def pose_fixture(collapse=False):
    t = np.arange(151)*.02
    progress = np.maximum(0, 1-np.abs(t-1.5)/.9)
    k = np.zeros((len(t), 17, 3))
    k[:, :, 2] = .99
    for joint, xy in {5:(-35,20),6:(35,20),11:(-20,80),12:(20,80),
                      13:(-25,120),14:(25,120),15:(-30,180),16:(30,180)}.items():
        k[:, joint, :2] = xy
    k[:, [11,12], 1] += 40*progress[:, None]
    k[:, [5,6], 1] += (88 if collapse else 40)*progress[:, None]
    k[:, [13,14], 1] += 20*progress[:, None]
    reps = pd.DataFrame([dict(rep_id=1,start_video_s=.6,end_video_s=2.4,bottom_video_s=1.5,
                              duration_s=1.8,hip_drop_leg_ratio=.4,ankle_motion_leg_ratio=0.,
                              bottom_video_frame=76)])
    return t, k, reps

class IndividualGradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = g.load_config()

    def test_no_good_from_absence_of_detected_faults(self):
        result = q.decide(row(video_joint_motion_observed=0), self.cfg)
        self.assertEqual(result['candidate_grade_code'], 2)
        self.assertGreater(result['critical_evidence_missing_count'], 0)
        self.assertIn('does not imply an observed technique fault', result['grade_reasons'])

    def test_positive_review_can_support_good(self):
        result = q.decide(row(), self.cfg, reviewed())
        self.assertEqual(result['candidate_grade_code'], 3)
        self.assertEqual(result['grade_provisional'], 0)

    def test_pelvic_fault_not_masked_by_quiet_feet(self):
        result = q.decide(row(gyro_rms_dps=.01, acc_magnitude_sd_g=0.), self.cfg,
                          reviewed(pelvis_control=2))
        self.assertEqual(result['candidate_grade_code'], 1)
        self.assertIn('Pelvic control', result['grade_reasons'])

    def test_lumbar_fault_can_be_minor_or_major(self):
        for level, grade in [(1,2), (2,1)]:
            r = q.decide(row(), self.cfg, reviewed(lumbar_control=level))
            self.assertEqual(r['candidate_grade_code'], grade)

    def test_one_major_fault_cannot_be_averaged_away(self):
        r = q.decide(row(video_view_hint='side_or_oblique',video_depth_deficit_thigh_ratio=.65), self.cfg, reviewed())
        self.assertEqual(r['candidate_grade_code'], 1)

    def test_torso_compression_is_not_pelvic_tilt_diagnosis(self):
        r = q.decide(row(video_torso_projection_ratio=.2,video_torso_to_shin_projection_ratio=.3), self.cfg)
        self.assertEqual(r['candidate_grade_code'],3)
        self.assertTrue(np.isnan(r['pelvis_control_fault_level']))
        self.assertIn('does not lower the grade alone',r['video_reasons'])
        self.assertEqual(r['video_fault_count'],0)

    def test_two_projection_ratios_must_agree(self):
        r=q.decide(row(video_torso_projection_ratio=.2,video_torso_to_shin_projection_ratio=.9),self.cfg)
        self.assertEqual(r['candidate_grade_code'],3)
        self.assertNotIn('forward lean or bending',r['grade_reasons'])

    def test_projection_evidence_has_to_cooccur_in_one_window(self):
        r=q.decide(row(video_torso_projection_ratio=.2,video_torso_to_shin_projection_ratio=.3,
                       video_torso_combined_projection_ratio=.9),self.cfg)
        self.assertEqual(r['candidate_grade_code'],3)
        self.assertNotIn('forward lean or bending',r['grade_reasons'])

    def test_imu_intensity_alone_is_not_a_quality_fault(self):
        r=q.decide(row(gyro_rms_dps=1000.,angular_acceleration_rms_dps2=100000.,acc_magnitude_sd_g=2.),self.cfg,reviewed())
        self.assertEqual(r['candidate_grade_code'],3)

    def test_bilateral_difference_is_support_not_loading(self):
        r=q.decide(row(left_gyro_rms_dps=10.,right_gyro_rms_dps=3.,
                       bilateral_gyro_rms_abs_diff_dps=7.,bilateral_gyro_rms_relative_diff_pct=107.69),self.cfg,reviewed())
        self.assertEqual(r['candidate_grade_code'],3)
        self.assertIn('not a load difference',r['imu_reasons'])

    def test_weak_rotation_does_not_create_asymmetry_fault(self):
        r=q.decide(row(left_gyro_rms_dps=.05,right_gyro_rms_dps=.01,
                       bilateral_gyro_rms_relative_diff_pct=133.),self.cfg,reviewed())
        self.assertEqual(r['candidate_grade_code'],3)

    def test_missing_imu_still_gets_one_of_three_grades(self):
        r=q.decide(row(imu_samples=0),self.cfg,reviewed())
        self.assertEqual(r['candidate_grade_code'],3)
        self.assertEqual(r['grade_provisional'],0)

    def test_hip_or_knee_first_alone_is_not_a_fault(self):
        for timing in [-.5,.5]:
            r=q.decide(row(hip_minus_knee_onset_s=timing),self.cfg,reviewed())
            self.assertEqual(r['candidate_grade_code'],3)
            self.assertIn('timing order alone does not lower the grade',r['video_reasons'])

    def test_hips_shooting_up_with_shoulders_lagging_is_a_fault(self):
        r=q.decide(row(video_view_hint='side_or_oblique',video_hip_shoulder_ascent_lag_leg_ratio=.2),self.cfg)
        self.assertEqual(r['candidate_grade_code'],1)
        self.assertIn('Hip-shoulder coordination',r['grade_reasons'])

    def test_other_repetitions_and_class_balance_do_not_change_result(self):
        target=row(video_view_hint='side_or_oblique',video_depth_deficit_thigh_ratio=.65)
        alone=q.assign(pd.DataFrame([target]),self.cfg)
        groups=[target]+[row(rep_id=i,video_view_hint='side_or_oblique',video_depth_deficit_thigh_ratio=.65) for i in range(2,8)]
        together=q.assign(pd.DataFrame(groups),self.cfg)
        self.assertEqual(together.candidate_grade_code.tolist(),[1]*7)
        pd.testing.assert_series_equal(alone.iloc[0],together.iloc[0])

    def test_good_single_rep_needs_no_reference_peers(self):
        result=q.assign(pd.DataFrame([row()]),self.cfg,{1:reviewed()})
        self.assertEqual(result.candidate_grade_code.tolist(),[3])
        self.assertEqual(result.reference_peer_count.tolist(),[0])

    def test_filename_like_metadata_not_used_for_grade(self):
        a=row(filename='correct_squat.mp4',expected_grade=3,wrist=100000,device_pitch=-999)
        b=row(filename='pelvic_error.mp4',expected_grade=1,wrist=-100000,device_pitch=999)
        self.assertEqual(q.decide(a,self.cfg),q.decide(b,self.cfg))

    def test_persistent_window_rejects_isolated_spike_and_gap(self):
        t=np.arange(100)*.02;x=np.zeros(100);x[50]=100.
        valid=np.ones(100,bool)
        self.assertEqual(q._persistent(t,x,valid,.12),0.)
        x[40:60]=10.
        self.assertEqual(q._persistent(t,x,valid,.12),10.)
        valid[:]=False;valid[40:43]=True
        self.assertTrue(np.isnan(q._persistent(t,x,valid,.12)))

    def test_frontal_pose_detects_compression_without_label_hints(self):
        for collapse, expected in [(False,3),(True,3)]:
            with TemporaryDirectory() as d:
                p=Path(d);t,k,reps=pose_fixture(collapse)
                np.savez(p/'keypoints.npz',time_s=t,keypoints=k)
                (p/'metadata.json').write_text(json.dumps({'parameters':{'side':'auto'}}))
                f=q.pose_features(p,reps,self.cfg).iloc[0].to_dict()
                self.assertEqual(f['video_view_hint'],'front_or_oblique')
                r=row();r.update(f)
                self.assertEqual(q.decide(r,self.cfg)['candidate_grade_code'],expected)

    def test_original_vector_metrics_and_physical_derivative(self):
        n=100;t=np.arange(n)*.02
        table=pd.DataFrame({'estimated_imu_time_s':t})
        for side in g.SIDES:
            table[g.ACC[side]]=np.c_[np.zeros(n),np.zeros(n),np.linspace(1,2,n)]
            table[g.GYRO[side]]=np.c_[2*t,np.zeros(n),np.zeros(n)]
        tt,sig=g.continuous_signals(table)
        m=g.window_metrics(tt,sig,.4,1.2)
        self.assertAlmostEqual(m['angular_acceleration_rms_dps2'],2.,places=9)
        self.assertAlmostEqual(m['gyro_rms_dps'],np.sqrt(np.mean((2*t[20:60])**2)),places=12)
        self.assertAlmostEqual(m['acc_magnitude_sd_g'],np.std(np.linspace(1,2,n)[20:60],ddof=1),places=12)
        self.assertEqual(m['bilateral_gyro_rms_abs_diff_dps'],0.)

    def test_review_validation_rejects_wrong_boundary_and_missing_provenance(self):
        reps=pd.DataFrame([dict(rep_id=1,start_video_frame=10,end_video_frame=50,start_video_s=.2,end_video_s=1.)])
        with TemporaryDirectory() as d:
            path=Path(d)/'review.csv';q.write_review_template(path,reps)
            self.assertEqual(q.read_reviews(path,reps)[1]['items'],{})
            tab=pd.read_csv(path).fillna('');tab.loc[0,'pelvis_control']='2'
            tab.to_csv(path,index=False)
            with self.assertRaises(ValueError):q.read_reviews(path,reps)
            tab.loc[0,'reviewer']='tester';tab.loc[0,'note']='Synthetic evidence'
            tab.loc[0,'end_video_frame']=51;tab.to_csv(path,index=False)
            with self.assertRaises(ValueError):q.read_reviews(path,reps)

    def test_wrong_pose_identity_cannot_supply_pass_evidence(self):
        reps=pd.DataFrame([dict(rep_id=1,start_video_frame=10,end_video_frame=50,start_video_s=.2,end_video_s=1.)])
        with TemporaryDirectory() as d:
            path=Path(d)/'review.csv';q.write_review_template(path,reps,'correct-hash')
            with self.assertRaises(ValueError):q.read_reviews(path,reps,'another-video-hash')

    def test_empty_table_has_valid_grade_columns(self):
        out=q.assign(pd.DataFrame({'rep_id':pd.Series(dtype=int)}),self.cfg)
        self.assertEqual(len(out),0)
        self.assertIn('candidate_grade_code',out)

if __name__=='__main__':
    unittest.main()

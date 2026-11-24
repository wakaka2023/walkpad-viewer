import json

import numpy as np


VERSION = 'observable-video-foot-v4.4'
LABELS = {3: 'GOOD', 2: 'MID', 1: 'POOR'}
FRONT = ('completion', 'foot_motion', 'knee_tracking_proxy', 'lateral_control', 'trunk_control', 'indirect_control')
SIDE = ('completion', 'foot_motion', 'depth', 'hip_knee_coordination', 'trunk_control', 'indirect_control')
NAMES = {'completion': 'Return to initial hip height', 'foot_motion': 'Visible ankle stability',
         'depth': 'Side projected depth', 'knee_tracking_proxy': 'Frontal knee tracking',
         'lateral_control': 'Frontal trunk lateral control', 'trunk_control': 'Trunk-leg coordination',
         'hip_knee_coordination': 'Hip-shoulder coordination during ascent',
         'indirect_control': 'Indirect phase-matched trunk-leg coordination cue'}


def decide(row, cfg, review=None):
    from individual_quality import REVIEW_ITEMS, ITEM_NAMES
    review = review or {'items': {}, 'reviewer': '', 'note': ''}
    criteria, faults, video_info, imu_info = {}, [], [], []
    gaps = []

    def finite(name, default=np.nan):
        value = row.get(name, default)
        return float(value) if value is not None and np.isfinite(value) else np.nan

    pose_ok = (finite('video_pose_coverage', 0) >= cfg['minimum_pose_coverage'] and
               finite('video_largest_gap_s', np.inf) <= cfg['maximum_pose_gap_s'])
    view = row.get('video_view_hint', 'oblique_or_uncertain')
    stable = finite('video_view_stable', 0) == 1
    expected = FRONT if view == 'front_or_oblique' else SIDE if view == 'side_or_oblique' else FRONT[:2]
    if not pose_ok:
        gaps.append('Insufficient observed body coverage')
    if view not in ('front_or_oblique', 'side_or_oblique') or not stable:
        gaps.append('A stable front or side view could not be established')

    def check(key, value, warn, severe, available, unit):
        available = bool(available and np.isfinite(value))
        level = (2 if value >= severe else 1 if value >= warn else 0) if available else None
        criteria[key] = {'status': ('PASS', 'WARN', 'FAIL')[level] if level is not None else 'UNAVAILABLE',
                         'level': level, 'value': float(value) if np.isfinite(value) else None,
                         'unit': unit, 'warn': float(warn), 'severe': float(severe)}
        if level:
            faults.append((key, level, NAMES[key]+f': {value:.3g} {unit}'))

    check('completion', finite('video_return_error_fraction'), cfg['return_error_warn_fraction'],
          cfg['return_error_severe_fraction'], pose_ok, 'excursion fraction')
    check('foot_motion', finite('video_ankle_motion_leg_ratio'), cfg['ankle_motion_warn_leg_ratio'],
          cfg['ankle_motion_severe_leg_ratio'], pose_ok, 'leg fraction')
    if view == 'front_or_oblique':
        front_ok = pose_ok and stable and finite('bilateral_pose_coverage', 0) >= cfg['minimum_pose_coverage']
        front_ok &= finite('bilateral_largest_gap_s', np.inf) <= cfg['maximum_pose_gap_s']
        check('knee_tracking_proxy', finite('video_knee_inward_change_leg_ratio'),
              cfg['knee_inward_warn_leg_ratio'], cfg['knee_inward_severe_leg_ratio'], front_ok, 'leg fraction')
        check('lateral_control', finite('video_trunk_lateral_change_deg'),
              cfg['trunk_lateral_warn_deg'], cfg['trunk_lateral_severe_deg'], front_ok, 'projected degrees')
        control_ok = front_ok
    elif view == 'side_or_oblique':
        side_ok = pose_ok and stable and finite('side_leg_pose_coverage', 0) >= cfg['minimum_pose_coverage']
        side_ok &= finite('side_leg_largest_gap_s', np.inf) <= cfg['maximum_pose_gap_s']
        check('depth', finite('video_depth_deficit_thigh_ratio'), cfg['depth_deficit_warn_thigh_ratio'],
              cfg['depth_deficit_severe_thigh_ratio'], side_ok, 'thigh fraction')
        check('hip_knee_coordination', finite('video_hip_shoulder_ascent_lag_leg_ratio'),
              cfg['hip_shoulder_lag_warn_leg_ratio'], cfg['hip_shoulder_lag_severe_leg_ratio'], side_ok, 'leg fraction')
        control_ok = side_ok
    else:
        control_ok = False
    if 'trunk_control' in expected:
        level = finite('video_control_dissociation_level')
        if view == 'front_or_oblique' and np.isfinite(level):
            level = min(level, 1)
        check('trunk_control', level, 1, 2,
              control_ok and finite('video_joint_motion_observed', 0) == 1, 'sustained event level')
        indirect=finite('indirect_fault_level')
        check('indirect_control', min(indirect,1) if np.isfinite(indirect) else np.nan, 1, 2,
              control_ok and finite('indirect_assessment_available',0)==1, 'review cue level')
        if criteria['indirect_control']['level']:
            cues=json.loads(row.get('indirect_cues_json','[]'))
            cue_text='; '.join(x['label']+f" ({x['start_s']:.3f}-{x['stop_s']:.3f}s; {x['value']:.3g} >= {x['threshold']:.3g} {x['unit']})" for x in cues)
            faults[-1]=('indirect_control',1,'Indirect coordination risk: '+(cue_text or 'sustained phase-matched motion cue')+'; anatomical cause unconfirmed')
    missing = [NAMES[k] for k in expected if criteria[k]['status'] == 'UNAVAILABLE']
    if missing:
        gaps.append('Unavailable in this view: '+', '.join(missing))

    difference = finite('video_trunk_shin_difference_deg')
    if np.isfinite(difference) and difference >= cfg['trunk_shin_warn_deg']:
        video_info.append(f'Trunk-shin projected difference {difference:.1f} degrees; static lean alone is not penalized')
    projection = finite('video_torso_combined_projection_ratio')
    if not np.isfinite(projection):
        a, b = finite('video_torso_projection_ratio'), finite('video_torso_to_shin_projection_ratio')
        projection = max(a, b) if np.isfinite(a) and np.isfinite(b) else np.nan
    if np.isfinite(projection) and projection < cfg['torso_projection_warn_ratio']:
        video_info.append(f'Torso projected length ratio {projection:.3f}; perspective/lean context only; does not lower the grade alone')
    timing = finite('hip_minus_knee_onset_s')
    if np.isfinite(timing) and abs(timing) >= cfg['joint_onset_lag_warn_s']:
        video_info.append(f'Projected hip-minus-knee onset {timing:.3f}s; timing order alone does not lower the grade')
    for key, level in review['items'].items():
        if level:
            faults.append(('manual_'+key, int(level), 'Optional video review: '+ITEM_NAMES[key]+', '+review['note']))
    video_faults = list(faults)
    metric_keys = ('gyro_rms_dps', 'angular_acceleration_rms_dps2', 'acc_magnitude_sd_g', 'bilateral_gyro_rms_abs_diff_dps')
    imu_ok = (finite('imu_samples', 0) >= cfg['minimum_imu_samples'] and
              finite('imu_time_coverage', 0) >= cfg['minimum_imu_coverage'] and
              all(np.isfinite(finite(k)) for k in metric_keys))
    fusion_level = asymmetry = 0
    if imu_ok:
        fusion = finite('joint_video_imu_evidence_level', 0)
        asym = finite('imu_asymmetry_supported_by_video', 0)
        fusion_level = int(fusion) if np.isfinite(fusion) else 0
        asymmetry = int(asym) if np.isfinite(asym) else 0
        if fusion_level and pose_ok:
            reasons = row.get('fusion_evidence_reasons', '') or 'Visible motion has time-coincident foot-metric support'
            faults.append(('imu_motion_support', fusion_level, str(reasons)))
        else:
            fusion_level = 0
        imu_info.append('Four foot metrics supplement time-coincident visible events; intensity alone is not penalized')
        pct = finite('bilateral_gyro_rms_relative_diff_pct')
        if np.isfinite(pct):
            imu_info.append(f'Bilateral gyro RMS relative difference: {pct:.1f}% (not a load difference)')
    else:
        imu_info.append('IMU unavailable or incomplete; using observable video criteria')
    levels = [level for _, level, _ in faults]
    major, minor = 2 in levels, 1 in levels
    code = 1 if major else 2 if minor or gaps else 3
    reasons = [text for _, _, text in faults]
    if gaps:
        reasons.append('Observation limits: '+'; '.join(gaps))
        if not major and not minor:
            reasons.append('MID indicates incomplete observation; does not imply an observed technique fault')
    if not reasons:
        reasons.append('No issue detected in '+str(len(expected))+' observable screening criteria; unmeasured anatomy is not cleared')
    observed_count = sum(criteria[k]['status'] != 'UNAVAILABLE' for k in expected)
    result = dict(candidate_grade_code=code, candidate_grade_zh=LABELS[code], candidate_grade_en=LABELS[code],
                  grade_reasons='; '.join(dict.fromkeys(reasons)),
                  video_reasons='; '.join([text for _, _, text in video_faults]+video_info),
                  imu_reasons='; '.join(imu_info), quality_reasons='; '.join(gaps),
                  grade_provisional=int(bool(gaps)), grade_confidence='limited_observation' if gaps else 'observable_rule_supported',
                  video_evidence_level=max([level for _, level, _ in video_faults] or [0]),
                  imu_asymmetry_evidence_level=asymmetry, fusion_evidence_level=fusion_level,
                  video_fault_count=len(video_faults), critical_evidence_missing_count=len(missing)+int(not pose_ok)+int(not stable or view not in ('front_or_oblique', 'side_or_oblique')),
                  manual_video_items_used=','.join(review['items']), video_reviewer=review['reviewer'],
                  reference_peer_count=0, reference_rep_ids='', other_repetitions_used_for_grade=0,
                  pelvis_control_fault_level=review['items'].get('pelvis_control', np.nan),
                  lumbar_control_fault_level=review['items'].get('lumbar_control', np.nan),
                  grading_basis='VIDEO_PLUS_IMU' if imu_ok else 'VIDEO_ONLY',
                  observable_criteria_count=observed_count, expected_observable_criteria_count=len(expected),
                  observable_criteria_json=json.dumps(criteria, allow_nan=False),
                  automatic_scope='view_specific_2d_motion_and_indirect_coordination_screening',
                  indirect_review_required=int(finite('indirect_fault_level',0)>0 and control_ok),
                  indirect_anatomical_diagnosis=False,
                  excluded_automatic_items='pelvic_tilt,lumbar_curvature,heel_contact,load_distribution')
    result.update({'criterion_'+key+'_level': criteria[key]['level'] for key in criteria})
    for key in REVIEW_ITEMS:
        result.setdefault('criterion_'+key+'_level', review['items'].get(key, np.nan))
    return result

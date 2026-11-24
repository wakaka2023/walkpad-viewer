from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

VERSION = 'observable-video-foot-v4.4'
LABELS = ENGLISH = {3: 'GOOD', 2: 'MID', 1: 'POOR'}
REVIEW_ITEMS = ('depth', 'trunk_control', 'pelvis_control', 'lumbar_control',
                'hip_knee_coordination', 'knee_tracking', 'foot_contact', 'completion')
ITEM_NAMES = dict(zip(REVIEW_ITEMS, ('Depth', 'Trunk control', 'Pelvic control', 'Lumbar control',
                               'Hip-knee coordination', 'Knee tracking', 'Foot contact', 'Completion')))
SOURCES = [
    {'publisher': 'Ultralytics', 'title': 'Pose estimation',
     'url': 'https://docs.ultralytics.com/tasks/pose/',
     'use': 'COCO-17 landmark coverage. View-specific numeric screening thresholds are configurable engineering rules, not an official validated scale.'},
]


def _angle(a, b, c):
    u, v = a-b, c-b
    denom = np.linalg.norm(u, axis=-1)*np.linalg.norm(v, axis=-1)
    return np.degrees(np.arccos(np.clip(np.sum(u*v, axis=-1)/np.maximum(denom, 1e-9), -1, 1)))

def _median(values, mask):
    x = np.asarray(values)[mask]
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan

def _persistent(t, values, mask, seconds, *, minimum=False, with_frame=False):
    from grading_pose_features import runs
    x = np.asarray(values)
    valid = mask & np.isfinite(x)
    candidates = []
    dt = float(np.median(np.diff(t)))
    for lo, hi in runs(valid):
        for i in range(lo, hi):
            j = int(np.searchsorted(t, t[i]+seconds, side='left'))
            if j >= hi or j-i < 2:
                continue
            if np.max(np.diff(t[i:j+1])) > max(.1, 3*dt):
                continue
            candidates.append((float(np.median(x[i:j+1])), (i+j)//2+1))
    selected = (min if minimum else max)(candidates, key=lambda pair: pair[0]) if candidates else (np.nan, None)
    return selected if with_frame else selected[0]

def _onset(t, signal, start, stop, cfg):
    span = (t >= start) & (t <= stop) & np.isfinite(signal)
    top = (t >= start) & (t <= min(start+.12, stop)) & np.isfinite(signal)
    base = _median(signal, top)
    if span.sum() < 6 or not np.isfinite(base):
        return np.nan
    end = _median(signal, span & (t >= stop-.12))
    change = end-base
    if abs(change) < cfg['joint_onset_min_excursion_deg']:
        return np.nan
    progress = (signal-base)/change
    test = span & (progress >= cfg['joint_onset_fraction'])
    from grading_pose_features import runs
    for lo, hi in runs(test):
        if t[hi-1]-t[lo] >= cfg['evidence_persistence_s']:
            return float(t[lo])
    return np.nan

def pose_features(video_dir, intervals, cfg):
    from grading_pose_features import features, smooth_valid
    root = Path(video_dir)
    if intervals.empty:
        return pd.DataFrame({'rep_id': pd.Series(dtype=int)})
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    params = meta.get('parameters', {})
    with np.load(root/'keypoints.npz', allow_pickle=False) as cache:
        f = features(cache, SimpleNamespace(keypoint_conf=cfg['keypoint_confidence'],
                                          smooth_seconds=.15, side=params.get('side', 'auto')))
        k = np.asarray(cache['keypoints'], float).copy()
    t, scale = f['t'], f['scale']
    geometry = f['geometry_metadata']
    side_hint = geometry['view_hint'] == 'side_or_oblique'
    front_hint = geometry['view_hint'] == 'front_or_oblique'

    if geometry['geometry_mode'] == 'right_visible':
        ids = [6, 12, 14, 16]
    else:
        options = [[5, 11, 13, 15], [6, 12, 14, 16]]
        ids = max(options, key=lambda ix: int((k[:, ix, 2] >= cfg['keypoint_confidence']).all(axis=1).sum()))
    leg_good = np.isfinite(k[:, ids]).all(axis=(1, 2)) & (k[:, ids, 2] >= cfg['keypoint_confidence']).all(axis=1)
    leg, _ = smooth_valid(t, k[:, ids, :2], leg_good, .15)
    shoulder, hip, knee, ankle = [leg[:, i] for i in range(4)]
    knee_angle = _angle(hip, knee, ankle)

    trunk_thigh_angle = _angle(shoulder, hip, knee)
    thigh_length = np.linalg.norm(hip-knee, axis=1)
    deficit = (knee[:, 1]-hip[:, 1])/np.maximum(thigh_length, 1.)
    trunk = shoulder-hip
    shin = knee-ankle
    lean = np.degrees(np.arctan2(np.abs(trunk[:, 0]), -trunk[:, 1]))
    shin_lean = np.degrees(np.arctan2(np.abs(shin[:, 0]), -shin[:, 1]))
    difference = np.abs(lean-shin_lean)
    torso_ids = [5, 6, 11, 12, 13, 14, 15, 16]
    both_good = np.isfinite(k[:, torso_ids]).all(axis=(1, 2)) & (k[:, torso_ids, 2] >= cfg['keypoint_confidence']).all(axis=1)
    both, _ = smooth_valid(t, k[:, torso_ids, :2], both_good, .15)
    sh = both[:, :2].mean(axis=1)
    hp = both[:, 2:4].mean(axis=1)
    kn = both[:, 4:6]
    an = both[:, 6:8]
    torso_height = hp[:, 1]-sh[:, 1]
    shin_height = (an[:, :, 1]-kn[:, :, 1]).mean(axis=1)
    knee_width = np.abs(kn[:, 0, 0]-kn[:, 1, 0])
    ankle_width = np.abs(an[:, 0, 0]-an[:, 1, 0])
    rows = []
    for e in intervals.itertuples(index=False):
        active = (t >= e.start_video_s) & (t < e.end_video_s)

        shoulder_valid = (k[:, [5, 6], 2] >= cfg['keypoint_confidence']).all(axis=1)
        width = np.linalg.norm(k[:, 5, :2]-k[:, 6, :2], axis=1)/scale
        local_width = _median(width, active & shoulder_valid)
        known = meta.get('view_information', {})
        declared = params.get('view', 'auto')
        selected = declared if declared in ('front', 'side') else known.get('view') if not known.get('view_uncertain', False) else None
        view = selected+'_or_oblique' if selected in ('front', 'side') else 'side_or_oblique' if local_width < .25 else 'front_or_oblique' if local_width > .35 else 'oblique_or_uncertain' 
        side_hint, front_hint = view == 'side_or_oblique', view == 'front_or_oblique'
        view_mask = active & shoulder_valid & np.isfinite(width)
        change = float(np.quantile(width[view_mask], .95)-np.quantile(width[view_mask], .05)) if view_mask.sum() >= 3 else np.nan
        view_stable = bool(change <= cfg['maximum_shoulder_width_change_leg_ratio']) if np.isfinite(change) else selected == 'side' 
        side_hint, front_hint = side_hint and view_stable, front_hint and view_stable
        valid = active & f['valid'] & np.isfinite(f['height'])

        pre = (t >= e.start_video_s-.3) & (t <= e.start_video_s+.03)
        bottom = active & (np.abs(t-e.bottom_video_s) <= max(.12, .08*e.duration_s))
        first, last = valid & (t <= e.start_video_s+.12), valid & (t >= e.end_video_s-.12)
        amplitude = float(e.hip_drop_leg_ratio)
        before, after = _median(f['height'], first), _median(f['height'], last)
        ret = abs(after-before)/scale/max(amplitude, 1e-6)
        gaps = np.diff(np.r_[e.start_video_s, t[valid], e.end_video_s])
        row = {'rep_id': int(e.rep_id), 'video_pose_coverage': float(valid.sum()/max(1, active.sum())),
               'video_largest_gap_s': float(max(gaps)), 'video_return_error_fraction': float(ret),
               'video_hip_drop_leg_ratio': amplitude, 'video_ankle_motion_leg_ratio': float(e.ankle_motion_leg_ratio),
               'video_view_hint': view, 'video_geometry_mode': geometry['geometry_mode'],
               'video_view_stable': int(view_stable), 'video_shoulder_width_change_leg_ratio': change,
               'side_leg_pose_coverage': float((active & leg_good).sum()/max(1, active.sum())),
               'bilateral_pose_coverage': float((active & both_good).sum()/max(1, active.sum())),
                'side_leg_largest_gap_s': float(np.max(np.diff(np.r_[e.start_video_s, t[active & leg_good], e.end_video_s]))),
                'bilateral_largest_gap_s': float(np.max(np.diff(np.r_[e.start_video_s, t[active & both_good], e.end_video_s]))),
               'video_depth_deficit_thigh_ratio': np.nan, 'video_trunk_shin_difference_deg': np.nan,
               'video_torso_projection_ratio': np.nan, 'video_torso_to_shin_projection_ratio': np.nan,
               'video_torso_combined_projection_ratio': np.nan,
               'video_knee_inward_change_leg_ratio': np.nan, 'video_trunk_lateral_change_deg': np.nan,
               'hip_onset_video_s': np.nan, 'knee_onset_video_s': np.nan, 'hip_minus_knee_onset_s': np.nan,
               'video_hip_shoulder_ascent_lag_leg_ratio': np.nan, 'automatic_pelvis_observed': 0,
               'automatic_lumbar_observed': 0, 'automatic_heel_contact_observed': 0,
               'video_motion_events_json': '[]', 'video_control_dissociation_level': 0,
               'video_control_dissociation_value': np.nan, 'video_joint_motion_observed': 0,
               'video_evidence_frame': int(e.bottom_video_frame)}
        events = []
        from motion_fusion import joint_video_events
        top_height = _median(f['height'], pre & f['valid'])
        progress = (f['height']-top_height)/max(amplitude*scale, 1.)
        if side_hint and row['side_leg_pose_coverage'] >= cfg['minimum_pose_coverage']:
            row['video_depth_deficit_thigh_ratio'] = _median(deficit, bottom & leg_good)
            row['video_trunk_shin_difference_deg'] = _persistent(t, difference, active & leg_good, cfg['evidence_persistence_s'])
            hip_on = _onset(t, trunk_thigh_angle, e.start_video_s-.3, e.bottom_video_s, cfg)
            knee_on = _onset(t, knee_angle, e.start_video_s-.3, e.bottom_video_s, cfg)
            row.update(hip_onset_video_s=hip_on, knee_onset_video_s=knee_on, hip_minus_knee_onset_s=hip_on-knee_on)
            row['trunk_thigh_onset_video_s'] = hip_on

            early_up = active & leg_good & (t >= e.bottom_video_s) & (t <= e.bottom_video_s+.4*e.duration_s)
            at_bottom = bottom & leg_good
            hy, sy = _median(hip[:, 1], at_bottom), _median(shoulder[:, 1], at_bottom)
            lag = ((hy-hip[:, 1])-(sy-shoulder[:, 1]))/scale
            knee_at_bottom = _median(knee_angle, at_bottom)

            extending = knee_angle-knee_at_bottom >= cfg['ascent_minimum_knee_extension_deg']
            row['video_hip_shoulder_ascent_lag_leg_ratio'] = _persistent(t, lag, early_up & extending, cfg['evidence_persistence_s'])
            events = joint_video_events(t, active, leg_good, hip, knee, ankle, lean, scale,
                                         e.bottom_video_s, progress, True, cfg)
            row['video_joint_motion_observed'] = 1
        if front_hint and row['bilateral_pose_coverage'] >= cfg['minimum_pose_coverage']:
            base_torso = _median(torso_height, pre & both_good)
            base_shin = _median(shin_height, pre & both_good)

            if base_torso > cfg['minimum_top_torso_leg_ratio']*scale and base_shin > .1*scale:
                ratio = torso_height/base_torso
                adjusted = ratio/np.maximum(shin_height/base_shin, .15)
                row['video_torso_projection_ratio'] = _persistent(t, ratio, active & both_good, cfg['evidence_persistence_s'], minimum=True)
                row['video_torso_to_shin_projection_ratio'] = _persistent(t, adjusted, active & both_good, cfg['evidence_persistence_s'], minimum=True)
                combined, evidence_frame = _persistent(t, np.maximum(ratio,adjusted), active & both_good,
                                                       cfg['evidence_persistence_s'], minimum=True, with_frame=True)
                row['video_torso_combined_projection_ratio'] = combined
                if evidence_frame is not None:
                    row['video_evidence_frame'] = evidence_frame
                events = joint_video_events(t, active, both_good, hp, kn.mean(axis=1), an.mean(axis=1),
                                             ratio, scale, e.bottom_video_s, progress, False, cfg)
                row['video_joint_motion_observed'] = 1
            base_width = _median(knee_width/np.maximum(ankle_width, 1.), pre & both_good)
            inward = (base_width*ankle_width-knee_width)/scale
            row['video_knee_inward_change_leg_ratio'] = _persistent(t, inward, active & both_good, cfg['evidence_persistence_s'])
            if base_torso > cfg['minimum_top_torso_leg_ratio']*scale:
                base_offset = _median(sh[:, 0]-hp[:, 0], pre & both_good)
                lateral = np.degrees(np.arctan2(np.abs((sh[:, 0]-hp[:, 0])-base_offset), base_torso))
                row['video_trunk_lateral_change_deg'] = _persistent(t, lateral, active & both_good, cfg['evidence_persistence_s'])
                row['video_lateral_measure'] = 'horizontal_trunk_offset_change_over_upright_torso_height'
        if events:
            dissociation = next((v for v in events if v['kind'] == 'trunk_leg_dissociation'), None)
            if dissociation:
                row['video_control_dissociation_level'] = dissociation['level']
                row['video_control_dissociation_value'] = dissociation['value']
            row['video_motion_events_json'] = json.dumps(events, ensure_ascii=False)
            chosen = dissociation or events[0]
            row['video_evidence_frame'] = int(np.argmin(abs(t-chosen['center_s']))+1)
        from indirect_video_cues import extract
        body=np.stack((sh,hp,kn.mean(axis=1),an.mean(axis=1)),axis=1) if view=='front_or_oblique' else leg
        body_valid=both_good if view=='front_or_oblique' else leg_good
        row.update(extract(t,e,body,body_valid,scale,view,view_stable,cfg))
        row['indirect_body_ids_json']=json.dumps([[5,6],[11,12],[13,14],[15,16]] if view=='front_or_oblique' else [[x] for x in ids])
        cues=json.loads(row['indirect_cues_json'])
        if cues:
            row['video_evidence_frame']=max(cues,key=lambda x:x['value']/x['threshold'])['frame_1based']
        rows.append(row)
    return pd.DataFrame(rows)

def read_reviews(path, reps, pose_sha256=None):
    if path is None:
        return {}
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'rep_id', 'start_video_frame', 'end_video_frame', 'reviewer', 'note', *REVIEW_ITEMS}
    if pose_sha256:
        required.add('pose_cache_sha256')
    if not required.issubset(table.columns):
        raise ValueError('Review CSV missing columns: '+str(sorted(required-set(table.columns))))
    expected = reps.set_index('rep_id')
    seen = {}
    for raw in table.to_dict('records'):
        rid = int(raw['rep_id'])
        if rid not in expected.index or rid in seen:
            raise ValueError('Unknown or duplicate review rep_id: '+str(rid))
        e = expected.loc[rid]
        if pose_sha256 and raw.get('pose_cache_sha256') != pose_sha256:
            raise ValueError('Review pose hash differs; review the matching video: '+str(rid))
        if int(raw['start_video_frame']) != int(e.start_video_frame) or int(raw['end_video_frame']) != int(e.end_video_frame):
            raise ValueError('Review boundaries differ; re-review this rep: '+str(rid))
        items = {}
        for key in REVIEW_ITEMS:
            text = raw[key].strip()
            if text:
                if text not in ('0', '1', '2'):
                    raise ValueError('Review item must be blank, 0, 1 or 2: '+key)
                items[key] = int(text)
        if items and (not raw['reviewer'].strip() or not raw['note'].strip()):
            raise ValueError('Observed video review needs reviewer and evidence note')
        seen[rid] = {'items': items, 'reviewer': raw['reviewer'], 'note': raw['note']}
    return seen

from observable_grading import decide

def assign(table, cfg, reviews=None):
    reviews = reviews or {}
    decisions = [decide(r, cfg, reviews.get(int(r['rep_id']))) for r in table.to_dict('records')]
    if decisions:
        return pd.concat([table.reset_index(drop=True), pd.DataFrame(decisions)], axis=1)
    out = table.copy()
    for name in ('candidate_grade_code', 'candidate_grade_zh', 'candidate_grade_en', 'grade_reasons',
                 'quality_reasons', 'video_reasons', 'imu_reasons', 'grade_provisional', 'grade_confidence'):
        out[name] = pd.Series(dtype=int if name == 'candidate_grade_code' else str)
    return out

def write_review_template(path, reps, pose_sha256=''):
    base = reps[['rep_id', 'start_video_frame', 'end_video_frame', 'start_video_s', 'end_video_s']].copy()
    for name in REVIEW_ITEMS+('reviewer', 'note'):
        base[name] = ''
    base['pose_cache_sha256'] = pose_sha256
    base.to_csv(path, index=False, encoding='utf-8-sig')

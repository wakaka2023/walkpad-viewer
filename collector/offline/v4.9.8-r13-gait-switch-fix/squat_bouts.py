import numpy as np

from action_rules import ACTIVITY_LABELS, cycle_events


SQUAT_FAMILY = {'squat', 'split_squat'}


def stable_recovery(f, active, start, stop, config):
    if stop <= start:
        return True
    if not active[start:stop].all() or not f['torso_visible'][start:stop].all():
        return False
    feet = f['points'][start:stop,15:17]
    both = f['raw_valid'][start:stop,15:17].all(axis=1) & np.isfinite(feet).all(axis=(1,2))
    speed = f['foot_speed'][start:stop]
    scale = f['torso_px'][start:stop]
    if both.mean() < .7 or np.isfinite(speed).mean() < .7 or not np.isfinite(scale).any():
        return False
    extent = np.quantile(feet[both], .9, axis=0)-np.quantile(feet[both], .1, axis=0)
    return bool(np.max(np.linalg.norm(extent,axis=1)) < .18*np.nanmedian(scale)
                and np.nanquantile(speed,.8) <= config['cycle_max_foot_speed'])


def normalize_squat_cycles(f, active, events, frames, config):
    previous = None
    for event in events:
        if event['activity'] not in SQUAT_FAMILY:
            previous = None
            continue
        event['stance_rule_activity'] = event['activity']
        event['activity'] = 'squat'
        event['activity_zh'] = ACTIVITY_LABELS['squat']
        start = event['start_video_frame']-1
        stop = event['end_video_frame']
        if previous is not None:
            gap = event['start_video_s']-previous['end_video_s']
            event['group_can_join_previous'] = 0 <= gap <= config['event_group_gap_s'] and stable_recovery(
                f, active, previous['end_video_frame'], start, config)
        for row in frames[start:stop]:
            row['activity'] = 'squat'
            row['activity_zh'] = ACTIVITY_LABELS['squat']
        previous = event


def signal_noise(values):
    diff = np.abs(np.diff(values))
    diff = diff[np.isfinite(diff)]
    return float(np.quantile(diff, .35)) if len(diff) else np.nan


def adaptive_cycle_recovery(f, events, frames, config):
    direct = [e for e in events if e['activity'] in SQUAT_FAMILY]
    knee = f['knee_flexion']
    scale = f['torso_px']
    ankles = f['points'][:, 15:17, 1]
    unit = np.nanmedian(scale) if np.isfinite(scale).any() else np.nan
    leg = (f['hip'][:, 1] - ankles.mean(axis=1)) / unit
    hip_noise = signal_noise(f['hip'][:, 1] / unit)
    observed = f['raw_valid'][:, [11, 12, 13, 14, 15, 16]]
    quiet = f['foot_speed'][np.isfinite(f['foot_speed'])]
    usable_knee = knee[np.isfinite(knee)]
    if min(len(quiet), len(usable_knee)) < 12:
        return events, []
    knee_noise, leg_noise = signal_noise(knee), signal_noise(leg)
    if not np.isfinite([knee_noise, leg_noise]).all():
        return events, []
    q10, q90 = np.quantile(usable_knee, [.1, .9])
    prominence = max(9 * knee_noise, .15 * (q90 - q10), np.finfo(float).eps)
    foot_baseline = np.quantile(quiet, .35)
    foot_spread = np.quantile(quiet, .60) - np.quantile(quiet, .15)
    motion_limit = max(3 * foot_baseline, foot_baseline + 3 * foot_spread)
    adapt = dict(config, squat_min_knee_flexion_deg=-np.inf,
                 squat_min_knee_excursion_deg=prominence,
                 cycle_min_hip_drop=0., cycle_max_foot_speed=motion_limit)
    provisional, _ = cycle_events(f, adapt)
    rescued = []
    for e in provisional:
        a, k, b = (e[key] - 1 for key in (
            'start_video_frame', 'peak_video_frame', 'end_exclusive_video_frame'))
        if any(a < p['end_video_frame'] and b >= p['start_video_frame'] - 1 for p in direct):
            continue
        if not np.isfinite(leg[[a, k, b]]).all() or observed[a:b+1].all(axis=1).mean() < .72:
            continue
        drop = min(leg[k] - leg[a], leg[k] - leg[b])
        camera_drop = e['camera_compensated_hip_drop_torso_ratio']
        if drop <= max(12 * leg_noise, .28 * camera_drop) or camera_drop <= 12 * hip_noise:
            continue
        hip_y = f['hip'][[a, k, b], 1, None]
        reach = hip_y - ankles[[a, k, b]]
        if np.min(reach[1] - reach[[0, 2]]) <= 0:
            continue
        e['recovery_basis'] = 'same_track_planted_feet_and_relative_hip_knee_cycle'
        e['reference_rep_count'] = len(direct)
        e['hip_to_foot_drop_torso_ratio'] = float(drop)
        e['knee_prominence_reference_deg'] = float(prominence)
        rescued.append(e)
    if len(rescued) >= 3:
        reference = np.quantile([e['hip_to_foot_drop_torso_ratio'] for e in rescued], .65)
        rescued = [e for e in rescued if e['hip_to_foot_drop_torso_ratio'] >= .34 * reference]
    if not rescued:
        return events, []
    displaced = [e for e in events if e['activity'] not in SQUAT_FAMILY and any(
        r['start_video_s'] < e['end_video_s'] and r['end_video_s'] > e['start_video_s'] for r in rescued)]
    kept = sorted([e for e in events if e not in displaced] + rescued, key=lambda e: e['start_video_s'])
    for e in displaced:
        for row in frames[e['start_video_frame']-1:e['end_video_frame']]:
            row['action_event_id'] = 0
    for eid, e in enumerate(kept, 1):
        e['event_id'] = eid
        for row in frames[e['start_video_frame']-1:e['end_video_frame']]:
            row['action_event_id'] = eid
            if 'recovery_basis' in e:
                row['activity_reason'] = e['recovery_basis']
    return kept, displaced
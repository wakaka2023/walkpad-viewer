import json

import numpy as np

def sustained_event(t, values, valid, threshold, persistence, kind, severe=None):
    from grading_pose_features import runs
    values = np.asarray(values, float)
    hit = valid & np.isfinite(values) & (values >= threshold)
    dt = float(np.median(np.diff(t)))
    candidates = []
    for a, b in runs(hit):
        cuts = np.flatnonzero(np.diff(t[a:b]) > max(.1, 3*dt))+a+1
        bounds = [a]+cuts.tolist()+[b]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi-lo < 3 or t[hi-1]-t[lo] < persistence:
                continue
            value = float(np.median(values[lo:hi]))
            candidates.append(dict(kind=kind, start_s=float(t[lo]), stop_s=float(t[hi-1]+dt),
                                   center_s=float(t[(lo+hi-1)//2]), value=value,
                                   frame_1based=int((lo+hi-1)//2+1),
                                   level=2 if severe is not None and value >= severe else 1))
    return max(candidates, key=lambda e: e['value']) if candidates else None

def joint_video_events(t, active, valid, hip, knee, ankle, trunk_signal, scale,
                       bottom_s, progress, side_view, cfg):
    from grading_pose_features import runs
    n = len(t)
    j = np.searchsorted(t, t+cfg['motion_comparison_window_s'])
    safe = np.minimum(j, n-1)
    block = np.full(n, -1, int)
    dt = float(np.median(np.diff(t)))
    for number, (a, b) in enumerate(runs(valid & active)):
        cuts = np.flatnonzero(np.diff(t[a:b]) > max(.1, 3*dt))+a+1
        bounds = [a]+cuts.tolist()+[b]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            block[lo:hi] = lo
    pair = (j < n) & (block >= 0) & (block == block[safe])
    pair &= (t[safe]-t <= cfg['motion_comparison_window_s']+max(.05, 2*dt))
    hip_relative, knee_relative = hip-ankle, knee-ankle
    leg_change = np.maximum(np.linalg.norm(hip_relative[safe]-hip_relative, axis=1),
                            np.linalg.norm(knee_relative[safe]-knee_relative, axis=1))/scale
    torso_change = np.abs(trunk_signal[safe]-trunk_signal)
    near_bottom = (progress >= cfg['dissociation_minimum_depth_progress']) & (progress[safe] >= cfg['dissociation_minimum_depth_progress'])
    quiet_leg = leg_change <= cfg['dissociation_maximum_leg_change_ratio']
    prefix = 'side' if side_view else 'front'
    event = sustained_event(t, torso_change, pair & near_bottom & quiet_leg,
                            cfg[prefix+'_dissociation_warn'], cfg['evidence_persistence_s'],
                            'trunk_leg_dissociation', cfg[prefix+'_dissociation_severe'])
    events = []
    if event:

        event['stop_s'] = min(float(t[np.flatnonzero(active)[-1]]+dt),
                              event['stop_s']+cfg['motion_comparison_window_s'])
        event['center_s'] = (event['start_s']+event['stop_s'])/2
        event['unit'] = 'projected_degrees' if side_view else 'upright_torso_fraction'
        events.append(event)
    foot_change = np.linalg.norm(ankle[safe]-ankle, axis=1)/scale
    foot = sustained_event(t, foot_change, pair, cfg['local_foot_displacement_leg_ratio'],
                           cfg['evidence_persistence_s'], 'foot_displacement')
    if foot:
        foot['stop_s'] = min(float(t[np.flatnonzero(active)[-1]]+dt),
                             foot['stop_s']+cfg['motion_comparison_window_s'])
        foot['center_s'] = (foot['start_s']+foot['stop_s'])/2
        foot['unit'] = 'projected_leg_fraction'
        events.append(foot)
    return events

def _masked_metrics(signals, mask):
    out = {}
    n = int(mask.sum())
    for side in ('left', 'right'):
        s = signals[side]
        out[side+'_gyro_rms_dps'] = float(np.sqrt(np.mean(np.sum(s['gyro'][mask]**2, axis=1)))) if n else np.nan
        out[side+'_angular_acceleration_rms_dps2'] = float(np.sqrt(np.mean(np.sum(s['alpha'][mask]**2, axis=1)))) if n else np.nan
        out[side+'_acc_magnitude_sd_g'] = float(np.std(s['acc_magnitude'][mask], ddof=1)) if n >= 2 else np.nan
    for key in ('gyro_rms_dps', 'angular_acceleration_rms_dps2', 'acc_magnitude_sd_g'):
        out[key] = float(np.hypot(out['left_'+key], out['right_'+key])/np.sqrt(2))
    left, right = out['left_gyro_rms_dps'], out['right_gyro_rms_dps']
    out['bilateral_gyro_rms_abs_diff_dps'] = abs(left-right)
    out['bilateral_gyro_rms_relative_diff_pct'] = 200*abs(left-right)/(left+right) if left+right > 1e-12 else np.nan
    return out

def _event_contrast(t, signals, rep_start, rep_stop, event, shift, cfg):

    start = max(rep_start, event['start_s'])+shift
    stop = min(rep_stop, event['stop_s'])+shift
    minimum = cfg['imu_event_minimum_duration_s']
    if stop-start < minimum:
        center = (start+stop)/2
        start = max(rep_start+shift, center-minimum/2)
        stop = min(rep_stop+shift, center+minimum/2)
    mask = (t >= start) & (t < stop)
    background = (t >= rep_start+shift) & (t < rep_stop+shift)
    background &= (t < start-cfg['imu_event_background_guard_s']) | (t >= stop+cfg['imu_event_background_guard_s'])
    dt = float(np.median(np.diff(t)))
    coverage = max(0., min(stop, t[-1]+dt)-max(start, t[0]))/max(stop-start, dt)
    if mask.sum() < cfg['imu_event_minimum_samples'] or background.sum() < cfg['imu_event_minimum_samples'] or coverage < cfg['minimum_imu_coverage']:
        return {'available': False, 'shift_s': float(shift), 'samples': int(mask.sum()),
                'background_samples': int(background.sum())}
    local, rest = _masked_metrics(signals, mask), _masked_metrics(signals, background)
    floors = {'gyro_rms_dps': cfg['imu_gyro_contrast_floor_dps'],
              'angular_acceleration_rms_dps2': cfg['imu_alpha_contrast_floor_dps2'],
              'acc_magnitude_sd_g': cfg['imu_acc_sd_contrast_floor_g']}
    votes = {}
    for key, floor in floors.items():
        votes[key] = bool(local[key] >= cfg['imu_event_contrast_ratio']*max(rest[key], floor)
                          and local[key]-rest[key] >= floor)
    strength = local['left_gyro_rms_dps']+local['right_gyro_rms_dps']
    asym = bool(strength >= 2*cfg['gyro_reliability_floor_dps']
                and local['bilateral_gyro_rms_relative_diff_pct'] >= cfg['bilateral_asymmetry_warn_pct'])
    return {'available': True, 'shift_s': float(shift), 'start_s': float(start), 'stop_s': float(stop),
            'samples': int(mask.sum()), 'background_samples': int(background.sum()),
            'metrics': local, 'same_rep_background_metrics': rest, 'transient_votes': votes,
            'burst': sum(votes.values()) >= 2, 'asymmetry': asym}

def imu_video_evidence(row, t, signals, cfg):
    events = json.loads(row.get('video_motion_events_json', '[]'))
    details, reasons = [], []
    level = asym_level = 0
    for event in events:
        tests = [_event_contrast(t, signals, row['start_video_s'], row['end_video_s'], event, shift, cfg)
                 for shift in (0., -cfg['sync_sensitivity_shift_s'], cfg['sync_sensitivity_shift_s'])]
        available = all(x['available'] for x in tests)
        burst = available and all(x['burst'] for x in tests)
        asym = available and all(x['asymmetry'] for x in tests)
        event_level = 0
        if event['kind'] == 'foot_displacement' and (burst or asym):
            event_level = 1
            asym_level = max(asym_level, int(asym))
            reasons.append('Local video ankle motion coincides with bilateral foot '+('multi-metric transients' if burst else 'rotation intensity differences')+'')
        elif event['kind'] == 'trunk_leg_dissociation' and burst:
            event_level = int(event['level'])
            reasons.append('Trunk-leg dissociation coincides with multi-metric foot transients (does not diagnose lumbar/pelvic faults)')
        level = max(level, event_level)
        details.append({'video_event': event, 'time_shift_tests': tests,
                        'burst_robust_to_tested_shifts': bool(burst),
                        'asymmetry_robust_to_tested_shifts': bool(asym), 'fusion_level': event_level})

    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list): return [clean(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    return {'joint_video_imu_evidence_level': level, 'imu_asymmetry_supported_by_video': asym_level,
            'fusion_event_count': len(details), 'fusion_evidence_reasons': '; '.join(dict.fromkeys(reasons)),
            'imu_event_details_json': json.dumps(clean(details), ensure_ascii=False, allow_nan=False),
            'fusion_uses_all_four_metrics': 1}
